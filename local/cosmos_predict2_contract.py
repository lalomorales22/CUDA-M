"""Opt-in Predict2 contract experiments, not NVIDIA or CUDA parity claims.

Sources are NVIDIA's initial-release commit, not a later sampler revision:
https://github.com/nvidia-cosmos/cosmos-predict2/tree/04f4823ae0572005c435c8300d20c2577aa4ef52
  cosmos_predict2/tokenizers/tokenizer.py:473-503 (posterior mean)
  cosmos_predict2/schedulers/rectified_flow_scheduler.py:47-127 (schedule)
  cosmos_predict2/functional/runge_kutta.py:54-117 (AB2 and Euler equations)
  cosmos_predict2/pipelines/video2world.py:891-928 (final clean prediction)

The installed Diffusers pipeline already converts x0 to (sample-x0)/sigma.
This adapter consumes that derivative and recovers x0; it must not be combined
with the diagnostic harness's older denoised_prediction_adapter.

For N pipeline evaluations this scheduler performs N-1 transitions followed
by one clean-prediction evaluation at sigma_min. NVIDIA's requested N sampling
steps instead mean N transitions PLUS one clean evaluation: request N+1 here
for matching evaluation counts. The clean pass is counted by the callback.

Differences remain: tensor arithmetic is FP32 (MPS cannot use FP64), although
schedule coefficients are calculated in Python FP64. This cannot undo earlier
BF16 x0 rounding inside Diffusers, nor fix geometry, RoPE, VAE kernels, or RNG
parity. Both contexts affect only the supplied instance, restore on exceptions,
and require exclusive use of that pipeline/VAE until their contexts exit.
"""
from contextlib import contextmanager
from functools import wraps
import math
from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput


NVIDIA_SOURCE_REVISION = "04f4823ae0572005c435c8300d20c2577aa4ef52"
_ABSENT = object()


class _PosteriorMode:
    def __init__(self, posterior):
        self._posterior = posterior

    def mode(self):
        return self._posterior.mode()

    def sample(self, generator=None):
        return self.mode()


@contextmanager
def experimental_posterior_mode(vae):
    """Use VAE posterior means without consuming the caller's random generator."""
    original = vae.encode
    instance_value = vars(vae).get("encode", _ABSENT)

    @wraps(original)
    def encode(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, AutoencoderKLOutput) or not callable(getattr(result.latent_dist, "mode", None)):
            raise TypeError("posterior-mode experiment requires an AutoencoderKLOutput with mode()")
        return AutoencoderKLOutput(latent_dist=_PosteriorMode(result.latent_dist))

    vae.encode = encode
    try:
        yield
    finally:
        if instance_value is _ABSENT:
            del vae.encode
        else:
            vae.encode = instance_value


class ExperimentalPredict2AB2Scheduler:
    """Bounded Diffusers adapter for the published Predict2 residual AB2 rule.

    ``order=1`` means one scheduler call per model evaluation in Diffusers;
    the numerical method after its first Euler transition is second order.
    Only the Predict2 pipeline's uniform 0..1 schedule placeholder is accepted.
    This is an inference experiment, not a serializable general scheduler.
    """

    order = 1
    solver_order = 2
    prediction_contract = "euler_derivative"

    def __init__(self, sigma_min=0.002, sigma_max=80.0, rho=7.0):
        if not all(math.isfinite(x) and x > 0 for x in (sigma_min, sigma_max, rho)) or sigma_min >= sigma_max:
            raise ValueError("AB2 needs finite positive sigma_min < sigma_max and rho")
        self.config = SimpleNamespace(sigma_min=sigma_min, sigma_max=sigma_max,
                                      sigma_data=1.0, final_sigmas_type="sigma_min")
        self.rho = rho
        self.timesteps = self.sigmas = None
        self.num_inference_steps = self.transition_steps = None
        self._step_index = 0
        self._previous_x0 = None
        self._coefficients = []

    @property
    def step_index(self):
        return self._step_index

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, timesteps=None):
        if timesteps is not None:
            raise ValueError("custom timesteps are unsupported by the Predict2 AB2 experiment")
        if sigmas is not None:
            if num_inference_steps is not None:
                raise ValueError("provide an evaluation count or the pipeline's sigma placeholder, not both")
            ramp = torch.as_tensor(sigmas, dtype=torch.float64, device="cpu")
            if ramp.ndim != 1 or len(ramp) < 2 or not torch.isfinite(ramp).all():
                raise ValueError("sigma placeholder must be a finite 1-D uniform 0..1 ramp")
            expected = torch.linspace(0, 1, len(ramp), dtype=torch.float64)
            if not torch.allclose(ramp, expected, rtol=0, atol=1e-7):
                raise ValueError("only the pipeline's uniform 0..1 sigma placeholder is supported")
            num_inference_steps = len(ramp)
        if (not isinstance(num_inference_steps, int) or isinstance(num_inference_steps, bool)
                or num_inference_steps < 2):
            raise ValueError("at least two model evaluations are needed: a transition and a clean pass")

        count = num_inference_steps
        hi = self.config.sigma_max ** (1 / self.rho)
        lo = self.config.sigma_min ** (1 / self.rho)
        schedule = [(hi + i / (count - 1) * (lo - hi)) ** self.rho for i in range(count)]
        coefficients = []
        for i in range(count - 1):
            ratio = schedule[i + 1] / schedule[i]
            if i == 0:
                coefficients.append((ratio, 1 - ratio, 0.0))
                continue
            # NVIDIA's h=t-s and c2=(m-s)/h use log-time t=-log(sigma).
            h = math.log(schedule[i] / schedule[i + 1])
            previous_h = math.log(schedule[i] / schedule[i - 1])
            if h <= 1e-6 or abs(previous_h) <= 1e-6:
                raise ValueError("AB2 log-time intervals are too small")
            c2 = previous_h / h
            phi1 = -math.expm1(-h) / h
            previous_weight = (1 - phi1) / c2
            coefficients.append((ratio, 1 - ratio - previous_weight, previous_weight))

        # The terminal duplicate accommodates the pipeline's existing sigma_min
        # assignment. step() extracts x0 at this entry instead of a dt=0 update.
        self.sigmas = torch.tensor(schedule + [schedule[-1]], dtype=torch.float32, device=device)
        self.timesteps = self.sigmas[:-1] * 1000
        if not torch.all(self.sigmas[:-2] > self.sigmas[1:-1]):
            raise ValueError("sigma schedule is not strictly descending in FP32")
        self.num_inference_steps = count
        self.transition_steps = count - 1
        self._coefficients = coefficients
        self._step_index = 0
        self._previous_x0 = None

    def step(self, model_output, timestep, sample, return_dict=True):
        if self.timesteps is None or self._step_index >= len(self.timesteps):
            raise ValueError("set_timesteps must precede an unconsumed scheduler step")
        index = self._step_index
        value = torch.as_tensor(timestep, device=self.timesteps.device)
        if value.ndim != 0 or not torch.equal(value, self.timesteps[index]):
            raise ValueError("AB2 steps must follow the scheduler timesteps in order")
        if (sample.dtype != torch.float32 or model_output.dtype != torch.float32
                or sample.shape != model_output.shape or sample.device != model_output.device
                or sample.device != self.sigmas.device):
            raise ValueError("AB2 requires matching FP32 sample and derivative tensors on the schedule device")
        if not torch.isfinite(sample).all() or not torch.isfinite(model_output).all():
            raise ValueError("AB2 sample and derivative must be finite")

        x0 = sample - self.sigmas[index] * model_output
        if index == self.transition_steps:
            result = x0
        else:
            a, b, c = self._coefficients[index]
            result = a * sample + b * x0
            if self._previous_x0 is not None:
                result = result + c * self._previous_x0
        if not torch.isfinite(result).all():
            raise ValueError("AB2 produced a non-finite state")
        self._previous_x0 = x0.detach()
        self._step_index += 1
        if not return_dict:
            return (result,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=result)


@contextmanager
def experimental_ab2_scheduler(pipeline):
    """Temporarily replace only this pipeline's compatible Euler scheduler."""
    original = pipeline.scheduler
    if not isinstance(original, FlowMatchEulerDiscreteScheduler):
        raise TypeError("AB2 experiment requires the baseline FlowMatchEulerDiscreteScheduler")
    config = original.config
    if (getattr(config, "sigma_data", None) != 1.0 or getattr(config, "sigma_min", None) != 0.002
            or getattr(config, "sigma_max", None) != 80.0
            or getattr(config, "final_sigmas_type", None) != "sigma_min"
            or config.stochastic_sampling or config.invert_sigmas or config.use_dynamic_shifting
            or not config.use_karras_sigmas or config.shift != 1.0 or config.shift_terminal is not None):
        raise ValueError("AB2 experiment requires the baseline Predict2 sigma/shift configuration")
    adapter = ExperimentalPredict2AB2Scheduler()
    pipeline.scheduler = adapter
    try:
        yield adapter
    finally:
        pipeline.scheduler = original
