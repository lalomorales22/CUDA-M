"""CPU-only, download-free checks for opt-in Predict2 contract experiments."""
import math
from types import SimpleNamespace
import unittest

import torch
from diffusers import (Cosmos2VideoToWorldPipeline, DiffusionPipeline,
                       FlowMatchEulerDiscreteScheduler, ModelMixin)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from cosmos_predict2_contract import (ExperimentalPredict2AB2Scheduler,
                                     experimental_ab2_scheduler, experimental_posterior_mode)


def baseline_scheduler():
    scheduler = FlowMatchEulerDiscreteScheduler(use_karras_sigmas=True)
    scheduler.register_to_config(sigma_min=0.002, sigma_max=80.0, sigma_data=1.0,
                                 final_sigmas_type="sigma_min")
    return scheduler


class FakeVae:
    def encode(self, value, return_dict=True):
        result = AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(value))
        return result if return_dict else (result.latent_dist,)


def reference_ab2(sample, x0, previous_x0, current, following, previous):
    """Independent literal FP64 form of NVIDIA initial-release runge_kutta.py."""
    sample, x0 = sample.double(), x0.double()
    if previous_x0 is None:
        return following / current * sample + (current - following) / current * x0
    h = -math.log(following) + math.log(current)
    c2 = (-math.log(previous) + math.log(current)) / h
    phi1 = math.expm1(-h) / -h
    phi2 = (phi1 - 1) / -h
    b1 = phi1 - phi2 / c2
    b2 = phi2 / c2
    return math.exp(-h) * sample + h * (b1 * x0 + b2 * previous_x0.double())


class PosteriorModeTest(unittest.TestCase):
    def setUp(self):
        self.vae = FakeVae()
        self.parameters = torch.tensor([[[[1.0]], [[2.0]], [[0.0]], [[0.0]]]])

    def test_mode_is_deterministic_and_does_not_consume_rng(self):
        generator = torch.Generator().manual_seed(35)
        initial_rng = generator.get_state().clone()
        with experimental_posterior_mode(self.vae):
            output = self.vae.encode(self.parameters)
            first = output.latent_dist.sample(generator)
            second = output.latent_dist.sample(generator)
        torch.testing.assert_close(first, self.parameters[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(initial_rng, generator.get_state()))

    def test_other_instances_and_existing_outputs_are_unchanged(self):
        other = FakeVae()
        prior = self.vae.encode(self.parameters)
        with experimental_posterior_mode(self.vae):
            self.assertIsInstance(prior.latent_dist, DiagonalGaussianDistribution)
            self.assertIsInstance(other.encode(self.parameters).latent_dist, DiagonalGaussianDistribution)
        self.assertIsInstance(self.vae.encode(self.parameters).latent_dist, DiagonalGaussianDistribution)
        self.assertNotIn("encode", vars(self.vae))

    def test_class_method_restored_after_exception(self):
        original = self.vae.encode
        with self.assertRaisesRegex(RuntimeError, "test"), experimental_posterior_mode(self.vae):
            raise RuntimeError("test")
        self.assertEqual(self.vae.encode, original)
        self.assertNotIn("encode", vars(self.vae))

    def test_instance_override_and_nested_contexts_restore(self):
        original = self.vae.encode
        self.vae.encode = original
        with experimental_posterior_mode(self.vae):
            outer = self.vae.encode
            with experimental_posterior_mode(self.vae):
                torch.testing.assert_close(self.vae.encode(self.parameters).latent_dist.sample(),
                                           self.parameters[:, :2])
            self.assertIs(self.vae.encode, outer)
        self.assertIs(vars(self.vae)["encode"], original)

    def test_unsupported_vae_output_fails_and_restores(self):
        with self.assertRaisesRegex(TypeError, "AutoencoderKLOutput"), experimental_posterior_mode(self.vae):
            self.vae.encode(self.parameters, return_dict=False)
        self.assertNotIn("encode", vars(self.vae))


class AB2Test(unittest.TestCase):
    def test_schedule_counts_and_initial_release_formula(self):
        for transitions in (1, 7, 35):
            with self.subTest(transitions=transitions):
                scheduler = ExperimentalPredict2AB2Scheduler()
                scheduler.set_timesteps(sigmas=torch.linspace(0, 1, transitions + 1))
                i = torch.arange(transitions + 1, dtype=torch.float64)
                reference = (80 ** (1 / 7) + i / transitions * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
                torch.testing.assert_close(scheduler.sigmas[:-1], reference.float(), rtol=0, atol=0)
                self.assertEqual(scheduler.transition_steps, transitions)
                self.assertEqual(len(scheduler.timesteps), transitions + 1)
                self.assertEqual(len(scheduler.sigmas), transitions + 2)
                self.assertEqual(float(scheduler.sigmas[-1]), float(scheduler.sigmas[-2]))

    def test_derivative_input_is_not_double_converted(self):
        scheduler = ExperimentalPredict2AB2Scheduler(sigma_min=1, sigma_max=4)
        scheduler.set_timesteps(2)
        sample, x0 = torch.tensor([9.0]), torch.tensor([1.0])
        derivative = (sample - x0) / scheduler.sigmas[0]
        result = scheduler.step(derivative, scheduler.timesteps[0], sample, return_dict=False)[0]
        torch.testing.assert_close(result, torch.tensor([3.0]), rtol=0, atol=0)

    def test_every_transition_matches_independent_fp64_reference(self):
        for evaluations in (3, 8, 36):
            with self.subTest(evaluations=evaluations):
                scheduler = ExperimentalPredict2AB2Scheduler()
                scheduler.set_timesteps(evaluations)
                count = evaluations - 1
                schedule = [(80 ** (1 / 7) + i / count * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
                            for i in range(evaluations)]
                sample = torch.tensor([[10.0, -3.0], [4.0, -2.0]])
                previous_x0 = None
                for i in range(count):
                    target = torch.tensor([[0.1 + i / 10, -0.3], [2.0, -1.0 - i / 20]])
                    derivative = (sample - target) / scheduler.sigmas[i]
                    recovered = sample - scheduler.sigmas[i] * derivative
                    expected = reference_ab2(sample, recovered, previous_x0, schedule[i],
                                             schedule[i + 1], schedule[i - 1] if i else None)
                    result = scheduler.step(derivative, scheduler.timesteps[i], sample).prev_sample
                    torch.testing.assert_close(result, expected.float(), rtol=2e-6, atol=2e-6)
                    previous_x0, sample = recovered, result

    def test_final_clean_pass_is_the_current_x0_not_a_noop(self):
        scheduler = ExperimentalPredict2AB2Scheduler()
        scheduler.set_timesteps(4)
        sample = torch.ones(2)
        for i, timestep in enumerate(scheduler.timesteps):
            target = torch.full_like(sample, 0.25 + i)
            derivative = (sample - target) / scheduler.sigmas[i]
            sample = scheduler.step(derivative, timestep, sample).prev_sample
        torch.testing.assert_close(sample, torch.full_like(sample, 3.25), rtol=0, atol=1e-6)
        self.assertEqual(scheduler.step_index, 4)
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            scheduler.step(torch.zeros(2), scheduler.timesteps[-1], sample)

    def test_reset_removes_prior_history(self):
        scheduler = ExperimentalPredict2AB2Scheduler()
        scheduler.set_timesteps(4)
        sample = torch.ones(1)
        first = scheduler.step(torch.zeros(1), scheduler.timesteps[0], sample).prev_sample
        scheduler.set_timesteps(4)
        self.assertEqual(scheduler.step_index, 0)
        self.assertIsNone(scheduler._previous_x0)
        again = scheduler.step(torch.zeros(1), scheduler.timesteps[0], sample).prev_sample
        torch.testing.assert_close(first, again, rtol=0, atol=0)

    def test_invalid_schedule_arguments_are_rejected(self):
        for arguments in ({}, {"num_inference_steps": 1}, {"num_inference_steps": True},
                          {"num_inference_steps": 3.5}, {"sigmas": [1, 0]}, {"sigmas": [0, 0.2, 1]},
                          {"sigmas": [[0, 1]]}, {"sigmas": [0, float("nan")]},
                          {"sigmas": [0, 1], "num_inference_steps": 2}, {"timesteps": [1, 2]}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ExperimentalPredict2AB2Scheduler().set_timesteps(**arguments)
        for arguments in ({"sigma_min": 0}, {"sigma_max": float("inf")}, {"rho": -1},
                          {"sigma_min": 80}, {"sigma_min": 81}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ExperimentalPredict2AB2Scheduler(**arguments)

    def test_wrong_timestep_shape_dtype_and_nonfinite_values_rejected(self):
        scheduler = ExperimentalPredict2AB2Scheduler()
        with self.assertRaises(ValueError):
            scheduler.step(torch.ones(1), 0, torch.ones(1))
        scheduler.set_timesteps(4)
        for timestep, derivative, sample in (
            (scheduler.timesteps[1], torch.ones(1), torch.ones(1)),
            (scheduler.timesteps[:1], torch.ones(1), torch.ones(1)),
            (scheduler.timesteps[0], torch.ones(2), torch.ones(1)),
            (scheduler.timesteps[0], torch.ones(1).half(), torch.ones(1)),
            (scheduler.timesteps[0], torch.ones(1), torch.ones(1).double()),
            (scheduler.timesteps[0], torch.tensor([float("nan")]), torch.ones(1)),
            (scheduler.timesteps[0], torch.ones(1), torch.tensor([float("inf")])),
        ):
            with self.subTest(timestep=timestep), self.assertRaises(ValueError):
                scheduler.step(derivative, timestep, sample)
            self.assertEqual(scheduler.step_index, 0)

    def test_context_restores_scheduler_and_pipeline_config_after_exception(self):
        pipeline = DiffusionPipeline()
        original = baseline_scheduler()
        pipeline.register_modules(scheduler=original)
        original_config = dict(pipeline.config)
        with self.assertRaisesRegex(RuntimeError, "test"), experimental_ab2_scheduler(pipeline) as adapter:
            self.assertIs(pipeline.scheduler, adapter)
            self.assertEqual(adapter.prediction_contract, "euler_derivative")
            raise RuntimeError("test")
        self.assertIs(pipeline.scheduler, original)
        self.assertEqual(dict(pipeline.config), original_config)

    def test_context_refuses_non_predict2_schedulers_without_mutation(self):
        pipeline = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler())
        original = pipeline.scheduler
        with self.assertRaises(ValueError), experimental_ab2_scheduler(pipeline):
            self.fail("incompatible scheduler accepted")
        self.assertIs(pipeline.scheduler, original)
        pipeline.scheduler = ExperimentalPredict2AB2Scheduler()
        with self.assertRaises(TypeError), experimental_ab2_scheduler(pipeline):
            self.fail("nested adapter accepted")

    def test_installed_pipeline_cpu_only_integration_with_fake_components(self):
        class Vae(ModelMixin):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.temperal_downsample = [False, True, True]
                self.config = SimpleNamespace(z_dim=16, latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

            def encode(self, video):
                shape = (video.shape[0], 32, (video.shape[2] - 1) // 4 + 1,
                         video.shape[3] // 8, video.shape[4] // 8)
                return AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(torch.zeros(shape)))

        class Transformer(ModelMixin):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(in_channels=17)
                self.calls = 0

            def forward(self, hidden_states, **kwargs):
                self.calls += 1
                self.assert_cpu(hidden_states)
                return (torch.zeros_like(hidden_states),)

            def assert_cpu(self, tensor):
                if tensor.device.type != "cpu":
                    raise AssertionError("test must never execute on a GPU")

        class SafetyDouble:
            def to(self, *args, **kwargs):
                return self

        vae, transformer = Vae(), Transformer()
        pipeline = Cosmos2VideoToWorldPipeline(tokenizer=None, text_encoder=None, vae=vae,
            transformer=transformer, scheduler=baseline_scheduler(), safety_checker=SafetyDouble())
        pipeline.set_progress_bar_config(disable=True)
        original = pipeline.scheduler
        callbacks = []

        def callback(_pipeline, step, timestep, tensors):
            callbacks.append(step)
            return tensors

        with torch.inference_mode(), experimental_posterior_mode(vae), experimental_ab2_scheduler(pipeline):
            result = pipeline(image=torch.zeros(1, 3, 16, 16), height=16, width=16,
                num_frames=5, num_inference_steps=2, fps=None, output_type="latent",
                prompt_embeds=torch.zeros(1, 2, 1024), negative_prompt_embeds=torch.zeros(1, 2, 1024),
                generator=torch.Generator().manual_seed(0), callback_on_step_end=callback)
        self.assertEqual(tuple(result.frames.shape), (1, 16, 2, 2, 2))
        self.assertEqual(result.frames.device.type, "cpu")
        self.assertTrue(torch.isfinite(result.frames).all())
        self.assertEqual(transformer.calls, 4)
        self.assertEqual(callbacks, [0, 1])
        self.assertIs(pipeline.scheduler, original)
        self.assertNotIn("encode", vars(vae))


if __name__ == "__main__":
    unittest.main()
