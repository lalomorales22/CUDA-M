// Independent numerical checks using ordinary CUDA C++.
// Build: build/cumetalc local/apple_silicon_smoke.cu -o local/apple_silicon_smoke
// Run:   CUMETAL_TRACE_GPU=1 local/apple_silicon_smoke
// Require Apple-GPU launch provenance in the trace as well as the checks below.
// This fixture covers only these kernels and shapes, not general CUDA parity.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

// External linkage is intentional: native AOT registers kernels from another TU.
__global__ void smoke_saxpy(const float* x, const float* y, float* out,
                            float alpha, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = alpha * x[i] + y[i];
}

__global__ void smoke_matmul(const float* a, const float* b, float* out,
                             int rows, int inner, int cols) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row < rows && col < cols) {
        float sum = 0.0f;
        for (int k = 0; k < inner; ++k) {
            sum += a[row * inner + k] * b[k * cols + col];
        }
        out[row * cols + col] = sum;
    }
}

__global__ void smoke_block_reduce(const float* values, float* partials, int n) {
    __shared__ float scratch[256];
    const int lane = static_cast<int>(threadIdx.x);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    scratch[lane] = i < n ? values[i] : 0.0f;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        // Every thread, including inactive input lanes, reaches every barrier.
        __syncthreads();
    }
    if (lane == 0) partials[blockIdx.x] = scratch[0];
}

namespace {

constexpr int kThreads = 256;
constexpr int kTail = 512;
constexpr float kCanary = 12345.625f;
bool cleanup_failed = false;

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "FAIL: %s: %s (code %d)\n", operation,
                 cudaGetErrorString(status), static_cast<int>(status));
    return false;
}

// Early returns and host allocation exceptions also release device allocations.
// A cleanup error prevents the final success message.
class DeviceBuffer {
public:
    float* data = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { release(); }

    bool allocate(std::size_t count) {
        return check_cuda(cudaMalloc(reinterpret_cast<void**>(&data),
                                     count * sizeof(float)), "cudaMalloc");
    }

    bool release() {
        if (data == nullptr) return true;
        const cudaError_t status = cudaFree(data);
        data = nullptr;
        const bool ok = check_cuda(status, "cudaFree");
        if (!ok) cleanup_failed = true;
        return ok;
    }
};

bool upload(DeviceBuffer& device, const std::vector<float>& host) {
    return check_cuda(cudaMemcpy(device.data, host.data(),
                                 host.size() * sizeof(float),
                                 cudaMemcpyHostToDevice), "cudaMemcpy host->device");
}

bool download(std::vector<float>& host, const DeviceBuffer& device) {
    return check_cuda(cudaMemcpy(host.data(), device.data,
                                 host.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost), "cudaMemcpy device->host");
}

bool complete_launch(const char* kernel) {
    const bool launch_ok = check_cuda(cudaGetLastError(), kernel);
    // Synchronize even after a launch error before the buffers are released.
    const bool sync_ok = check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return launch_ok && sync_ok;
}

bool equal_finite(const char* label, std::size_t index, double actual,
                  double expected) {
    if (std::isfinite(actual) && std::isfinite(expected) && actual == expected) {
        return true;
    }
    std::fprintf(stderr, "FAIL: %s[%zu]: got %.17g, expected %.17g\n",
                 label, index, actual, expected);
    return false;
}

bool check_tail(const char* label, const std::vector<float>& output,
                std::size_t count) {
    for (std::size_t i = count; i < output.size(); ++i) {
        if (!equal_finite(label, i, output[i], kCanary)) return false;
    }
    return true;
}

bool run_saxpy(int n) {
    const std::size_t count = static_cast<std::size_t>(n) + kTail;
    std::vector<float> x(count, kCanary), y(count, kCanary), out(count, kCanary);
    const float alpha = -1.5f;
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>((i * 17 + 3) % 129 - 64) * 0.0625f;
        y[i] = static_cast<float>((i * 29 + 5) % 127 - 63) * 0.125f;
    }

    DeviceBuffer dx, dy, dout;
    if (!dx.allocate(count) || !dy.allocate(count) || !dout.allocate(count) ||
        !upload(dx, x) || !upload(dy, y) || !upload(dout, out)) return false;

    smoke_saxpy<<<(n + kThreads - 1) / kThreads, kThreads>>>(
        dx.data, dy.data, dout.data, alpha, n);
    if (!complete_launch("smoke_saxpy launch") || !download(out, dout)) return false;
    for (int i = 0; i < n; ++i) {
        const double expected = static_cast<double>(alpha) * x[i] + y[i];
        if (!equal_finite("saxpy", i, out[i], expected)) return false;
    }
    if (!check_tail("saxpy canary", out, n)) return false;
    // Evaluate all releases; one failure must not skip later cleanup.
    const bool x_freed = dx.release();
    const bool y_freed = dy.release();
    const bool out_freed = dout.release();
    if (!x_freed || !y_freed || !out_freed) return false;
    std::printf("PASS: saxpy n=%d, every output and %d tail canaries\n", n, kTail);
    return true;
}

bool run_matmul() {
    constexpr int rows = 37, inner = 29, cols = 41;
    constexpr int outputs = rows * cols;
    std::vector<float> a(rows * inner + kTail, kCanary);
    std::vector<float> b(inner * cols + kTail, kCanary);
    std::vector<float> out(outputs + kTail, kCanary);
    for (int i = 0; i < rows * inner; ++i) {
        a[i] = static_cast<float>((i * 13 + 7) % 65 - 32) * 0.125f;
    }
    for (int i = 0; i < inner * cols; ++i) {
        b[i] = static_cast<float>((i * 19 + 11) % 63 - 31) * 0.0625f;
    }

    DeviceBuffer da, db, dout;
    if (!da.allocate(a.size()) || !db.allocate(b.size()) ||
        !dout.allocate(out.size()) || !upload(da, a) || !upload(db, b) ||
        !upload(dout, out)) return false;

    const dim3 block(16, 16);
    const dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    smoke_matmul<<<grid, block>>>(da.data, db.data, dout.data, rows, inner, cols);
    if (!complete_launch("smoke_matmul launch") || !download(out, dout)) return false;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double expected = 0.0;
            for (int k = 0; k < inner; ++k) {
                expected += static_cast<double>(a[row * inner + k]) * b[k * cols + col];
            }
            const int index = row * cols + col;
            if (!equal_finite("matmul", index, out[index], expected)) return false;
        }
    }
    if (!check_tail("matmul canary", out, outputs)) return false;
    const bool a_freed = da.release();
    const bool b_freed = db.release();
    const bool out_freed = dout.release();
    if (!a_freed || !b_freed || !out_freed) return false;
    std::printf("PASS: matmul 37x29 * 29x41, all %d outputs, 2D blocks and tail canaries\n",
                outputs);
    return true;
}

bool run_reduction() {
    constexpr int n = 100003;
    constexpr int blocks = (n + kThreads - 1) / kThreads;
    std::vector<float> values(n + kTail, kCanary);
    std::vector<float> partials(blocks + kTail, kCanary);
    double expected_total = 0.0;
    for (int i = 0; i < n; ++i) {
        values[i] = static_cast<float>((i * 23 + 9) % 257 - 128) * 0.03125f;
        expected_total += values[i];
    }

    DeviceBuffer dvalues, dpartials;
    if (!dvalues.allocate(values.size()) || !dpartials.allocate(partials.size()) ||
        !upload(dvalues, values) || !upload(dpartials, partials)) return false;

    smoke_block_reduce<<<blocks, kThreads>>>(dvalues.data, dpartials.data, n);
    if (!complete_launch("smoke_block_reduce launch") || !download(partials, dpartials)) {
        return false;
    }
    double actual_total = 0.0;
    for (int block = 0; block < blocks; ++block) {
        double expected = 0.0;
        const int begin = block * kThreads;
        const int end = begin + kThreads < n ? begin + kThreads : n;
        for (int i = begin; i < end; ++i) expected += values[i];
        if (!equal_finite("reduction block", block, partials[block], expected)) return false;
        actual_total += partials[block];
    }
    if (!equal_finite("reduction total", 0, actual_total, expected_total) ||
        !check_tail("reduction canary", partials, blocks)) return false;
    const bool values_freed = dvalues.release();
    const bool partials_freed = dpartials.release();
    if (!values_freed || !partials_freed) return false;
    std::printf("PASS: reduction n=%d, all %d GPU block partials, CPU final sum=%.8f, tail canaries\n",
                n, blocks, actual_total);
    return true;
}

bool check_device() {
    int count = 0;
    if (!check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) return false;
    if (count < 1) {
        std::fprintf(stderr, "FAIL: no CUDA-visible device\n");
        return false;
    }
    if (!check_cuda(cudaSetDevice(0), "cudaSetDevice")) return false;
    cudaDeviceProp props{};
    if (!check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties")) return false;
    if (props.maxThreadsPerBlock < kThreads || props.maxThreadsDim[0] < kThreads ||
        props.maxThreadsDim[1] < 16 || props.maxGridSize[0] < 391 ||
        props.maxGridSize[1] < 3 || props.sharedMemPerBlock < kThreads * sizeof(float)) {
        std::fprintf(stderr, "FAIL: device limits cannot support the fixture launch sizes\n");
        return false;
    }
    std::printf("DEVICE: %s; warp=%d; max threads/block=%d; shared bytes/block=%zu\n",
                props.name, props.warpSize, props.maxThreadsPerBlock, props.sharedMemPerBlock);
    std::printf("CHECKS: signed dyadic inputs; exact finite CPU references; guarded output tails\n");
    return true;
}

} // namespace

int main() {
    try {
        // All intermediate values in these bounded cases are exactly representable
        // in FP32. Exact comparisons also hold if multiply-add is contracted.
        if (!check_device() || !run_saxpy(1) || !run_saxpy(257) ||
            !run_saxpy(100003) || !run_matmul() || !run_reduction() || cleanup_failed) {
            return 1;
        }
        std::printf("ALL CHECKS PASSED\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: host exception: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "FAIL: unknown host exception\n");
        return 1;
    }
}
