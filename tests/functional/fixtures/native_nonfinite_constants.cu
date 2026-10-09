#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

__global__ void nonfinite_constants(float* output) {
    output[0] = INFINITY;
    output[1] = -INFINITY;
    output[2] = __int_as_float(0x7fc00001);
    output[3] = -0.0f;
}

int main() {
    float* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), 4 * sizeof(float)) != cudaSuccess) return 1;
    nonfinite_constants<<<1, 1>>>(device);
    float output[4];
    const bool copied = cudaGetLastError() == cudaSuccess &&
        cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost) == cudaSuccess;
    const bool freed = cudaFree(device) == cudaSuccess;
    if (!copied || !freed) return 1;
    std::uint32_t actual[4];
    std::memcpy(actual, output, sizeof(actual));
    const std::uint32_t expected[4] = {0x7f800000, 0xff800000, 0x7fc00001, 0x80000000};
    for (int i = 0; i < 4; ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "FAIL: constant %d: %08x != %08x\n", i, actual[i], expected[i]);
            return 1;
        }
    }
    std::puts("PASS: non-finite constants and signed zero retain exact FP32 bits");
    return 0;
}
