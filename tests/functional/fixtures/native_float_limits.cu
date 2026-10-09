#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>

static_assert(FLT_MANT_DIG == 24 && DBL_MANT_DIG == 53, "IEEE floating limits");
static_assert(FLT_RADIX == 2 && FLT_MAX_EXP == 128 && DBL_MAX_EXP == 1024,
              "floating exponent limits");
static_assert(DBL_EPSILON == __DBL_EPSILON__ && DBL_MIN == __DBL_MIN__ &&
              DBL_MAX == __DBL_MAX__, "double limits remain available");

__global__ void float_limits(float* out) {
    out[0] = FLT_MAX;
    out[1] = FLT_MIN;
    out[2] = FLT_EPSILON;
}

int main() {
    float* out = nullptr;
    if (cudaMallocManaged(&out, 3 * sizeof(float)) != cudaSuccess) return 1;
    float_limits<<<1, 1>>>(out);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
        return 1;
    const bool correct = out[0] == FLT_MAX && out[1] == FLT_MIN && out[2] == FLT_EPSILON;
    if (cudaFree(out) != cudaSuccess || !correct) return 1;
    std::printf("PASS: CUDA umbrella floating limits in host and device code\n");
    return 0;
}
