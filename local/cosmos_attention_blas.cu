// Experimental FP32 SDPA: CUDA-M cuBLAS/MPS GEMMs plus a CUDA-to-Metal softmax.
// Build: ./cuda-mac compile local/cosmos_attention_blas.cu -o local/out/cosmos-attention-blas
// This uses the repository's clean-room cublas_v2.h, not NVIDIA headers.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void cosmos_blas_softmax(float* scores, int tile_queries, int keys,
                                   int query_offset, int causal) {
    __shared__ float scratch[128];
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int query = query_offset + row % tile_queries;
    const int base = row * keys;
    float maximum = -INFINITY;
    for (int j = lane; j < keys; j += 128) {
        const float value = causal && j > query ? -INFINITY : scores[base + j];
        scores[base + j] = value;
        maximum = fmaxf(maximum, value);
    }
    scratch[lane] = maximum;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    maximum = scratch[0];
    // Do not overwrite the reduction before every lane has consumed its result.
    __syncthreads();
    float sum = 0.0f;
    for (int j = lane; j < keys; j += 128) {
        const float value = expf(scores[base + j] - maximum);
        scores[base + j] = value;
        sum += value;
    }
    scratch[lane] = sum;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        __syncthreads();
    }
    sum = scratch[0];
    for (int j = lane; j < keys; j += 128) scores[base + j] /= sum;
}

namespace {
constexpr std::uint64_t kWorkspaceElements = 64ULL * 1024 * 1024;
constexpr std::uint64_t kTensorElements = 128ULL * 1024 * 1024;
constexpr std::uint64_t kMultiplyAccumulates = 512ULL * 1024 * 1024 * 1024;

void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}

void check_blas(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS/MPS operation failed: " + std::to_string(int(code)));
}

struct Buffer {
    float* pointer = nullptr;
    explicit Buffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(float)));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (pointer) cudaFree(pointer); }
    void release() { check(cudaFree(pointer)); pointer = nullptr; }
};

struct BlasHandle {
    cublasHandle_t handle = nullptr;
    BlasHandle() { check_blas(cublasCreate(&handle)); }
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;
    ~BlasHandle() { if (handle) cublasDestroy(handle); }
    void release() { check_blas(cublasDestroy(handle)); handle = nullptr; }
};

std::vector<float> read_values(std::size_t count) {
    std::vector<float> values(count);
    if (std::fread(values.data(), sizeof(float), count, stdin) != count)
        throw std::runtime_error("truncated input");
    for (float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite input");
    return values;
}
}

int main(int argc, char** argv) {
    try {
        // --validate-only consumes the complete protocol but never touches CUDA.
        const bool validate_only = argc == 2 && std::strcmp(argv[1], "--validate-only") == 0;
        if (argc != 1 && !validate_only) throw std::runtime_error("unsupported argument");
        std::uint32_t header[8];
        float scale;
        if (std::fread(header, sizeof(header), 1, stdin) != 1 ||
            std::fread(&scale, sizeof(scale), 1, stdin) != 1)
            throw std::runtime_error("truncated header");
        if (header[0] != 0x434d4131 || !std::isfinite(scale) || header[7] > 1)
            throw std::runtime_error("invalid header");
        for (int i = 1; i <= 6; ++i)
            if (header[i] == 0 || header[i] > 16384)
                throw std::runtime_error("dimension out of range");
        const std::uint64_t heads = std::uint64_t(header[1]) * header[2];
        const int queries = header[3], keys = header[4], width = header[5], value_width = header[6];
        if (heads > 128 || width > 128 || value_width > 128)
            throw std::runtime_error("unsupported attention dimensions");
        const std::uint64_t rows = heads * queries;
        const std::uint64_t q_count = rows * width;
        const std::uint64_t k_count = heads * keys * width;
        const std::uint64_t v_count = heads * keys * value_width;
        const std::uint64_t output_count = rows * value_width;
        const std::uint64_t multiply_accumulates = rows * keys * (width + value_width);
        if (q_count + k_count + v_count + output_count > kTensorElements ||
            multiply_accumulates > kMultiplyAccumulates)
            throw std::runtime_error("attention exceeds bounded memory or operation budget");
        const int tile_limit = int(std::min<std::uint64_t>(256, kWorkspaceElements / (heads * keys)));
        const int tile_queries = std::min(queries, tile_limit);
        if (tile_queries < 1) throw std::runtime_error("unsupported score workspace");
        const std::uint64_t score_count = heads * tile_queries * keys;
        auto q = read_values(q_count);
        auto k = read_values(k_count);
        auto v = read_values(v_count);
        if (std::fgetc(stdin) != EOF || std::ferror(stdin))
            throw std::runtime_error("unexpected trailing input or input error");
        if (validate_only) {
            std::fprintf(stderr,
                "COSMOS_BLAS_VALIDATION status=passed gpu_executed=false tile_queries=%d "
                "workspace_elements=%llu multiply_accumulates=%llu\n", tile_queries,
                static_cast<unsigned long long>(score_count),
                static_cast<unsigned long long>(multiply_accumulates));
            return 0;
        }
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0));
        if (std::strncmp(device.name, "Apple ", 6) != 0)
            throw std::runtime_error("Apple GPU required");
        Buffer dq(q_count), dk(k_count), dv(v_count), ds(score_count), out(output_count);
        check(cudaMemcpy(dq.pointer, q.data(), q_count * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dk.pointer, k.data(), k_count * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dv.pointer, v.data(), v_count * sizeof(float), cudaMemcpyHostToDevice));
        // Release CPU staging storage before allocating the result vector.
        std::vector<float>().swap(q);
        std::vector<float>().swap(k);
        std::vector<float>().swap(v);
        BlasHandle blas;
        const float one = 1.0f, zero = 0.0f;
        int tiles = 0;
        for (int offset = 0; offset < queries; offset += tile_queries) {
            const int count = std::min(tile_queries, queries - offset);
            // Row-major QK^T is the transpose of this column-major KQ^T GEMM.
            check_blas(cublasSgemmStridedBatched(blas.handle, CUBLAS_OP_T, CUBLAS_OP_N,
                keys, count, width, &scale, dk.pointer, width, keys * width,
                dq.pointer + offset * width, width, queries * width, &zero,
                ds.pointer, keys, count * keys, int(heads)));
            check(cudaDeviceSynchronize());
            cosmos_blas_softmax<<<int(heads) * count, 128>>>(ds.pointer, count, keys, offset, header[7]);
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
            // V^T S^T writes directly to this query tile in every output head.
            check_blas(cublasSgemmStridedBatched(blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                value_width, count, keys, &one, dv.pointer, value_width, keys * value_width,
                ds.pointer, keys, count * keys, &zero, out.pointer + offset * value_width,
                value_width, queries * value_width, int(heads)));
            check(cudaDeviceSynchronize());
            ++tiles;
        }
        std::vector<float> result(output_count);
        check(cudaMemcpy(result.data(), out.pointer, output_count * sizeof(float), cudaMemcpyDeviceToHost));
        for (float value : result)
            if (!std::isfinite(value)) throw std::runtime_error("non-finite output");
        blas.release();
        dq.release(); dk.release(); dv.release(); ds.release(); out.release();
        if (std::fwrite(result.data(), sizeof(float), result.size(), stdout) != result.size() ||
            std::fflush(stdout) != 0) throw std::runtime_error("output write failed");
        // Application receipt, not runtime kernel provenance: GEMMs use public MPS.
        std::fprintf(stderr,
            "COSMOS_BLAS_COMPLETION status=passed gemm_backend=cumetal_cublas_mps "
            "softmax_backend=cumetal_cuda_metal tiles=%d gemm_calls=%d "
            "tile_queries=%d workspace_elements=%llu device_name=\"%s\"\n",
            tiles, 2 * tiles, tile_queries, static_cast<unsigned long long>(score_count), device.name);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "cosmos BLAS attention failed: %s\n", error.what());
        return 1;
    }
}
