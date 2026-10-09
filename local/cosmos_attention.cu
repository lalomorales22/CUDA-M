// Bounded FP32 SDPA worker for the experimental Cosmos/PyTorch bridge.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

__global__ void cosmos_scores(const float* q, const float* k, float* scores,
                              int queries, int keys, int width, float scale,
                              int causal, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int key = i % keys;
    int row = i / keys;
    int query = row % queries;
    int head = row / queries;
    float sum = 0.0f;
    for (int d = 0; d < width; ++d)
        sum += q[row * width + d] * k[(head * keys + key) * width + d];
    scores[i] = causal && key > query ? -INFINITY : sum * scale;
}

__global__ void cosmos_softmax(float* scores, int rows, int keys) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int base = row * keys;
    float maximum = scores[base];
    for (int j = 1; j < keys; ++j) maximum = fmaxf(maximum, scores[base + j]);
    float sum = 0.0f;
    for (int j = 0; j < keys; ++j) {
        float value = expf(scores[base + j] - maximum);
        scores[base + j] = value;
        sum += value;
    }
    for (int j = 0; j < keys; ++j) scores[base + j] /= sum;
}

__global__ void cosmos_values(const float* scores, const float* v, float* output,
                              int queries, int keys, int width, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int row = i / width;
    int d = i % width;
    int head = row / queries;
    float sum = 0.0f;
    for (int j = 0; j < keys; ++j)
        sum += scores[row * keys + j] * v[(head * keys + j) * width + d];
    output[i] = sum;
}

namespace {
void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
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

std::vector<float> read_values(std::size_t count) {
    std::vector<float> values(count);
    if (std::fread(values.data(), sizeof(float), count, stdin) != count)
        throw std::runtime_error("truncated input");
    for (float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite input");
    return values;
}
}

int main() {
    try {
        // Little-endian arm64 protocol: magic, B, H, Q, K, D, Dv, causal, scale.
        std::uint32_t header[8];
        float scale;
        if (std::fread(header, sizeof(header), 1, stdin) != 1 ||
            std::fread(&scale, sizeof(scale), 1, stdin) != 1)
            throw std::runtime_error("truncated header");
        if (header[0] != 0x434d4131 || !std::isfinite(scale) || header[7] > 1)
            throw std::runtime_error("invalid header");
        for (int i = 1; i <= 6; ++i)
            if (header[i] == 0 || header[i] > 4096)
                throw std::runtime_error("dimension out of range");
        const std::size_t heads = std::size_t(header[1]) * header[2];
        const int queries = header[3], keys = header[4], width = header[5], value_width = header[6];
        const std::size_t rows = heads * queries, score_count = rows * keys;
        if (heads > 128 || width > 128 || value_width > 128 || score_count > 67108864)
            throw std::runtime_error("unsupported attention size");
        auto q = read_values(rows * width);
        auto k = read_values(heads * keys * width);
        auto v = read_values(heads * keys * value_width);
        if (std::fgetc(stdin) != EOF) throw std::runtime_error("unexpected trailing input");
        Buffer dq(q.size()), dk(k.size()), dv(v.size()), ds(score_count), out(rows * value_width);
        check(cudaMemcpy(dq.pointer, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dk.pointer, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dv.pointer, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
        cosmos_scores<<<(score_count + 127) / 128, 128>>>(dq.pointer, dk.pointer, ds.pointer,
            queries, keys, width, scale, header[7], score_count);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        cosmos_softmax<<<(rows + 127) / 128, 128>>>(ds.pointer, rows, keys);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        cosmos_values<<<(rows * value_width + 127) / 128, 128>>>(ds.pointer, dv.pointer,
            out.pointer, queries, keys, value_width, rows * value_width);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        std::vector<float> result(rows * value_width);
        check(cudaMemcpy(result.data(), out.pointer, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (float value : result)
            if (!std::isfinite(value)) throw std::runtime_error("non-finite output");
        dq.release(); dk.release(); dv.release(); ds.release(); out.release();
        if (std::fwrite(result.data(), sizeof(float), result.size(), stdout) != result.size() ||
            std::fflush(stdout) != 0) throw std::runtime_error("output write failed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "cosmos attention failed: %s\n", error.what());
        return 1;
    }
}
