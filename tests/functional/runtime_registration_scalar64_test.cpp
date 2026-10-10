#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

extern "C" {
void** __cudaRegisterFatBinary(const void*);
void __cudaUnregisterFatBinary(void**);
void __cudaRegisterFunction(void**, const void*, char*, const char*, int,
                            void*, void*, void*, void*, int*);
}

namespace {

const char kPtx[] = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry copy_scalar_bits(.param .u64 out, .param .u64 value)
{
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [value];
    cvta.to.global.u64 %rd3, %rd1;
    st.global.u64 [%rd3], %rd2;
    ret;
}
)ptx";

struct FatbinWrapper {
    std::uint32_t magic = 0x466243b1u;
    std::uint32_t version = 1;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

void copy_stub() {}
void alias_stub() {}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) return 1;
    struct FatbinHeader {
        std::uint32_t magic = 0xba55ed50u;
        std::uint16_t version = 1;
        std::uint16_t header_size = 16;
        std::uint64_t fat_size = sizeof(kPtx);
    } header;
    std::vector<unsigned char> blob(sizeof(header) + sizeof(kPtx));
    std::memcpy(blob.data(), &header, sizeof(header));
    std::memcpy(blob.data() + sizeof(header), kPtx, sizeof(kPtx));
    FatbinWrapper wrapper;
    wrapper.data = blob.data();
    void** handle = __cudaRegisterFatBinary(&wrapper);
    if (handle == nullptr) return 1;
    char name[] = "copy_scalar_bits";
    for (const auto stub : {copy_stub, alias_stub}) {
        __cudaRegisterFunction(handle, reinterpret_cast<const void*>(stub), name,
                               nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    void* allocation = nullptr;
    if (cudaMalloc(&allocation, 2 * sizeof(std::uint64_t)) != cudaSuccess ||
        cudaMemset(allocation, 0, 2 * sizeof(std::uint64_t)) != cudaSuccess) return 1;
    // Use an interior device address to exercise allocation offset resolution.
    void* output = static_cast<unsigned char*>(allocation) + sizeof(std::uint64_t);
    const std::uint64_t values[] = {
        0, 1, 0x3f8000003f800000ull, 0xffffffffffffffffull,
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(allocation)),
        0x8000000000000000ull,
    };
    for (std::size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        std::uint64_t value = values[i];
        void* args[] = {&output, &value};
        const auto stub = i == 5 ? alias_stub : copy_stub;
        if (cudaLaunchKernel(reinterpret_cast<const void*>(stub), dim3(1), dim3(1),
                             args, 0, nullptr) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: scalar64 launch %zu\n", i);
            return 1;
        }
        std::uint64_t actual[2] = {};
        if (cudaMemcpy(actual, allocation, sizeof(actual), cudaMemcpyDeviceToHost) != cudaSuccess ||
            actual[0] != 0 || actual[1] != value) {
            std::fprintf(stderr, "FAIL: scalar64 bits %zu: got %llx expected %llx\n", i,
                         static_cast<unsigned long long>(actual[1]),
                         static_cast<unsigned long long>(value));
            return 1;
        }
    }

    // A compiler-known pointer must never be reinterpreted as a scalar merely
    // because its invalid address happens to be a small integer.
    for (const std::uintptr_t invalid : {std::uintptr_t{1}, std::uintptr_t{0x7ffffffffffff000ull}}) {
        void* bad_output = reinterpret_cast<void*>(invalid);
        std::uint64_t value = 7;
        void* args[] = {&bad_output, &value};
        if (cudaLaunchKernel(reinterpret_cast<const void*>(copy_stub), dim3(1), dim3(1),
                             args, 0, nullptr) != cudaErrorInvalidDevicePointer ||
            cudaDeviceSynchronize() != cudaErrorInvalidDevicePointer ||
            cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: invalid pointer launch/error propagation\n");
            return 1;
        }
    }
    cudaFree(allocation);
    __cudaUnregisterFatBinary(handle);
    std::puts("REGISTRATION_SCALAR64_OK");
    return 0;
}
