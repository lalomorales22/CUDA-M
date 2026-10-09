#include <cuda_runtime.h>
#include <cstdio>

__global__ void scoped_atomic_returns(int* signed_slot, unsigned* unsigned_slot,
                                     unsigned* old_values) {
    old_values[0] = static_cast<unsigned>(atomicAdd_system(signed_slot, 3));
    old_values[1] = static_cast<unsigned>(atomicExch_system(signed_slot, 7));
    old_values[2] = static_cast<unsigned>(atomicMax_system(signed_slot, -3));
    old_values[3] = static_cast<unsigned>(atomicMin_system(signed_slot, -12));
    old_values[4] = static_cast<unsigned>(atomicCAS_system(signed_slot, 7, 99));
    old_values[5] = static_cast<unsigned>(atomicCAS_system(signed_slot, -12, -4));
    old_values[6] = atomicOr_system(unsigned_slot, 15u);
    old_values[7] = atomicAnd_system(unsigned_slot, 0x80000000u);
    old_values[8] = atomicXor_system(unsigned_slot, 0x80000001u);
    old_values[9] = atomicMax_system(unsigned_slot, 0xfffffffeu);
    old_values[10] = atomicMin_system(unsigned_slot, 0x80000000u);
    old_values[11] = atomicInc_system(unsigned_slot, 17u);
    old_values[12] = atomicDec_system(unsigned_slot, 5u);
}

__global__ void scoped_atomic_tickets(int* counter, unsigned* extrema, unsigned* tickets) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    tickets[tid] = static_cast<unsigned>(atomicAdd_system(counter, 1));
    atomicMax_system(&extrema[0], 0x80000000u + tid);
    atomicMin_system(&extrema[1], 0x80000000u + tid);
}

int main() {
    int* signed_slot = nullptr;
    unsigned* unsigned_slot = nullptr;
    unsigned* old_values = nullptr;
    if (cudaMallocManaged(&signed_slot, sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&unsigned_slot, 2 * sizeof(unsigned)) != cudaSuccess ||
        cudaMallocManaged(&old_values, (13 + 1024) * sizeof(unsigned)) != cudaSuccess) return 1;
    *signed_slot = -9;
    *unsigned_slot = 0xfffffff0u;
    for (unsigned i = 0; i < 13; ++i) old_values[i] = 0xdeadbeefu;
    scoped_atomic_returns<<<1, 1>>>(signed_slot, unsigned_slot, old_values);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 1;
    const unsigned expected[] = {
        static_cast<unsigned>(-9), static_cast<unsigned>(-6), 7u, 7u,
        static_cast<unsigned>(-12), static_cast<unsigned>(-12),
        0xfffffff0u, 0xffffffffu, 0x80000000u, 1u,
        0xfffffffeu, 0x80000000u, 0u
    };
    int failures = 0;
    for (unsigned i = 0; i < 13; ++i) {
        if (old_values[i] == expected[i]) continue;
        std::printf("FAIL: old value %u got %08x expected %08x\n", i, old_values[i], expected[i]);
        ++failures;
    }
    failures += *signed_slot != -4 || *unsigned_slot != 5u;
    *signed_slot = 0;
    unsigned_slot[0] = 0;
    unsigned_slot[1] = 0xffffffffu;
    for (unsigned i = 0; i < 1024; ++i) old_values[13 + i] = 0xdeadbeefu;
    scoped_atomic_tickets<<<8, 128>>>(signed_slot, unsigned_slot, old_values + 13);
    __sync_fetch_and_add(signed_slot, 7);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 1;
    bool seen[1031] = {};
    for (unsigned i = 0; i < 1024; ++i) {
        const unsigned ticket = old_values[13 + i];
        if (ticket >= 1031 || seen[ticket]) {
            ++failures;
        } else {
            seen[ticket] = true;
        }
    }
    failures += *signed_slot != 1031 || unsigned_slot[0] != 0x800003ffu ||
                unsigned_slot[1] != 0x80000000u;
    if (cudaFree(signed_slot) != cudaSuccess || cudaFree(unsigned_slot) != cudaSuccess ||
        cudaFree(old_values) != cudaSuccess) return 1;
    if (failures) return 1;
    std::printf("PASS: system atomic old values, unsigned boundaries, CAS outcomes, and 1024 unique tickets\n");
    return 0;
}
