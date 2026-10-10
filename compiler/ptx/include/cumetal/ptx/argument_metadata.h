#pragma once

#include <cstdint>

namespace cumetal::ptx {

// Explicit CUDA launch arguments only; hidden runtime bindings are excluded.
struct ArgumentMetadata {
    bool is_pointer = false;
    std::uint32_t size_bytes = 0;
    bool is_aggregate = false;
};

}  // namespace cumetal::ptx
