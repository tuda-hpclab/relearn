#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <cstddef>

namespace utility {

/**
 * @brief Splits a byte chunk into fixed-size units and hands them out once in address order.
 *
 * Fitting requests consume one whole unit until capacity is exhausted. return_pointer() deliberately does nothing,
 * so storage is reclaimed only when the MonotonicChunk itself is destroyed. Object lifetimes and synchronization
 * remain the caller's responsibility.
 */
template <typename chunk_type, std::size_t size_of_allocation_unit_>
class MonotonicChunk {
public:
    constexpr static std::size_t number_bytes = chunk_type::number_bytes;
    constexpr static std::size_t size_of_allocation_unit = size_of_allocation_unit_;
    static_assert(size_of_allocation_unit > 0, "MonotonicChunk allocation units must not be empty");
    constexpr static std::size_t capacity = number_bytes / size_of_allocation_unit_;
    static_assert(capacity > 0, "MonotonicChunk allocation units must fit into the underlying chunk");

    constexpr MonotonicChunk() = default;

    constexpr MonotonicChunk(const MonotonicChunk& other) = delete;
    constexpr MonotonicChunk(MonotonicChunk&& other) noexcept = delete;

    constexpr MonotonicChunk& operator=(const MonotonicChunk& other) = delete;
    constexpr MonotonicChunk& operator=(MonotonicChunk&& other) noexcept = delete;

    constexpr ~MonotonicChunk() = default;

    /** @brief Consumes and returns the next unit, or nullptr for an oversized request or an exhausted chunk. */
    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) noexcept {
        if (size > size_of_allocation_unit) {
            return nullptr;
        }

        if (filling >= capacity) {
            return nullptr;
        }

        auto* data = chunk.get_pointer(size);
        data += filling * size_of_allocation_unit_;
        filling++;
        return data;
    }

    /** @brief No-op: monotonic allocations are never individually reclaimed. */
    constexpr void return_pointer([[maybe_unused]] std::byte* const ptr) noexcept {
    }

    [[nodiscard]] constexpr std::size_t get_size() const noexcept {
        return capacity - filling;
    }

private:
    chunk_type chunk{};
    std::size_t filling{ 0 };
};

} // namespace utility
