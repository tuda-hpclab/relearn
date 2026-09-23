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
 * @brief Owns a fixed number of bytes inline and exposes them as one non-allocating storage block.
 *
 * get_pointer() returns the same base address for every fitting request. return_pointer() is intentionally a no-op;
 * allocation policies such as ManagedChunk add reuse bookkeeping around this storage. The byte array itself has byte
 * alignment, so clients requiring stronger alignment must verify the returned address.
 */
template <std::size_t number_bytes_>
class StackChunk {
    static_assert(number_bytes_ > 0, "StackChunk must contain at least one byte");

public:
    constexpr static std::size_t number_bytes = number_bytes_;

    constexpr StackChunk() = default;

    constexpr StackChunk(const StackChunk& other) = delete;
    constexpr StackChunk(StackChunk&& other) noexcept = delete;

    constexpr StackChunk& operator=(const StackChunk& other) = delete;
    constexpr StackChunk& operator=(StackChunk&& other) noexcept = delete;

    constexpr ~StackChunk() = default;

    /** @brief Returns the block's base address if @p size fits, otherwise nullptr. */
    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) noexcept {
        if (size > number_bytes) {
            return nullptr;
        }

        return data;
    }

    /** @brief No-op because the chunk always represents one permanently available block. */
    constexpr void return_pointer([[maybe_unused]] std::byte* const ptr) noexcept {
    }

    [[nodiscard]] constexpr std::size_t get_size() const noexcept {
        return number_bytes;
    }

    [[nodiscard]] constexpr const std::byte* get_first_address() const noexcept {
        return data;
    }

    /** @brief Returns the exclusive end address of the block. */
    [[nodiscard]] constexpr const std::byte* get_last_address() const noexcept {
        return data + number_bytes;
    }

private:
    std::byte data[number_bytes_];
};

/**
 * @brief Heap-allocated counterpart of StackChunk with the same single-block interface.
 *
 * Construction owns one byte array and may throw std::bad_alloc. The block has the alignment supplied by the global
 * array allocation function; clients requiring over-alignment must still verify the returned address.
 */
template <std::size_t number_bytes_>
class HeapChunk {
    static_assert(number_bytes_ > 0, "HeapChunk must contain at least one byte");

public:
    constexpr static std::size_t number_bytes = number_bytes_;

    constexpr HeapChunk() {
        data = new std::byte[number_bytes_];
    }

    constexpr HeapChunk(const HeapChunk& other) = delete;
    constexpr HeapChunk(HeapChunk&& other) noexcept = delete;

    constexpr HeapChunk& operator=(const HeapChunk& other) = delete;
    constexpr HeapChunk& operator=(HeapChunk&& other) noexcept = delete;

    constexpr ~HeapChunk() {
        delete[] data;
    }

    /** @brief Returns the block's base address if @p size fits, otherwise nullptr. */
    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) noexcept {
        if (size > number_bytes) {
            return nullptr;
        }

        return data;
    }

    /** @brief No-op because the chunk always represents one permanently available block. */
    constexpr void return_pointer([[maybe_unused]] std::byte* const ptr) noexcept {
    }

    [[nodiscard]] constexpr std::size_t get_size() const noexcept {
        return number_bytes;
    }

    [[nodiscard]] constexpr const std::byte* get_first_address() const noexcept {
        return data;
    }

    /** @brief Returns the exclusive end address of the block. */
    [[nodiscard]] constexpr const std::byte* get_last_address() const noexcept {
        return data + number_bytes;
    }

private:
    std::byte* data;
};

} // namespace utility
