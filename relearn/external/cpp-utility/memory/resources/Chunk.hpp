#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <cstddef>

namespace utility {

template <std::size_t number_bytes_>
class StackChunk {
public:
    constexpr static std::size_t number_bytes = number_bytes_;

    constexpr StackChunk() = default;

    constexpr StackChunk(const StackChunk& other) = delete;
    constexpr StackChunk(StackChunk&& other) noexcept = delete;

    constexpr StackChunk& operator=(const StackChunk& other) = delete;
    constexpr StackChunk& operator=(StackChunk&& other) noexcept = delete;

    constexpr ~StackChunk() = default;

    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) noexcept {
        if (size > number_bytes) {
            return nullptr;
        }

        return data;
    }

    constexpr void return_pointer([[maybe_unused]] std::byte* const ptr) noexcept {
        // NOOP
    }

    [[nodiscard]] constexpr std::size_t get_size() const noexcept {
        return number_bytes;
    }

    [[nodiscard]] constexpr const std::byte* get_first_address() const noexcept {
        return data;
    }

    [[nodiscard]] constexpr const std::byte* get_last_address() const noexcept {
        return data + number_bytes;
    }

private:
    std::byte data[number_bytes_];
};

template <std::size_t number_bytes_>
class HeapChunk {
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

    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) noexcept {
        if (size > number_bytes) {
            return nullptr;
        }

        return data;
    }

    constexpr void return_pointer([[maybe_unused]] std::byte* const ptr) noexcept {
        // NOOP
    }

    [[nodiscard]] constexpr std::size_t get_size() const noexcept {
        return number_bytes;
    }

    [[nodiscard]] constexpr const std::byte* get_first_address() const noexcept {
        return data;
    }

    [[nodiscard]] constexpr const std::byte* get_last_address() const noexcept {
        return data + number_bytes;
    }

private:
    std::byte* data;
};

} // namespace utility
