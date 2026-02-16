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

#include <cpp-utility/Cast.hpp>

namespace utility {

template <typename chunk_type, std::size_t size_of_allocation_unit_>
class StackManagedChunk {
public:
    constexpr static std::size_t number_bytes = chunk_type::number_bytes;
    constexpr static std::size_t size_of_allocation_unit = size_of_allocation_unit_;
    constexpr static std::size_t capacity = number_bytes / size_of_allocation_unit_;

    constexpr StackManagedChunk() {
        for (auto i = std::size_t{ 0 }; i < capacity; ++i) {
            free_indices[i] = i * size_of_allocation_unit;
        }
    }

    constexpr StackManagedChunk(const StackManagedChunk& other) = delete;
    constexpr StackManagedChunk(StackManagedChunk&& other) noexcept = delete;

    constexpr StackManagedChunk& operator=(const StackManagedChunk& other) = delete;
    constexpr StackManagedChunk& operator=(StackManagedChunk&& other) noexcept = delete;

    constexpr ~StackManagedChunk() = default;

    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) noexcept {
        if (size > size_of_allocation_unit) {
            return nullptr;
        }

        if (filling >= capacity) {
            return nullptr;
        }

        auto* data = chunk.get_pointer(size);

        const auto current_free_index = free_indices[filling];
        data += current_free_index;
        filling++;

        return data;
    }

    [[nodiscard]] constexpr bool return_pointer(std::byte* const ptr) noexcept {
        if (ptr < chunk.get_first_address() || ptr >= chunk.get_last_address()) {
            return false;
        }

        auto* data = chunk.get_pointer(0);
        const auto index = ptr - data;
        filling--;
        free_indices[filling] = save_cast<std::size_t>(index);

        return true;
    }

    [[nodiscard]] constexpr std::size_t get_size() const noexcept {
        return capacity - filling;
    }

    [[nodiscard]] constexpr const std::byte* get_first_address() const noexcept {
        return chunk.get_first_address();
    }

    [[nodiscard]] constexpr const std::byte* get_last_address() const noexcept {
        return chunk.get_last_address();
    }

private:
    chunk_type chunk{};

    std::size_t free_indices[capacity];
    std::size_t filling{ 0 };
};

template <typename chunk_type, std::size_t size_of_allocation_unit_>
class HeapManagedChunk {
public:
    constexpr static std::size_t number_bytes = chunk_type::number_bytes;
    constexpr static std::size_t size_of_allocation_unit = size_of_allocation_unit_;
    constexpr static std::size_t capacity = number_bytes / size_of_allocation_unit_;

    constexpr HeapManagedChunk() {
        free_indices = new std::size_t[capacity];
        for (auto i = std::size_t{ 0 }; i < capacity; ++i) {
            free_indices[i] = i * size_of_allocation_unit;
        }
    }

    constexpr HeapManagedChunk(const HeapManagedChunk& other) = delete;
    constexpr HeapManagedChunk(HeapManagedChunk&& other) noexcept = delete;

    constexpr HeapManagedChunk& operator=(const HeapManagedChunk& other) = delete;
    constexpr HeapManagedChunk& operator=(HeapManagedChunk&& other) noexcept = delete;

    constexpr ~HeapManagedChunk() {
        delete[] free_indices;
    }

    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) noexcept {
        if (size > size_of_allocation_unit) {
            return nullptr;
        }

        if (filling >= capacity) {
            return nullptr;
        }

        auto* data = chunk.get_pointer(size);

        const auto current_free_index = free_indices[filling];
        data += current_free_index;
        filling++;

        return data;
    }

    [[nodiscard]] constexpr bool return_pointer(std::byte* const ptr) noexcept {
        if (ptr < chunk.get_first_address() || ptr >= chunk.get_last_address()) {
            return false;
        }

        auto* data = chunk.get_pointer(0);
        const auto index = ptr - data;
        filling--;
        free_indices[filling] = save_cast<std::size_t>(index);

        return true;
    }

    [[nodiscard]] constexpr std::size_t get_size() const noexcept {
        return capacity - filling;
    }

    [[nodiscard]] constexpr const std::byte* get_first_address() const noexcept {
        return chunk.get_first_address();
    }

    [[nodiscard]] constexpr const std::byte* get_last_address() const noexcept {
        return chunk.get_last_address();
    }

private:
    chunk_type chunk{};

    std::size_t* free_indices;
    std::size_t filling{ 0 };
};

} // namespace utility
