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

#include "cpp-utility/Cast.hpp"

#include <cstddef>
#include <cstdint>
#include <new>

namespace utility {

/**
 * @brief Splits a byte chunk into a fixed-capacity pool whose free-list metadata is stored inline.
 *
 * Requests no larger than size_of_allocation_unit receive one whole unit. Returned pointers are accepted only when
 * they name the start of a currently checked-out unit; foreign, interior, null, and duplicate pointers are rejected
 * without changing the free list. No object lifetime management or synchronization is performed.
 */
template <typename chunk_type, std::size_t size_of_allocation_unit_>
class StackManagedChunk {
public:
    constexpr static std::size_t number_bytes = chunk_type::number_bytes;
    constexpr static std::size_t size_of_allocation_unit = size_of_allocation_unit_;
    static_assert(size_of_allocation_unit > 0, "ManagedChunk allocation units must not be empty");
    constexpr static std::size_t capacity = number_bytes / size_of_allocation_unit_;
    static_assert(capacity > 0, "ManagedChunk allocation units must fit into the underlying chunk");

    constexpr StackManagedChunk() {
        for (auto i = std::size_t{ 0 }; i < capacity; ++i) {
            slots[i].free_offset = i * size_of_allocation_unit;
        }
    }

    constexpr StackManagedChunk(const StackManagedChunk& other) = delete;
    constexpr StackManagedChunk(StackManagedChunk&& other) noexcept = delete;

    constexpr StackManagedChunk& operator=(const StackManagedChunk& other) = delete;
    constexpr StackManagedChunk& operator=(StackManagedChunk&& other) noexcept = delete;

    constexpr ~StackManagedChunk() = default;

    /** @brief Checks out one unit for a fitting request, or returns nullptr when the request cannot be served. */
    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) noexcept {
        if (size > size_of_allocation_unit) {
            return nullptr;
        }

        if (filling >= capacity) {
            return nullptr;
        }

        auto* data = chunk.get_pointer(size);

        const auto current_free_offset = slots[filling].free_offset;
        data += current_free_offset;
        slots[current_free_offset / size_of_allocation_unit].checked_out = true;
        filling++;

        return data;
    }

    /** @brief Returns a checked-out unit; invalid, interior, or duplicate pointers return false and do nothing. */
    [[nodiscard]] constexpr bool return_pointer(std::byte* const ptr) noexcept {
        if (ptr == nullptr) {
            return false;
        }

        const auto first_address = reinterpret_cast<std::uintptr_t>(chunk.get_first_address());
        const auto pointer_address = reinterpret_cast<std::uintptr_t>(ptr);
        if (pointer_address < first_address || pointer_address >= first_address + number_bytes) {
            return false;
        }

        const auto offset = pointer_address - first_address;
        if (offset % size_of_allocation_unit != 0) {
            return false;
        }

        const auto slot_index = safe_cast<std::size_t>(offset / size_of_allocation_unit);
        if (slot_index >= capacity || !slots[slot_index].checked_out) {
            return false;
        }

        slots[slot_index].checked_out = false;
        filling--;
        slots[filling].free_offset = safe_cast<std::size_t>(offset);

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
    struct SlotState {
        std::size_t free_offset{};
        bool checked_out{};
    };

    chunk_type chunk{};
    SlotState slots[capacity]{};
    std::size_t filling{ 0 };
};

/**
 * @brief StackManagedChunk counterpart that stores its free-list metadata on the heap.
 *
 * Pointer validation, capacity, and allocation semantics are identical to StackManagedChunk. Construction may throw
 * std::bad_alloc while allocating metadata.
 */
template <typename chunk_type, std::size_t size_of_allocation_unit_>
class HeapManagedChunk {
public:
    constexpr static std::size_t number_bytes = chunk_type::number_bytes;
    constexpr static std::size_t size_of_allocation_unit = size_of_allocation_unit_;
    static_assert(size_of_allocation_unit > 0, "ManagedChunk allocation units must not be empty");
    constexpr static std::size_t capacity = number_bytes / size_of_allocation_unit_;
    static_assert(capacity > 0, "ManagedChunk allocation units must fit into the underlying chunk");

    constexpr HeapManagedChunk() {
        slots = new SlotState[capacity]{};
        for (auto i = std::size_t{ 0 }; i < capacity; ++i) {
            slots[i].free_offset = i * size_of_allocation_unit;
        }
    }

    constexpr HeapManagedChunk(const HeapManagedChunk& other) = delete;
    constexpr HeapManagedChunk(HeapManagedChunk&& other) noexcept = delete;

    constexpr HeapManagedChunk& operator=(const HeapManagedChunk& other) = delete;
    constexpr HeapManagedChunk& operator=(HeapManagedChunk&& other) noexcept = delete;

    constexpr ~HeapManagedChunk() {
        delete[] slots;
    }

    /** @brief Checks out one unit for a fitting request, or returns nullptr when the request cannot be served. */
    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) noexcept {
        if (size > size_of_allocation_unit) {
            return nullptr;
        }

        if (filling >= capacity) {
            return nullptr;
        }

        auto* data = chunk.get_pointer(size);

        const auto current_free_offset = slots[filling].free_offset;
        data += current_free_offset;
        slots[current_free_offset / size_of_allocation_unit].checked_out = true;
        filling++;

        return data;
    }

    /** @brief Returns a checked-out unit; invalid, interior, or duplicate pointers return false and do nothing. */
    [[nodiscard]] constexpr bool return_pointer(std::byte* const ptr) noexcept {
        if (ptr == nullptr) {
            return false;
        }

        const auto first_address = reinterpret_cast<std::uintptr_t>(chunk.get_first_address());
        const auto pointer_address = reinterpret_cast<std::uintptr_t>(ptr);
        if (pointer_address < first_address || pointer_address >= first_address + number_bytes) {
            return false;
        }

        const auto offset = pointer_address - first_address;
        if (offset % size_of_allocation_unit != 0) {
            return false;
        }

        const auto slot_index = safe_cast<std::size_t>(offset / size_of_allocation_unit);
        if (slot_index >= capacity || !slots[slot_index].checked_out) {
            return false;
        }

        slots[slot_index].checked_out = false;
        filling--;
        slots[filling].free_offset = safe_cast<std::size_t>(offset);

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
    struct SlotState {
        std::size_t free_offset{};
        bool checked_out{};
    };

    chunk_type chunk{};
    SlotState* slots{};
    std::size_t filling{ 0 };
};

} // namespace utility
