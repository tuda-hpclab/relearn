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
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief Owns a fixed number of equally sized, suitably aligned raw-memory chunks.
 *
 * get_pointer() returns storage for exactly @p chunk_size objects of type @p T. The objects' lifetimes are not
 * started; callers that construct objects in the storage must destroy them before returning the pointer. If all
 * pooled chunks are in use, the request transparently falls back to std::allocator. The class is not thread-safe.
 *
 * @tparam T The element type whose size and alignment determine the storage.
 * @tparam chunk_size The positive number of T objects that fit into every returned chunk.
 */
template <typename T, std::size_t chunk_size>
class MemoryPool {
    static_assert(chunk_size > 0, "MemoryPool chunks must contain at least one element");

    using allocator_traits = std::allocator_traits<std::allocator<T>>;

public:
    /**
     * @brief Allocates storage for @p number_chunks pooled chunks.
     * @param number_chunks The pool capacity; zero creates an always-fallback pool.
     * @exception std::length_error If the requested element count overflows size_type or exceeds allocator limits.
     * @exception std::bad_alloc If the storage allocation fails.
     */
    explicit MemoryPool(const std::size_t number_chunks)
        : number_of_chunks(number_chunks) {
        const auto max_elements = allocator_traits::max_size(default_allocator);
        if (number_chunks > max_elements / chunk_size) {
            throw std::length_error{ "MemoryPool capacity exceeds allocator limits" };
        }

        pointers.resize(number_chunks);
        checked_out.resize(number_chunks, false);

        const auto element_count = number_chunks * chunk_size;
        if (element_count == 0) {
            return;
        }

        memory = allocator_traits::allocate(default_allocator, element_count);
        for (auto i = std::size_t{ 0 }; i < number_chunks; ++i) {
            pointers[i] = memory + i * chunk_size;
        }
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&& other) noexcept
        : memory{ std::exchange(other.memory, nullptr) }
        , pointers{ std::move(other.pointers) }
        , checked_out{ std::move(other.checked_out) }
        , number_of_chunks{ std::exchange(other.number_of_chunks, 0) }
        , default_allocator{ std::move(other.default_allocator) } {
    }

    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool& operator=(MemoryPool&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        release_storage();
        memory = std::exchange(other.memory, nullptr);
        pointers = std::move(other.pointers);
        checked_out = std::move(other.checked_out);
        number_of_chunks = std::exchange(other.number_of_chunks, 0);
        default_allocator = std::move(other.default_allocator);
        return *this;
    }

    ~MemoryPool() {
        release_storage();
    }

    /**
     * @brief Obtains uninitialized storage for @p chunk_size T objects.
     * @return A pooled pointer, or a separately allocated pointer while the pool is exhausted.
     * @exception std::bad_alloc If fallback allocation fails.
     */
    [[nodiscard]] T* get_pointer() {
        if (pointers.empty()) {
            return allocator_traits::allocate(default_allocator, chunk_size);
        }

        auto* const ptr = pointers.back();
        pointers.pop_back();
        const auto chunk_index = static_cast<std::size_t>(ptr - memory) / chunk_size;
        checked_out[chunk_index] = true;
        return ptr;
    }

    /**
     * @brief Returns storage obtained from this pool; fallback storage is deallocated immediately.
     * @param ptr A pointer previously returned by get_pointer(); nullptr is ignored.
     * @exception std::invalid_argument If @p ptr points into the pool but is not the start of a currently checked-out chunk.
     */
    void return_pointer(T* const ptr) {
        if (!ptr) {
            return;
        }

        if (memory == nullptr) {
            allocator_traits::deallocate(default_allocator, ptr, chunk_size);
            return;
        }

        const auto begin_address = reinterpret_cast<std::uintptr_t>(memory);
        const auto end_address = reinterpret_cast<std::uintptr_t>(memory + number_of_chunks * chunk_size);
        const auto pointer_address = reinterpret_cast<std::uintptr_t>(ptr);
        if (pointer_address < begin_address || pointer_address > end_address) {
            allocator_traits::deallocate(default_allocator, ptr, chunk_size);
            return;
        }

        if (pointer_address == end_address) {
            throw std::invalid_argument{ "MemoryPool cannot return its one-past-the-end pointer" };
        }

        const auto byte_offset = pointer_address - begin_address;
        constexpr auto chunk_bytes = chunk_size * sizeof(T);
        if (byte_offset % chunk_bytes != 0) {
            throw std::invalid_argument{ "MemoryPool can only return the start of a chunk" };
        }

        const auto chunk_index = byte_offset / chunk_bytes;
        if (!checked_out[chunk_index]) {
            throw std::invalid_argument{ "MemoryPool chunk was already returned" };
        }
        checked_out[chunk_index] = false;
        pointers.push_back(ptr);
    }

    /**
     * @brief Returns the number of available pointers currently in the pool
     * @return The number of pointers
     */
    [[nodiscard]] std::size_t get_number_available_pointers() const noexcept {
        return pointers.size();
    }

    /**
     * @brief Returns the capacity of the pool, i.e., how many pointers are available and how many are currently in use
     * @return The capacity
     */
    [[nodiscard]] std::size_t get_capacity() const noexcept {
        return number_of_chunks;
    }

private:
    void release_storage() noexcept {
        if (memory != nullptr) {
            allocator_traits::deallocate(default_allocator, memory, number_of_chunks * chunk_size);
            memory = nullptr;
        }
    }

    T* memory{};
    std::vector<T*> pointers{};
    std::vector<bool> checked_out{};
    std::size_t number_of_chunks{};
    std::allocator<T> default_allocator{};
};

} // namespace utility
