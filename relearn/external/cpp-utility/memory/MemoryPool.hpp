#pragma once

/*
 * This file is part of the ScalableGraphAlgorithm software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cpp-utility/Exception.hpp"

#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace utility {

/**
 * @brief Represents a memory pool resource
 * @tparam T The type to store
 * @tparam chunk_size The size of a chunk, i.e., always returns a pointer with enough space to store that number of elements
 */
template <typename T, std::size_t chunk_size>
class MemoryPool {
    constexpr static std::size_t default_memory_size = 1024UL * 1024UL;

public:
    /**
     * @brief Initializes the current resource
     * @param number_chunks The number of chunks that should be allocated.
     */
    constexpr MemoryPool(const std::size_t number_chunks) {
        const auto sizeoft = sizeof(T);

        const auto total_memory_size = number_chunks * chunk_size * sizeoft + chunk_size * sizeoft;

        memory.resize(total_memory_size);
        pointers.resize(number_chunks);

        auto* current_ptr = memory.data();
        auto size = number_chunks * chunk_size * sizeoft;

        for (auto i = std::size_t{ 0 }; i < number_chunks; i++) {
            auto* cast_current_ptr = static_cast<void*>(current_ptr);
            auto* aligned_ptr = std::align(alignof(T), sizeoft, cast_current_ptr, size);
            pointers[i] = reinterpret_cast<T*>(aligned_ptr);

            size -= chunk_size * sizeoft;
            current_ptr += sizeoft * chunk_size;
        }

        number_of_chunks = number_chunks;
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = default;

    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool& operator=(MemoryPool&&) = default;

    /**
     * @brief Retrieves a pointer to memory with enough space for chunk_size.
     *		If all slots of the pool are filled, returns a default-allocated pointer
     * @return A pointer
     */
    [[nodiscard]] constexpr T* get_pointer() {
        if (pointers.empty()) {
            return default_allocator.allocate(chunk_size);
        }

        auto* ptr = pointers.back();
        pointers.pop_back();

        std::ranges::uninitialized_value_construct_n(ptr, chunk_size);

        return ptr;
    }

    /**
     * @brief Returns the pointer to the memory pool; can also deal with default-allocated pointers.
     * @param ptr The pointer to return
     */
    constexpr void return_pointer(T* const ptr) {
        if (!ptr) {
            return;
        }

        const auto comparator = std::less{};
        const auto* const other_ptr = reinterpret_cast<std::byte*>(ptr);

        auto* const first_memory = memory.data();
        auto* const last_memory = memory.data() + number_of_chunks * chunk_size * sizeof(T);

        const auto larger = comparator(last_memory, other_ptr);

        if (const auto smaller = comparator(other_ptr, first_memory); smaller || larger) {
            default_allocator.deallocate(ptr, chunk_size);
            return;
        }

        std::ranges::destroy_n(ptr, chunk_size);

        pointers.push_back(ptr);
    }

    /**
     * @brief Returns the number of available pointers currently in the pool
     * @return The number of pointers
     */
    [[nodiscard]] constexpr std::size_t get_number_available_pointers() noexcept {
        return pointers.size();
    }

    /**
     * @brief Returns the capacity of the pool, i.e., how many pointers are available and how many are currently in use
     * @return The capacity
     */
    [[nodiscard]] constexpr std::size_t get_capacity() noexcept {
        return number_of_chunks;
    }

private:
    std::vector<std::byte> memory{};
    std::deque<T*> pointers{};
    std::size_t number_of_chunks{};
    std::allocator<T> default_allocator{};
};

} // namespace utility
