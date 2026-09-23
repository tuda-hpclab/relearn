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
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief A stateful standard allocator that first tries a size-segregated CombinatorChunk.
 *
 * Requests that do not fit, whose bucket is exhausted, or whose returned address is insufficiently aligned for T
 * fall back to std::allocator. Copies and rebound allocators retain the same non-owning combinator pointer, so the
 * combinator must outlive every container and allocation using this allocator. A null combinator selects the
 * std::allocator fallback for every request. The combinator is not synchronized.
 *
 * @tparam T The allocated element type.
 * @tparam combinator_chunk_type A resource exposing get_pointer(byte_count) and return_pointer(byte_count, pointer).
 */
template <class T, class combinator_chunk_type>
class Allocator {
public:
    template <class U, class combinator_chunk_ty>
    friend class Allocator;

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using pointer = T*;
    using const_pointer = const T*;

    using reference = T&;
    using const_reference = const T&;

    template <class _Tp1>
    struct rebind {
        using other = Allocator<_Tp1, combinator_chunk_type>;
    };

    using is_always_equal = std::false_type;

    /**
     * @brief Associates the allocator with a combinator resource.
     * @param combinator_ptr A resource that must outlive this allocator and its allocations, or nullptr to use only
     *      the std::allocator fallback.
     */
    constexpr explicit Allocator(combinator_chunk_type* const combinator_ptr) noexcept
        : combinator{ combinator_ptr } {
    }

    template <class U>
    constexpr Allocator(const Allocator<U, combinator_chunk_type>& a) noexcept
        : combinator{ a.combinator }
        , alloc{ a.alloc } {
    }

    ~Allocator() = default;

    /**
     * @brief Allocates uninitialized storage for @p n elements.
     * @exception std::bad_array_new_length If @p n exceeds max_size().
     * @exception std::bad_alloc If both resource selection and fallback allocation cannot provide storage.
     */
    [[nodiscard]] T* allocate(const size_type n, [[maybe_unused]] const void* const hint = nullptr) {
        if (n > max_size()) {
            throw std::bad_array_new_length{};
        }
        if (n == 0) {
            return alloc.allocate(0);
        }
        if (combinator == nullptr) {
            return alloc.allocate(n);
        }

        const auto number_bytes = n * sizeof(T);
        auto* const ptr = combinator->get_pointer(number_bytes);
        if (ptr == nullptr) {
            return alloc.allocate(n);
        }

        const auto address = reinterpret_cast<std::uintptr_t>(ptr);
        if (address % alignof(T) == 0) {
            return reinterpret_cast<T*>(ptr);
        }

        [[maybe_unused]] const auto returned = combinator->return_pointer(number_bytes, ptr);
        return alloc.allocate(n);
    }

    /**
     * @brief Deallocates storage using the same byte-count bucket used by allocate().
     * @param p A pointer obtained from an equal allocator; nullptr is ignored.
     * @param n The exact element count passed to allocate().
     */
    void deallocate(T* const p, const size_type n) {
        if (!p) {
            return;
        }

        if (n == 0) {
            alloc.deallocate(p, 0);
            return;
        }
        if (combinator == nullptr) {
            alloc.deallocate(p, n);
            return;
        }

        auto* const cast_ptr = reinterpret_cast<std::byte*>(p);
        const auto is_returned = combinator->return_pointer(n * sizeof(T), cast_ptr);
        if (is_returned) {
            return;
        }

        alloc.deallocate(p, n);
    }

    pointer address(reference x) const noexcept {
        return std::addressof(x);
    }

    const_pointer address(const_reference x) const noexcept {
        return std::addressof(x);
    }

    [[nodiscard]] constexpr size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new (reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) {
        p->~U();
    }

    /** @brief Returns the non-owning resource pointer used by allocator equality. */
    [[nodiscard]] constexpr combinator_chunk_type* get_combinator() const noexcept {
        return combinator;
    }

    template <class U>
    [[nodiscard]] friend constexpr bool operator==(
        const Allocator& lhs, const Allocator<U, combinator_chunk_type>& rhs) noexcept {
        return lhs.combinator == rhs.get_combinator();
    }

private:
    combinator_chunk_type* combinator{};
    std::allocator<T> alloc{};
};

} // namespace utility
