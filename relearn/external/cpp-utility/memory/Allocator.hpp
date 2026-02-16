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

#include "cpp-utility/Cast.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief A custom allocator that uses the memory pool.
 * @tparam T The data type
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

    constexpr Allocator(combinator_chunk_type* const combinator_ptr) noexcept
        : combinator{ combinator_ptr } {
    }

    template <class U>
    constexpr explicit Allocator(const Allocator<U, combinator_chunk_type>& a) noexcept
        : combinator{ a.combinator }
        , alloc{ a.alloc } {
    }

    ~Allocator() = default;

    [[nodiscard]] constexpr T* allocate(const size_type n, [[maybe_unused]] const void* const hint = nullptr) {
        auto* const ptr = combinator->get_pointer(n * sizeof(T));
        if (ptr == nullptr) {
            return alloc.allocate(n);
        }

        auto* const cast_ptr = reinterpret_cast<T*>(ptr);
        return cast_ptr;
    }

    constexpr void deallocate(T* const p, const size_type n) {
        if (!p) {
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

    size_type max_size() const noexcept {
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

    [[nodiscard]] friend constexpr bool operator==(const Allocator&, const Allocator&) = default;

private:
    combinator_chunk_type* combinator{};
    std::allocator<T> alloc{};
};

} // namespace utility
