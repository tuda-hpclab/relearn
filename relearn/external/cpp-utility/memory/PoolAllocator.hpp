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

#include "cpp-utility/MemoryPool.hpp"

#include <cstddef>
#include <type_traits>

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || (!defined(_MSVC_LANG) && __cplusplus < 202002L))
#include <limits>
#include <utility>
#endif

#include <memory>

namespace utility {

/**
 * @brief A custom allocator that uses the memory pool.
 * @tparam T The data type
 */
template <typename T>
class PoolAllocator {
public:
    template <class U>
    friend class PoolAllocator;

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || (!defined(_MSVC_LANG) && __cplusplus < 202002L))
    using pointer = T*;
    using const_pointer = const T*;

    using reference = T&;
    using const_reference = const T&;

    template <typename _Tp1>
    struct rebind {
        using other = PoolAllocator<_Tp1>;
    };
#endif

    using is_always_equal = std::false_type;

    constexpr PoolAllocator() noexcept = default;

    template <class U>
    constexpr explicit PoolAllocator(const PoolAllocator<U>& a) noexcept
        : alloc(a.alloc) {
    }

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || (!defined(_MSVC_LANG) && __cplusplus < 202002L))
    ~PoolAllocator() = default;
#endif

    [[nodiscard]] constexpr T* allocate(const size_type n, [[maybe_unused]] const void* const /*hint*/ = nullptr) {
        if (n <= 1) {
            return MemoryPool<T, 1>::get_pointer();
        }

        if (n < 16) {
            return MemoryPool<T, 16>::get_pointer();
        }

        if (n < 64) {
            return MemoryPool<T, 64>::get_pointer();
        }

        return alloc.allocate(n);
    }

    constexpr void deallocate(T* p, size_type n) {
        if (n <= 1) {
            MemoryPool<T, 1>::return_pointer(p);
            return;
        }

        if (n < 16) {
            MemoryPool<T, 16>::return_pointer(p);
            return;
        }

        if (n < 64) {
            MemoryPool<T, 64>::return_pointer(p);
            return;
        }

        alloc.deallocate(p, n);
    }

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || (!defined(_MSVC_LANG) && __cplusplus < 202002L))
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
        ::new ((void*)p) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) {
        p->~U();
    }
#endif

    [[nodiscard]] friend constexpr bool operator==(const PoolAllocator&, const PoolAllocator&) = default;

private:
    std::allocator<T> alloc{};
};

} // namespace utility
