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

#include "cpp-utility/memory/MemoryPool.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief A stateless standard allocator backed by three shared fixed-size memory pools.
 *
 * Requests for up to 1, 16, or 64 elements use the correspondingly sized pool; larger requests use
 * std::allocator directly. Pools are shared by all PoolAllocator instances of the same value type, and access is
 * serialized so equal same-value-type allocator instances can safely deallocate each other's storage from different threads.
 * The shared pools and mutex intentionally have process lifetime, which also makes the allocator safe for containers with
 * static storage duration. Storage is uninitialized, as required by the standard allocator interface.
 *
 * @tparam T The allocated element type.
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

    using is_always_equal = std::true_type;

    constexpr PoolAllocator() noexcept = default;

    template <class U>
    constexpr PoolAllocator(const PoolAllocator<U>&) noexcept {
    }

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || (!defined(_MSVC_LANG) && __cplusplus < 202002L))
    ~PoolAllocator() = default;
#endif

    /**
     * @brief Allocates uninitialized storage for @p n elements, using the smallest fitting pool.
     * @param n The element count; zero is delegated to std::allocator.
     * @param hint Ignored allocation hint retained for the legacy allocator interface.
     * @exception std::bad_array_new_length If @p n exceeds max_size().
     * @exception std::bad_alloc If allocation fails.
     */
    [[nodiscard]] T* allocate(const size_type n, [[maybe_unused]] const void* const hint = nullptr) {
        if (n > std::allocator_traits<std::allocator<T>>::max_size(alloc)) {
            throw std::bad_array_new_length{};
        }

        if (n == 0) {
            return alloc.allocate(0);
        }

        if (n <= 1) {
            return allocate_from_pool<1>(n);
        }

        if (n <= 16) {
            return allocate_from_pool<16>(n);
        }

        if (n <= 64) {
            return allocate_from_pool<64>(n);
        }

        return alloc.allocate(n);
    }

    /**
     * @brief Returns storage previously obtained by an equal PoolAllocator with the same @p n.
     * @param p The allocated pointer; nullptr is ignored.
     * @param n The exact element count passed to allocate().
     */
    void deallocate(T* const p, const size_type n) {
        if (p == nullptr) {
            return;
        }

        if (n == 0 || n > 64) {
            alloc.deallocate(p, n);
            return;
        }

        if (n <= 1) {
            deallocate_to_pool<1>(p, n);
            return;
        }

        if (n <= 16) {
            deallocate_to_pool<16>(p, n);
            return;
        }

        if (n <= 64) {
            deallocate_to_pool<64>(p, n);
            return;
        }
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

    template <typename U>
    [[nodiscard]] friend constexpr bool operator==(const PoolAllocator&, const PoolAllocator<U>&) noexcept {
        return std::is_same_v<T, U>;
    }

private:
    static constexpr std::size_t target_pool_bytes = 1024UL * 1024UL;

    template <std::size_t chunk_size>
    [[nodiscard]] static constexpr std::size_t pool_capacity() noexcept {
        if constexpr (chunk_size > target_pool_bytes / sizeof(T)) {
            return 0;
        }
        return target_pool_bytes / (chunk_size * sizeof(T));
    }

    template <std::size_t chunk_size>
    [[nodiscard]] T* allocate_from_pool(const size_type n) {
        if constexpr (pool_capacity<chunk_size>() == 0) {
            return alloc.allocate(n);
        } else {
            auto lock = std::scoped_lock{ pool_mutex() };
            return pool<chunk_size>().get_pointer();
        }
    }

    template <std::size_t chunk_size>
    void deallocate_to_pool(T* const p, const size_type n) {
        if constexpr (pool_capacity<chunk_size>() == 0) {
            alloc.deallocate(p, n);
        } else {
            auto lock = std::scoped_lock{ pool_mutex() };
            pool<chunk_size>().return_pointer(p);
        }
    }

    template <std::size_t chunk_size>
    [[nodiscard]] static MemoryPool<T, chunk_size>& pool() {
        static auto* const instance = new MemoryPool<T, chunk_size>{ pool_capacity<chunk_size>() };
        return *instance;
    }

    [[nodiscard]] static std::mutex& pool_mutex() {
        static auto* const mutex = new std::mutex{};
        return *mutex;
    }

    std::allocator<T> alloc{};
};

} // namespace utility
