#pragma once

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || (!defined(_MSVC_LANG) && __cplusplus < 202002L))
#include <limits>
#include <utility>
#endif

#include <memory>

template <typename T>
class RelearnAllocator : public std::allocator<T> {
public:
    using value_type = T;
    using size_type = std::size_t;
    using differnece_type = std::ptrdiff_t;

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || (!defined(_MSVC_LANG) && __cplusplus < 202002L))
    using pointer = T*;
    using const_pointer = const T*;

    using reference = T&;
    using const_reference = const T&;

    template <typename _Tp1>
    struct rebind {
        using other = RelearnAllocator<_Tp1>;
    };
#endif

    using is_always_equal = std::true_type;

    constexpr RelearnAllocator() noexcept
        : std::allocator<T>() { }

    template <class U>
    constexpr explicit RelearnAllocator(const RelearnAllocator<U>& a) noexcept
        : std::allocator<T>(a) {
    }

    constexpr RelearnAllocator(const RelearnAllocator& a) noexcept = default;
    RelearnAllocator& operator=(const RelearnAllocator&) = default;

    RelearnAllocator(RelearnAllocator&&) noexcept = default;
    RelearnAllocator& operator=(RelearnAllocator&&) noexcept = default;

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || (!defined(_MSVC_LANG) && __cplusplus < 202002L))
    ~RelearnAllocator() {
    }
#else
    constexpr ~RelearnAllocator() {
    }
#endif

    [[nodiscard]] constexpr T* allocate(const size_type n, const void* /*hint*/ = nullptr) {
        if (n == 0) {
            return nullptr;
        }

        return alloc.allocate(n);
    }

    constexpr void deallocate(T* p, size_type n) {
        if (p == nullptr) {
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
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* p) {
        p->~U();
    }
#endif

private:
    std::allocator<T> alloc{};
};
