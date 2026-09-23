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

#include "cpp-utility/sequences/Index.hpp"

#include <cstddef>
#include <tuple>

namespace utility {

/** @brief Binds a managed-chunk template and its underlying chunk so only the unit size remains variable. */
template <template <typename chunk_type_inner, std::size_t size_of_allocation_unit> typename managed_chunk_type, typename chunk_type>
struct ChunkCurry {
    template <std::size_t size_of_allocation_unit_>
    using type = managed_chunk_type<chunk_type, size_of_allocation_unit_>;
};

/**
 * @brief Routes byte requests to the smallest of several strictly increasing fixed-size chunk pools.
 *
 * Requests larger than the largest bucket are not served and return nullptr/false/zero as appropriate. Deallocation
 * must use the same byte count as allocation; the selected managed chunk then verifies pointer ownership. The
 * address accessors expose the selected bucket's half-open storage interval for diagnostics.
 */
template <typename chunk_curry, std::size_t... sizes_>
    requires StrictlyIncreasing<sizes_...> && ((sizes_ > 0) && ...)
class CombinatorChunk {
public:
    /** @brief Obtains one unit from the smallest bucket that can hold @p size, or nullptr if none can. */
    [[nodiscard]] constexpr std::byte* get_pointer(const std::size_t size) {
        return do_allocate<0, sizes_...>(size);
    }

    /** @brief Returns @p ptr to the bucket selected by @p size; false means it was not owned/checked out there. */
    [[nodiscard]] constexpr bool return_pointer(const std::size_t size, std::byte* const ptr) {
        return do_return<0, sizes_...>(size, ptr);
    }

    /** @brief Returns the number of currently available units in the bucket selected by @p size. */
    [[nodiscard]] constexpr std::size_t get_size(const std::size_t size) const {
        return do_size<0, sizes_...>(size);
    }

    /** @brief Returns the selected bucket's first address, or nullptr when no bucket fits. */
    [[nodiscard]] constexpr const std::byte* get_first_address(const std::size_t size) const {
        return do_first<0, sizes_...>(size);
    }

    /** @brief Returns the selected bucket's exclusive end address, or nullptr when no bucket fits. */
    [[nodiscard]] constexpr const std::byte* get_last_address(const std::size_t size) const {
        return do_last<0, sizes_...>(size);
    }

private:
    template <std::size_t index, std::size_t C, std::size_t... Cs>
    [[nodiscard]] constexpr std::byte* do_allocate(const std::size_t n) {
        if (n <= C) {
            return std::get<index>(chunks).get_pointer(n);
        }
        return do_allocate<index + 1, Cs...>(n);
    }

    template <std::size_t>
    [[nodiscard]] constexpr std::byte* do_allocate([[maybe_unused]] const std::size_t n) {
        return nullptr;
    }

    template <std::size_t index, std::size_t C, std::size_t... Cs>
    [[nodiscard]] constexpr bool do_return(const std::size_t n, std::byte* const ptr) {
        if (n <= C) {
            return std::get<index>(chunks).return_pointer(ptr);
        }
        return do_return<index + 1, Cs...>(n, ptr);
    }

    template <std::size_t>
    [[nodiscard]] constexpr bool do_return([[maybe_unused]] const std::size_t n, [[maybe_unused]] std::byte* const ptr) {
        return false;
    }

    template <std::size_t index, std::size_t C, std::size_t... Cs>
    [[nodiscard]] constexpr std::size_t do_size(const std::size_t n) const {
        if (n <= C) {
            return std::get<index>(chunks).get_size();
        }
        return do_size<index + 1, Cs...>(n);
    }

    template <std::size_t>
    constexpr std::size_t do_size([[maybe_unused]] const std::size_t n) const {
        return 0;
    }

    template <std::size_t index, std::size_t C, std::size_t... Cs>
    [[nodiscard]] constexpr const std::byte* do_first(const std::size_t n) const {
        if (n <= C) {
            return std::get<index>(chunks).get_first_address();
        }
        return do_first<index + 1, Cs...>(n);
    }

    template <std::size_t>
    constexpr const std::byte* do_first([[maybe_unused]] const std::size_t n) const {
        return nullptr;
    }

    template <std::size_t index, std::size_t C, std::size_t... Cs>
    [[nodiscard]] constexpr const std::byte* do_last(const std::size_t n) const {
        if (n <= C) {
            return std::get<index>(chunks).get_last_address();
        }
        return do_last<index + 1, Cs...>(n);
    }

    template <std::size_t>
    constexpr const std::byte* do_last([[maybe_unused]] const std::size_t n) const {
        return nullptr;
    }

    std::tuple<typename chunk_curry::template type<sizes_>...> chunks{};
};

} // namespace utility
