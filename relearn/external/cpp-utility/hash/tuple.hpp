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

#include "cpp-utility/hash/hash.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace utility {

/**
 * @brief Hashes a tuple of arbitrary arity by using the hash of the first element as seed
 *      and combining the hashes of the remaining elements one by one.
 *      A two-element tuple hashes exactly like the corresponding std::pair
 * @tparam FirstType The type of the first element
 * @tparam OtherTypes The types of the remaining elements
 */
template <typename FirstType, typename... OtherTypes>
struct hash<std::tuple<FirstType, OtherTypes...>> {
    [[nodiscard]] std::size_t operator()(const std::tuple<FirstType, OtherTypes...>& k) const noexcept(
        std::is_nothrow_invocable_v<hash<FirstType>, const FirstType&> && (std::is_nothrow_invocable_v<hash<OtherTypes>, const OtherTypes&> && ...)) {
        return std::apply(
            [](const FirstType& first_value, const OtherTypes&... other_values) {
                auto current_hash = hash<FirstType>{}(first_value);
                ((current_hash = detail::hash_combine(current_hash, hash<OtherTypes>{}(other_values))), ...);
                return current_hash;
            },
            k);
    }
};

template <>
struct hash<std::tuple<>> {
    /** Returns the fixed seed for a tuple without components. */
    [[nodiscard]] std::size_t operator()(const std::tuple<>&) const noexcept {
        return 0;
    }
};

} // namespace utility
