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

#include <concepts>
#include <ranges>

namespace utility {

/**
 * @brief Checks if the intersection of the two containers is non-empty, i.e., if they share at least one element.
 *      Returns as soon as the first common element is found
 * @tparam SetType The type of the lookup container. Must provide contains(...) for the elements of the range,
 *      e.g., std::unordered_set, std::set, or a map searched by its keys
 * @tparam RangeType The type of the range with the candidate elements
 * @param set The lookup container
 * @param elements The candidate elements
 * @return True iff the intersection of the two containers is non-empty
 */
template <typename SetType, std::ranges::input_range RangeType>
    requires requires(const SetType& set, std::ranges::range_reference_t<RangeType> element) {
        { set.contains(element) } -> std::convertible_to<bool>;
    }
[[nodiscard]] bool containers_intersect(const SetType& set, RangeType&& elements) {
    for (auto&& element : elements) {
        if (set.contains(element)) {
            return true;
        }
    }

    return false;
}

} // namespace utility
