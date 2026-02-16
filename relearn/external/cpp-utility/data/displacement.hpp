#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cpp-utility/Exception.hpp"

#include <concepts>
#include <numeric>
#include <span>
#include <vector>

namespace utility {

/**
 * @brief Calculates the necessary displacements for a bunch of sizes such that the elements are tightly packed.
 *      For sizes = [a, b, c, ..., y, z] returns [0, a, a+b, ..., a+b+...+y]
 * @tparam count_type The type for the displacements/sizes
 * @param sizes The sizes, not empty
 * @exception Throws an Exception if sizes is empty
 * @return The displacements for the sizes
 */
template <std::integral count_type>
[[nodiscard]] std::vector<count_type> calculate_displacements(const std::span<const count_type> sizes) {
    Exception::check(!sizes.empty(), "Util::calculate_displacements: sizes must not be empty");
    const auto number_elements = sizes.size();

    auto displs = std::vector<count_type>(number_elements, 0);
    std::partial_sum(sizes.begin(), sizes.end() - 1, displs.begin() + 1);

    return displs;
}

} // namespace utility
