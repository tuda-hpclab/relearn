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

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace utility {

/**
 * @brief Reorganize the flattened data in global_data into vectors
 *
 * @tparam T The type of the data
 * @tparam count_type The type for the counts per part
 * @tparam displacement_type The type for the displacements per part
 *
 * @param global_data The data elements; flat
 * @param counts The counts of data per part
 * @param displacements The displacements of data elements per part
 *
 * @exception Throws an Exception if counts.size() != displacements.size(), a count or displacement
 *      is negative or not representable by std::size_t, or a requested part lies outside global_data
 *
 * @return std::vector<std::vector<T>> Deflattened data
 */
template <typename T, std::integral count_type, std::integral displacement_type>
[[nodiscard]] std::vector<std::vector<T>> reorganize_data(const std::span<const T> global_data,
                                                          const std::span<const count_type> counts, const std::span<const displacement_type> displacements) {
    const auto number_sizes = counts.size();
    const auto number_displacements = displacements.size();

    Exception::check(number_sizes == number_displacements, "reorganize_data: number of counts is unequal to number of displacements: {} vs {}", number_sizes, number_displacements);

    const auto number_elements = global_data.size();

    auto deflattened_data = std::vector<std::vector<T>>();
    deflattened_data.reserve(number_sizes);

    for (auto part = std::size_t{ 0 }; part < number_sizes; ++part) {
        const auto size = safe_cast<std::size_t>(counts[part]);
        const auto displacement = safe_cast<std::size_t>(displacements[part]);

        Exception::check(displacement <= number_elements && size <= number_elements - displacement,
                         "reorganize_data: size ({}) at displacement ({}) is too large for {} elements", size, displacement, number_elements);

        const auto span_for_part = global_data.subspan(displacement, size);
        deflattened_data.emplace_back(span_for_part.begin(), span_for_part.end());
    }

    return deflattened_data;
}

} // namespace utility
