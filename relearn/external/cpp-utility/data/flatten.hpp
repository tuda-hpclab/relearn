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

#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

namespace utility {

/**
 * @brief Flattens the data into a single vector
 * @tparam T The type of the data
 * @param data A vector of vectors of data
 * @return The flatten data
 */
template <typename T>
[[nodiscard]] std::vector<T> flatten_data(const std::span<const std::vector<T>> data) {
    const auto number_elements = std::accumulate(data.begin(), data.end(), std::size_t{ 0 }, [](const auto sum, const auto& part) { return sum + part.size(); });

    auto flattened_data = std::vector<T>();
    flattened_data.reserve(number_elements);

    for (const auto& part : data) {
        flattened_data.insert(flattened_data.end(), part.begin(), part.end());
    }

    return flattened_data;
}

} // namespace utility
