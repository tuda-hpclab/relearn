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

#include "cpp-utility/Exception.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace utility {

/**
 * @brief Copies all parts into a single vector, preserving their order and the order within each part
 * @tparam T The type of the data
 * @param data A vector of vectors of data
 * @exception Throws an Exception if the combined size exceeds the maximum size of std::vector<T>
 * @return The flattened data
 */
template <typename T>
[[nodiscard]] std::vector<T> flatten_data(const std::span<const std::vector<T>> data) {
    auto flattened_data = std::vector<T>();
    auto number_elements = std::size_t{ 0 };
    for (const auto& part : data) {
        Exception::check(part.size() <= flattened_data.max_size() - number_elements,
                         "flatten_data: combined size exceeds the maximum vector size of {}", flattened_data.max_size());
        number_elements += part.size();
    }

    flattened_data.reserve(number_elements);

    for (const auto& part : data) {
        flattened_data.insert(flattened_data.end(), part.begin(), part.end());
    }

    return flattened_data;
}

} // namespace utility
