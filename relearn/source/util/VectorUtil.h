#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <algorithm>
#include <functional>
#include <vector>

class VectorUtil {
public:
    /**
     * Split a vector of discrete sortable (every element must have a defined successor) elements into a vector of vectors with consecutive elements.
     * Example: Input: Vector [ 12 3 9 4 5  13 7 15]
     * Output [ [3 4 5 [ 7 ] [ 9 ] [ 12 13 ] [ 15 ] ]
     * @tparam T Type of the elements. Must be totally orderable
     * @param vector The input vector
     * @param is_successor_or_equal Predictive that take two objects of T and returns if the second object is the successor of the first one or equal
     * @return Sorted vector of vector with consecutive elements
     */
    template <typename T>
        requires std::totally_ordered<T>
    [[nodiscard]] static std::vector<std::vector<T>> splitConsecutiveOrEqual(std::vector<T> vector, std::function<bool(const T&, const T&)> is_successor_or_equal) {
        std::ranges::sort(vector);

        auto result = std::vector<std::vector<T>>{};

        if (vector.empty()) {
            return result;
        }

        auto currentGroup = std::vector<T>{};
        currentGroup.push_back(vector[0]);

        for (auto i = std::size_t{ 1 }; i < vector.size(); ++i) {
            if (!is_successor_or_equal(vector[i - 1], vector[i])) {
                result.push_back(currentGroup);
                currentGroup.clear();
            }
            currentGroup.push_back(vector[i]);
        }

        result.push_back(currentGroup);
        return result;
    }
};
