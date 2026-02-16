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

#include <unordered_set>
#include <vector>

class SetUtil {
public:
    template <typename T>
    [[nodiscard]] static bool containers_have_common_element(const std::unordered_set<T>& set, const std::vector<T>& vec) {
        for (const auto& element : vec) {
            if (set.contains(element)) {
                return true;
            }
        }
        return false;
    }
};