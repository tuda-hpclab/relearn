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

#include "hash.hpp"

#include <utility>

namespace utility {

template <typename Type1, typename Type2>
struct hash<std::pair<Type1, Type2>> {
    [[nodiscard]] std::size_t operator()(const std::pair<Type1, Type2>& k) const {
        const auto first_hash = hash<Type1>{};
        const auto second_hash = hash<Type2>{};

        const auto& [val_1, val_2] = k;

        auto current_hash = first_hash(val_1);
        current_hash ^= second_hash(val_2) + 0x9e3779b9 + (current_hash << 6U) + (current_hash >> 2U);

        return current_hash;
    }
};

} // namespace utility
