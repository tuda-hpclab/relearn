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

#include <utility>

namespace utility {

/** Hashes both pair elements in order using utility::hash and hash_combine. */
template <typename Type1, typename Type2>
struct hash<std::pair<Type1, Type2>> {
    [[nodiscard]] std::size_t operator()(const std::pair<Type1, Type2>& k) const
        noexcept(noexcept(hash<Type1>{}(k.first)) && noexcept(hash<Type2>{}(k.second))) {
        const auto first_hash = hash<Type1>{};
        const auto second_hash = hash<Type2>{};

        const auto& [val_1, val_2] = k;

        auto current_hash = first_hash(val_1);
        current_hash = detail::hash_combine(current_hash, second_hash(val_2));

        return current_hash;
    }
};

} // namespace utility
