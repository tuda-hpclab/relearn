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

#include "cpp-utility/data-structure/Vec3.hpp"
#include "cpp-utility/hash/hash.hpp"

namespace utility {

/** Hashes the x, y, and z components in order using utility::hash and hash_combine. */
template <typename Type>
struct hash<Vec3<Type>> {
    [[nodiscard]] std::size_t operator()(const Vec3<Type>& k) const
        noexcept(noexcept(hash<Type>{}(k.get_x())) && noexcept(hash<Type>{}(k.get_y())) && noexcept(hash<Type>{}(k.get_z()))) {
        const auto component_hash = hash<Type>{};

        auto current_hash = component_hash(k.get_x());
        current_hash = detail::hash_combine(current_hash, component_hash(k.get_y()));
        current_hash = detail::hash_combine(current_hash, component_hash(k.get_z()));

        return current_hash;
    }
};

} // namespace utility
