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

#include "cpp-utility/data-structure/TaggedID.hpp"
#include "cpp-utility/hash/hash.hpp"

#include <concepts>
#include <cstddef>

namespace utility {

/**
 * Uses the identifier's packed value, including its flags, as its hash.
 * TaggedID.hpp specializes std::hash the same way, so this header is only needed where utility::hash is used explicitly.
 */
template <std::unsigned_integral Type, std::size_t NumFlags, typename Traits>
struct hash<TaggedID<Type, NumFlags, Traits>> {
    [[nodiscard]] std::size_t operator()(const TaggedID<Type, NumFlags, Traits>& k) const noexcept(noexcept(k.hash_value())) {
        return k.hash_value();
    }
};

} // namespace utility
