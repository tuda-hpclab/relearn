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

#include <cstddef>
#include <functional>

namespace utility {

namespace detail {

/** Combines an already accumulated hash with one additional component hash. */
[[nodiscard]] constexpr std::size_t hash_combine(const std::size_t seed, const std::size_t component) noexcept {
    return seed ^ (component + std::size_t{ 0x9e3779b9U } + (seed << 6U) + (seed >> 2U));
}

} // namespace detail

/**
 * @brief Uses std::hash as the default hashing strategy for a type.
 * Specializations in this library recursively use utility::hash for their components.
 */
template <typename Type>
struct hash {
    [[nodiscard]] std::size_t operator()(const Type& value) const noexcept(noexcept(std::hash<Type>{}(value))) {
        return std::hash<Type>{}(value);
    }
};

} // namespace utility
