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

#include "cpp-utility/data-structure/BoundedInteger.hpp"
#include "cpp-utility/hash/hash.hpp"

#include <concepts>
#include <cstddef>

namespace utility {

/** Hashes the bounded integer's underlying value using utility::hash. */
template <std::integral T, T Minimum, T Maximum>
    requires(!std::same_as<std::remove_cv_t<T>, bool> && std::same_as<T, std::remove_cv_t<T>>)
struct hash<BoundedInteger<T, Minimum, Maximum>> {
    [[nodiscard]] std::size_t operator()(const BoundedInteger<T, Minimum, Maximum>& value) const
        noexcept(noexcept(hash<T>{}(value.get()))) {
        return hash<T>{}(value.get());
    }
};

} // namespace utility
