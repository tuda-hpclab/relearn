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

#include <range/v3/range/conversion.hpp>

#include <span>
#include <type_traits>
#include <vector>

namespace utility {
/**
 * @brief Constructs a vector from a span
 * @tparam T The type of elements
 * @param span The non-owning span
 * @return A vector with a copy of the elements
 */
template <typename T>
static std::vector<std::decay_t<T>> vectorify_span(const std::span<T> span) {
    return span | ranges::to_vector;
}
} // namespace utility
