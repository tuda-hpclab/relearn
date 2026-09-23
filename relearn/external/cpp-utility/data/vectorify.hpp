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

#include <span>
#include <type_traits>
#include <vector>

namespace utility {
/**
 * @brief Copies the elements of a span into a new owning vector
 * @tparam T The type of elements
 * @param span The non-owning span
 * @return A vector with a copy of the elements
 */
template <typename T>
[[nodiscard]] std::vector<std::remove_cv_t<T>> vectorify_span(const std::span<T> span) {
    return { span.begin(), span.end() };
}
} // namespace utility
