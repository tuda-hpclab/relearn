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

#include "cpp-utility/Cast.hpp"

#include <concepts>
#include <cstddef>
#include <iterator>
#include <vector>

namespace utility {

/**
 * @brief Resizes the vector, constructing every element that is added in place from the given arguments.
 *      Shrinking erases the elements behind the new size, an unchanged size does nothing.
 *      Unlike std::vector::resize, this needs neither a default constructible element type nor a
 *      ready-made element to copy from, so it also works for a type that is only constructible from arguments
 * @tparam T The type of the elements
 * @tparam Allocator The allocator of the vector
 * @tparam SizeType The integral type of the requested size
 * @tparam Args The types of the constructor arguments
 * @param values The vector to resize
 * @param new_size The requested number of elements, must be non-negative
 * @param args The arguments every added element is constructed from. They are used once per added
 *      element and therefore copied instead of moved. They must not refer to an element of @p values,
 *      because growing the vector can invalidate such a reference
 * @exception Throws an Exception if @p new_size is negative or not representable by std::size_t.
 *      Throws whatever the allocation or the constructor of T throws; the vector then holds the
 *      elements that were added before, i.e., it is left with a valid but unspecified size
 */
template <typename T, typename Allocator, std::integral SizeType, typename... Args>
    requires std::constructible_from<T, const Args&...>
void resize_with_args(std::vector<T, Allocator>& values, const SizeType new_size, const Args&... args) {
    const auto target_size = safe_cast<std::size_t>(new_size);

    if (target_size < values.size()) {
        using difference_type = typename std::vector<T, Allocator>::difference_type;
        values.erase(std::next(values.begin(), safe_cast<difference_type>(target_size)), values.end());
        return;
    }

    values.reserve(target_size);

    while (values.size() < target_size) {
        values.emplace_back(args...);
    }
}

} // namespace utility
