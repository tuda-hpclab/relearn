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

#include "neurons/enums/UpdateStatus.h"
#include "util/RelearnException.h"

#include <span>
#include <tuple>
#include <type_traits>

namespace Util {
/**
 * @brief Calculates the minimum, maximum, and sum over all values in the span, for which the disable flags are not enabled
 * @tparam T Must be a arithmetic (floating point or integral)
 * @param values The values that should be reduced
 * @param disable_flags The flags that indicate which values to skip
 * @exception Throws a RelearnException if (a) values.empty(), (b) values.size() != disable_flags.size(), (c) all values are disabled
 * @return Returns a tuple with (1) minimum and (2) maximum value from values, (3) the sum of all enabled values and (4) the number of enabled values
 */
template <typename T>
std::tuple<T, T, T, std::size_t> min_max_acc(const std::span<const T>& values, const std::span<const UpdateStatus> disable_flags) {
    static_assert(std::is_arithmetic_v<T>);

    RelearnException::check(!values.empty(), "Util::min_max_acc: values are empty");
    RelearnException::check(values.size() == disable_flags.size(), "Util::min_max_acc: values and disable_flags had different sizes");

    auto first_index = std::size_t{ 0 };

    while (first_index < values.size() && disable_flags[first_index] != UpdateStatus::Enabled) {
        first_index++;
    }

    RelearnException::check(first_index != values.size(), "Util::min_max_acc: all were disabled");

    auto min = values[first_index];
    auto max = values[first_index];
    auto acc = values[first_index];

    auto num_values = std::size_t{ 1 };

    for (auto i = first_index + 1; i < values.size(); i++) {
        if (disable_flags[i] != UpdateStatus::Enabled) {
            continue;
        }

        const auto& current_value = values[i];

        if (current_value < min) {
            min = current_value;
        } else if (current_value > max) {
            max = current_value;
        }

        acc += current_value;
        num_values++;
    }

    return { min, max, acc, num_values };
}
} // namespace Util