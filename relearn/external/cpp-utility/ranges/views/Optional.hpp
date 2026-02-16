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

#include <range/v3/view/cache1.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

namespace utility::views {
inline constexpr auto optional_values = ranges::views::cache1
                                        | ranges::views::filter([](const auto& opt) { return static_cast<bool>(opt); })
                                        | ranges::views::transform([]<typename T>(T&& stimulus) { return *std::forward<T>(stimulus); });
} // namespace utility::views
