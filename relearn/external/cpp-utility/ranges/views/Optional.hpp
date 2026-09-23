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

#include <range/v3/view/cache1.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

#include <utility>

namespace utility::views {

/**
 * @brief Single-pass view that drops disengaged optional-/pointer-like elements and yields present values by value.
 *
 * Elements must be contextually convertible to bool, dereferenceable, and copy/move constructible into the cache.
 * cache1 ensures that input ranges are dereferenced only once across filter and transform; consequently references to
 * source elements are intentionally not exposed and mutations of yielded values do not write through to the source.
 */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
// GCC's flow analysis loses track of initialization through range-v3's heavily inlined,
// pointer-cast-based cursor/adaptor chain when this dereference is instantiated for types
// like std::string, and spuriously flags the moved-from value as possibly uninitialized.
// The dereference is always safe here: filter has already checked the element is engaged.
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
inline constexpr auto optional_values = ranges::views::cache1
                                        | ranges::views::filter([](const auto& opt) { return static_cast<bool>(opt); })
                                        | ranges::views::transform([]<typename T>(T&& stimulus) { return *std::forward<T>(stimulus); });
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

} // namespace utility::views
