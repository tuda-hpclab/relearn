//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Range v3 library
//
//  Copyright Filip Matzner 2015
//
//  Use, modification and distribution is subject to the
//  Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// Project home: https://github.com/ericniebler/range-v3
//

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

// This header includes the implementation of the shuffle function from libcxx of the LLVM Project.
// The original source of the algorithm is in `libcxx/include/__algorithm/shuffle.h` and was modified to
// use boost's `uniform_int_distribution` instead of the standard one, to provide portability.

// Additionally, a range-v3 shuffle action was implemented using the range-v3 implementation as reference

#pragma once

#include <boost/random/uniform_int_distribution.hpp>
#include <iterator>
#include <range/v3/action/action.hpp>
#include <range/v3/functional/bind_back.hpp>
#include <range/v3/functional/reference_wrapper.hpp>
#include <range/v3/iterator/concepts.hpp>
#include <range/v3/iterator/operations.hpp>
#include <range/v3/iterator/traits.hpp>
#include <range/v3/range/access.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/utility/random.hpp>
#include <type_traits>

/**
 * @brief Implementation of std::shuffle with a portable random number distribution
 *
 * Original source: libcxx from the LLVM Project
 * Modified: names, formatting, variable scopes, random distribution, range-v3 concepts
 *
 * @tparam RandomAccessIterator iterator type for the range
 * @tparam Sentinel sentinel type for the range
 * @tparam UniformRandomNumberGenerator type of the random number generator
 * @param first begin iterator
 * @param sentinel sentinel
 * @param gen random number generator
 */
template <ranges::random_access_iterator RandomAccessIterator, ranges::sized_sentinel_for<RandomAccessIterator> Sentinel, typename UniformRandomNumberGenerator>
    requires ranges::permutable<RandomAccessIterator> && ranges::uniform_random_bit_generator<std::remove_cvref_t<UniformRandomNumberGenerator>>
void shuffle(RandomAccessIterator first, Sentinel sentinel,
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    UniformRandomNumberGenerator&& gen) {
    using difference_type = ranges::iter_difference_t<RandomAccessIterator>;
    using distribution_type = boost::random::uniform_int_distribution<ptrdiff_t>;
    using distribution_param_type = typename distribution_type::param_type;

    if (difference_type distance = ranges::distance(first, sentinel); distance > 1) {
        const distribution_type uid;
        for (--sentinel, --distance; first < sentinel; ++first, --distance) { // NOLINT(hicpp-use-nullptr,modernize-use-nullptr)
            if (const difference_type random_val = uid(gen, distribution_param_type(0, distance));
                random_val != difference_type(0)) {
                ranges::iter_swap(first, first + random_val);
            }
        }
    }
}

template <ranges::random_access_range Range, typename UniformRandomNumberGenerator>
void shuffle(Range& range,
    UniformRandomNumberGenerator&& gen) {
    shuffle(ranges::begin(range), ranges::end(range), std::forward<UniformRandomNumberGenerator>(gen));
}

namespace actions {
namespace detail {
    struct ShuffleFn {
        template <typename UniformRandomNumberGenerator>
            requires ranges::uniform_random_bit_generator<std::remove_cvref_t<UniformRandomNumberGenerator>>
        [[nodiscard]] constexpr auto operator()(UniformRandomNumberGenerator& gen) const {
            return ranges::make_action_closure(
                ranges::bind_back(
                    ShuffleFn{},
                    ranges::detail::reference_wrapper_<UniformRandomNumberGenerator>(gen)));
        }

        template <typename UniformRandomNumberGenerator>
            requires ranges::uniform_random_bit_generator<std::remove_cvref_t<UniformRandomNumberGenerator>>
        [[nodiscard]] constexpr auto operator()(UniformRandomNumberGenerator&& gen) const {
            return ranges::make_action_closure(ranges::bind_back(ShuffleFn{}, std::forward<UniformRandomNumberGenerator>(gen)));
        }

        template <ranges::random_access_range Range, typename UniformRandomNumberGenerator>
            requires ranges::permutable<ranges::iterator_t<Range>> && ranges::uniform_random_bit_generator<std::remove_cvref_t<UniformRandomNumberGenerator>>
        [[nodiscard]] constexpr auto operator()(Range&& range, UniformRandomNumberGenerator&& gen) const -> decltype(auto) {
            shuffle(range, std::forward<UniformRandomNumberGenerator>(gen));
            return std::forward<Range>(range);
        }

        template <ranges::random_access_range Range, typename UniformRandomNumberGenerator>
            requires ranges::permutable<ranges::iterator_t<Range>> && ranges::uniform_random_bit_generator<std::remove_cvref_t<UniformRandomNumberGenerator>>
        [[nodiscard]] constexpr auto operator()(Range&& range, ranges::detail::reference_wrapper_<UniformRandomNumberGenerator> gen) const -> decltype(auto) {
            shuffle(range, gen.get());
            return std::forward<Range>(range);
        }
    };
} // namespace detail

inline constexpr ranges::actions::action_closure<detail::ShuffleFn> shuffle{};
} // namespace actions
