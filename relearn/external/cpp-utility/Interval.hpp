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

#include <concepts>
#include <cstdint>
#include <span>
#include <type_traits>

namespace utility {

/**
 * Represents a closed interval and a sampling frequency. A valid interval hits
 * begin, begin + |frequency|, begin + 2*|frequency|, ... while the sampled step is <= end.
 * Reversed intervals are treated as empty, and a zero frequency hits only begin.
 * @tparam s_type The type of the steps, e.g., std::uint32_t
 */
template <std::integral s_type = std::uint32_t>
    requires(!std::same_as<std::remove_cv_t<s_type>, bool>)
struct Interval {
    using step_type = s_type;

    step_type begin{};
    step_type end{};
    step_type frequency{};

    friend constexpr bool operator==(const Interval&, const Interval&) noexcept = default;

    /**
     * @brief Checks whether a step lies in the closed interval and is aligned with its frequency.
     * A negative frequency is treated by magnitude; a zero frequency hits only begin.
     * @param current_step The current step
     * @return True if the interval hits current_step
     */
    [[nodiscard]] constexpr bool hits_step(const step_type current_step) const noexcept {
        if (current_step < begin) {
            return false;
        }

        if (current_step > end) {
            return false;
        }

        if (frequency == step_type{ 0 }) {
            return current_step == begin;
        }

        using unsigned_step_type = std::make_unsigned_t<std::remove_cv_t<step_type>>;
        const auto relative_offset = static_cast<unsigned_step_type>(current_step) - static_cast<unsigned_step_type>(begin);

        auto frequency_magnitude = static_cast<unsigned_step_type>(frequency);
        if constexpr (std::signed_integral<step_type>) {
            if (frequency < step_type{ 0 }) {
                frequency_magnitude = unsigned_step_type{ 0 } - frequency_magnitude;
            }
        }

        return relative_offset % frequency_magnitude == 0;
    }

    /**
     * @brief Checks whether two valid closed intervals overlap, ignoring their frequencies.
     * Reversed intervals are empty and therefore never intersect.
     * @param other The other interval
     * @return True iff the intervals intersect
     */
    [[nodiscard]] constexpr bool check_for_intersection(const Interval& other) const noexcept {
        return begin <= end && other.begin <= other.end && begin <= other.end && other.begin <= end;
    }

    /**
     * @brief Pairwise-checks whether any two valid closed intervals overlap.
     * @param intervals All intervals
     * @return True iff any two intervals intersect
     */
    [[nodiscard]] static constexpr bool check_intervals_for_intersection(const std::span<const Interval> intervals) noexcept {
        for (auto i = std::size_t{ 0 }; i < intervals.size(); i++) {
            for (auto j = i + std::size_t{ 1 }; j < intervals.size(); j++) {
                const auto intervals_intersect = intervals[i].check_for_intersection(intervals[j]);
                if (intervals_intersect) {
                    return true;
                }
            }
        }

        return false;
    }
};

} // namespace utility
