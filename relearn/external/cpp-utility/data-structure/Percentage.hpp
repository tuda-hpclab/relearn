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

#include "cpp-utility/Exception.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <compare>
#include <concepts>
#include <limits>
#include <ostream>
#include <type_traits>

namespace utility {

/**
 * @brief An immutable percentage in the closed interval [0, 100].
 *
 * A Percentage can be copied or moved to construct another object, but cannot be assigned a new value.
 * Operations that produce another percentage return a new object instead of modifying the existing one.
 *
 * @tparam T The floating-point type used to store the percentage
 */
template <std::floating_point T = float>
class Percentage {
public:
    using value_type = T;

    static constexpr value_type min_value = value_type{ 0 };
    static constexpr value_type max_value = value_type{ 100 };

    /**
     * @brief Constructs a percentage from its value.
     * @param value The percentage value in the closed interval [0, 100]
     * @exception Throws an Exception if value is outside [0, 100] or is not finite
     */
    template <typename U>
        requires(std::is_arithmetic_v<U> && !std::same_as<std::remove_cv_t<U>, bool>)
    constexpr explicit Percentage(const U value)
        : value_{ static_cast<value_type>(value) } {
        // Both infinities and NaN fail at least one of these comparisons.
        Exception::check(value >= U{ 0 } && value <= U{ 100 },
                         "Percentage::Percentage: The value must be finite and in [0, 100], was {}", value);
    }

    constexpr Percentage(const Percentage&) noexcept = default;
    constexpr Percentage(Percentage&&) noexcept = default;
    Percentage& operator=(const Percentage&) = delete;
    Percentage& operator=(Percentage&&) = delete;

    /**
     * @brief Returns the stored percentage value.
     * @return The value in the underlying type
     */
    [[nodiscard]] constexpr value_type get() const noexcept {
        return value_;
    }

    /**
     * @brief Converts the stored value to another arithmetic type.
     * @tparam U The requested result type; must be an arithmetic type other than bool
     * @return The percentage value converted with static_cast
     */
    template <typename U>
        requires(std::is_arithmetic_v<U> && !std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr U as() const noexcept {
        return static_cast<U>(value_);
    }

    /**
     * @brief Returns the smallest valid percentage.
     * @return 0 percent
     */
    [[nodiscard]] static constexpr Percentage min() noexcept {
        return Percentage{ min_value };
    }

    /**
     * @brief Returns the largest valid percentage.
     * @return 100 percent
     */
    [[nodiscard]] static constexpr Percentage max() noexcept {
        return Percentage{ max_value };
    }

    /**
     * @brief Constructs a percentage from a part and its whole
     * @tparam PartType The arithmetic type of part
     * @tparam WholeType The arithmetic type of whole
     * @param part The finite non-negative part
     * @param whole The finite positive whole; part must not be larger than whole
     * @exception Throws an Exception unless 0 <= part <= whole and whole is positive and finite
     * @return 100 * part / whole percent
     */
    template <typename PartType, typename WholeType>
        requires(std::is_arithmetic_v<PartType> && !std::same_as<std::remove_cv_t<PartType>, bool>
                 && std::is_arithmetic_v<WholeType> && !std::same_as<std::remove_cv_t<WholeType>, bool>)
    [[nodiscard]] static constexpr Percentage from_fraction(const PartType part, const WholeType whole) {
        using common_type = std::common_type_t<long double, value_type, PartType, WholeType>;
        const auto common_part = static_cast<common_type>(part);
        const auto common_whole = static_cast<common_type>(whole);

        Exception::check(common_part >= common_type{ 0 } && common_part <= common_whole
                             && common_whole > common_type{ 0 } && common_whole <= std::numeric_limits<common_type>::max(),
                         "Percentage::from_fraction: Expected finite values with 0 <= part <= whole and whole > 0, found part {} and whole {}",
                         part, whole);

        return Percentage{ common_part / common_whole * common_type{ 100 } };
    }

    /**
     * @brief Checks whether this percentage is smaller than another percentage.
     * @tparam U The other percentage's underlying type
     * @param other The other percentage
     * @return True iff this percentage is strictly smaller
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr bool smaller_than(const Percentage<U>& other) const noexcept {
        return *this < other;
    }

    /**
     * @brief Checks whether this percentage is smaller than a raw percentage value.
     * @tparam U The raw value's floating-point type
     * @param other The other value, which must be in [0, 100]
     * @exception Throws an Exception if other is not a valid percentage
     * @return True iff this percentage is strictly smaller
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr bool smaller_than(const U other) const {
        return smaller_than(Percentage<U>{ other });
    }

    /**
     * @brief Checks whether this percentage is larger than another percentage.
     * @tparam U The other percentage's underlying type
     * @param other The other percentage
     * @return True iff this percentage is strictly larger
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr bool larger_than(const Percentage<U>& other) const noexcept {
        return *this > other;
    }

    /**
     * @brief Checks whether this percentage is larger than a raw percentage value.
     * @tparam U The raw value's floating-point type
     * @param other The other value, which must be in [0, 100]
     * @exception Throws an Exception if other is not a valid percentage
     * @return True iff this percentage is strictly larger
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr bool larger_than(const U other) const {
        return larger_than(Percentage<U>{ other });
    }

    /**
     * @brief Returns the smaller of this and another percentage.
     * @tparam U The other percentage's underlying type
     * @param other The other percentage
     * @return The smaller percentage, stored in the common underlying type
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr auto min(const Percentage<U>& other) const noexcept {
        using common_type = std::common_type_t<value_type, U>;
        return Percentage<common_type>{ std::min(static_cast<common_type>(value_), static_cast<common_type>(other.get())) };
    }

    /**
     * @brief Returns the smaller of this percentage and a raw percentage value.
     * @tparam U The raw value's floating-point type
     * @param other The other value, which must be in [0, 100]
     * @exception Throws an Exception if other is not a valid percentage
     * @return The smaller percentage, stored in the common underlying type
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr auto min(const U other) const {
        return min(Percentage<U>{ other });
    }

    /**
     * @brief Returns the larger of this and another percentage.
     * @tparam U The other percentage's underlying type
     * @param other The other percentage
     * @return The larger percentage, stored in the common underlying type
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr auto max(const Percentage<U>& other) const noexcept {
        using common_type = std::common_type_t<value_type, U>;
        return Percentage<common_type>{ std::max(static_cast<common_type>(value_), static_cast<common_type>(other.get())) };
    }

    /**
     * @brief Returns the larger of this percentage and a raw percentage value.
     * @tparam U The raw value's floating-point type
     * @param other The other value, which must be in [0, 100]
     * @exception Throws an Exception if other is not a valid percentage
     * @return The larger percentage, stored in the common underlying type
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr auto max(const U other) const {
        return max(Percentage<U>{ other });
    }

    /**
     * @brief Clamps this percentage to the closed interval [lower, upper].
     * @tparam LowerType The lower bound's underlying type
     * @tparam UpperType The upper bound's underlying type
     * @param lower The inclusive lower bound
     * @param upper The inclusive upper bound
     * @exception Throws an Exception if lower is larger than upper
     * @return The clamped percentage, stored in the common underlying type
     */
    template <std::floating_point LowerType, std::floating_point UpperType>
    [[nodiscard]] constexpr auto clamp(const Percentage<LowerType>& lower, const Percentage<UpperType>& upper) const {
        using common_type = std::common_type_t<value_type, LowerType, UpperType>;
        const auto common_lower = static_cast<common_type>(lower.get());
        const auto common_upper = static_cast<common_type>(upper.get());
        Exception::check(common_lower <= common_upper,
                         "Percentage::clamp: The lower bound {} must not be larger than the upper bound {}", common_lower, common_upper);
        return Percentage<common_type>{ std::clamp(static_cast<common_type>(value_), common_lower, common_upper) };
    }

    /**
     * @brief Clamps this percentage to raw percentage bounds.
     * @tparam LowerType The lower bound's floating-point type
     * @tparam UpperType The upper bound's floating-point type
     * @param lower The inclusive lower bound, which must be in [0, 100]
     * @param upper The inclusive upper bound, which must be in [0, 100]
     * @exception Throws an Exception if a bound is invalid or lower is larger than upper
     * @return The clamped percentage, stored in the common underlying type
     */
    template <std::floating_point LowerType, std::floating_point UpperType>
    [[nodiscard]] constexpr auto clamp(const LowerType lower, const UpperType upper) const {
        return clamp(Percentage<LowerType>{ lower }, Percentage<UpperType>{ upper });
    }

    /**
     * @brief Returns the percentage needed to reach 100 percent.
     * @return 100 percent minus this percentage
     */
    [[nodiscard]] constexpr Percentage complement() const noexcept {
        return Percentage{ max_value - value_ };
    }

    /**
     * @brief Returns the percentage as a multiplication factor in the closed interval [0, 1]
     * @return The stored percentage divided by 100
     */
    [[nodiscard]] constexpr value_type factor() const noexcept {
        return value_ / value_type{ 100 };
    }

    /**
     * @brief Calculates this percentage of a value.
     * @tparam U The arithmetic input type
     * @param value The value whose percentage should be calculated
     * @return value multiplied by this percentage divided by 100, in the common type
     */
    template <typename U>
        requires(std::is_arithmetic_v<U> && !std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr auto of(const U value) const noexcept {
        using common_type = std::common_type_t<value_type, U>;
        const auto factor = static_cast<common_type>(value_) / common_type{ 100 };
        return static_cast<common_type>(value) * factor;
    }

    /**
     * @brief Checks whether part is strictly more than this percentage of whole.
     *
     * For example, Percentage{60}.is_exceeded_by(7, 10) is true, while calls with 6 or less are false.
     * The strict comparison also means that 0 is not more than 0 percent of any non-negative whole.
     *
     * @tparam PartType The arithmetic type of part
     * @tparam WholeType The arithmetic type of whole
     * @param part The non-negative part
     * @param whole The finite non-negative whole; part must not be larger than whole
     * @exception Throws an Exception unless 0 <= part <= whole and both values are finite
     * @return True iff part is strictly more than this percentage of whole
     */
    template <typename PartType, typename WholeType>
        requires(std::is_arithmetic_v<PartType> && !std::same_as<std::remove_cv_t<PartType>, bool>
                 && std::is_arithmetic_v<WholeType> && !std::same_as<std::remove_cv_t<WholeType>, bool>)
    [[nodiscard]] constexpr bool is_exceeded_by(const PartType part, const WholeType whole) const {
        using common_type = std::common_type_t<long double, value_type, PartType, WholeType>;
        const auto common_part = static_cast<common_type>(part);
        const auto common_whole = static_cast<common_type>(whole);

        // Comparing to max also rejects positive infinity; the other comparisons reject NaN and negative infinity.
        Exception::check(common_part >= common_type{ 0 } && common_part <= common_whole
                             && common_whole <= std::numeric_limits<common_type>::max(),
                         "Percentage::is_exceeded_by: Expected finite values with 0 <= part <= whole, found part {} and whole {}",
                         part, whole);

        if (common_whole == common_type{ 0 }) {
            return false;
        }

        const auto factor = static_cast<common_type>(value_) / common_type{ 100 };
        return common_part / common_whole > factor;
    }

    /**
     * @brief Checks whether two percentages have exactly equal stored values after promotion to a common type.
     * @tparam U The other percentage's underlying type
     * @param other The other percentage
     * @return True iff the values are equal
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr bool operator==(const Percentage<U>& other) const noexcept {
        using common_type = std::common_type_t<value_type, U>;
        return static_cast<common_type>(value_) == static_cast<common_type>(other.get());
    }

    /**
     * @brief Compares two percentages after promotion to a common type.
     * @tparam U The other percentage's underlying type
     * @param other The other percentage
     * @return The three-way comparison result
     */
    template <std::floating_point U>
    [[nodiscard]] constexpr auto operator<=>(const Percentage<U>& other) const noexcept {
        using common_type = std::common_type_t<value_type, U>;
        return static_cast<common_type>(value_) <=> static_cast<common_type>(other.get());
    }

    /**
     * @brief Prints the value followed by a percent sign.
     * @param output_stream The stream to which the percentage should be printed
     * @param percentage The percentage that should be printed
     * @return A reference to output_stream
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const Percentage& percentage) {
        return output_stream << percentage.value_ << '%';
    }

private:
    const value_type value_;
};

} // namespace utility

/**
 * @brief Formats a Percentage<T> as its value followed by a percent sign.
 *      The format specification is applied to the value, e.g., "{:.2f}" produces "12.50%".
 * @tparam T The percentage's underlying type
 */
template <std::floating_point T>
struct fmt::formatter<utility::Percentage<T>> : fmt::formatter<T> {
    /**
     * @brief Formats the percentage into the output of the format context.
     * @param percentage The percentage that should be formatted
     * @param ctx The format context that provides the output iterator
     * @return The output iterator past the formatted percentage
     */
    auto format(const utility::Percentage<T>& percentage, fmt::format_context& ctx) const {
        auto out = fmt::formatter<T>::format(percentage.get(), ctx);
        *out++ = '%';
        return out;
    }
};
