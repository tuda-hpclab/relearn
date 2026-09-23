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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <type_traits>
#include <utility>

namespace utility {

namespace detail {

/** Compares arbitrary integral types without signedness conversions. Unlike std::cmp_less, this includes character types. */
template <std::integral Left, std::integral Right>
[[nodiscard]] constexpr bool bounded_integer_less(const Left left, const Right right) noexcept {
    if constexpr (std::is_signed_v<Left> == std::is_signed_v<Right>) {
        using comparison_type = std::conditional_t<std::is_signed_v<Left>, std::intmax_t, std::uintmax_t>;
        return static_cast<comparison_type>(left) < static_cast<comparison_type>(right);
    } else if constexpr (std::is_signed_v<Left>) {
        return left < Left{ 0 }
               || static_cast<std::uintmax_t>(left) < static_cast<std::uintmax_t>(right);
    } else {
        return right >= Right{ 0 }
               && static_cast<std::uintmax_t>(left) < static_cast<std::uintmax_t>(right);
    }
}

/** Checks arbitrary integral types for numeric equality without signedness conversions. */
template <std::integral Left, std::integral Right>
[[nodiscard]] constexpr bool bounded_integer_equal(const Left left, const Right right) noexcept {
    return !bounded_integer_less(left, right) && !bounded_integer_less(right, left);
}

/** Checks whether an integral value is representable in another integral type. */
template <std::integral Destination, std::integral Source>
[[nodiscard]] constexpr bool bounded_integer_in_range(const Source value) noexcept {
    return !bounded_integer_less(value, std::numeric_limits<Destination>::lowest())
           && !bounded_integer_less(std::numeric_limits<Destination>::max(), value);
}

} // namespace detail

/**
 * @brief An immutable integer whose value is restricted to the closed interval [Minimum, Maximum].
 *
 * Constructor arguments are checked before they are converted to T, so signedness changes and narrowing cannot
 * accidentally turn an invalid argument into a valid value. A BoundedInteger can be copied or moved to construct
 * another object, but cannot be assigned a new value.
 *
 * @tparam T The integral type used to store the value; bool is not supported
 * @tparam Minimum The smallest valid value
 * @tparam Maximum The largest valid value
 */
template <std::integral T, T Minimum, T Maximum>
    requires(!std::same_as<std::remove_cv_t<T>, bool> && std::same_as<T, std::remove_cv_t<T>>)
class BoundedInteger {
public:
    using value_type = T;

    static constexpr value_type min_value = Minimum;
    static constexpr value_type max_value = Maximum;

    static_assert(min_value <= max_value, "BoundedInteger: Minimum must not be larger than Maximum");

    /**
     * @brief Constructs a bounded integer from an integral value.
     * @tparam U The argument's integral type
     * @param value The value, which must be in [Minimum, Maximum]
     * @exception Throws an Exception if value is outside the configured interval
     */
    template <std::integral U>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    constexpr explicit BoundedInteger(const U value)
        : value_{ static_cast<value_type>(value) } {
        Exception::check(!detail::bounded_integer_less(value, min_value) && !detail::bounded_integer_less(max_value, value),
                         "BoundedInteger::BoundedInteger: The value must be in [{}, {}], was {}",
                         printable(min_value), printable(max_value), printable(value));
    }

    constexpr BoundedInteger(const BoundedInteger&) noexcept = default;
    constexpr BoundedInteger(BoundedInteger&&) noexcept = default;
    BoundedInteger& operator=(const BoundedInteger&) = delete;
    BoundedInteger& operator=(BoundedInteger&&) = delete;

    /**
     * @brief Returns the stored value.
     * @return The value in the underlying type
     */
    [[nodiscard]] constexpr value_type get() const noexcept {
        return value_;
    }

    /**
     * @brief Converts the stored value to another integral type without changing its numeric value.
     * @tparam U The requested integral result type; bool is not supported
     * @exception Throws an Exception if the value is not representable in U
     * @return The value represented as U
     */
    template <std::integral U>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr U as() const {
        Exception::check(detail::bounded_integer_in_range<U>(value_),
                         "BoundedInteger::as: The value {} is not representable in the requested type", printable(value_));
        return static_cast<U>(value_);
    }

    /**
     * @brief Returns the smallest value allowed by this specialization.
     * @return A bounded integer containing Minimum
     */
    [[nodiscard]] static constexpr BoundedInteger min() noexcept {
        return BoundedInteger{ min_value };
    }

    /**
     * @brief Returns the largest value allowed by this specialization.
     * @return A bounded integer containing Maximum
     */
    [[nodiscard]] static constexpr BoundedInteger max() noexcept {
        return BoundedInteger{ max_value };
    }

    /**
     * @brief Checks whether this value is smaller than another bounded integer.
     * @tparam U The other value's underlying type
     * @tparam OtherMinimum The other value's minimum
     * @tparam OtherMaximum The other value's maximum
     * @param other The other bounded integer
     * @return True iff this value is strictly smaller
     */
    template <std::integral U, U OtherMinimum, U OtherMaximum>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr bool smaller_than(const BoundedInteger<U, OtherMinimum, OtherMaximum>& other) const noexcept {
        return detail::bounded_integer_less(value_, other.get());
    }

    /**
     * @brief Checks whether this value is smaller than a raw integral value.
     * @tparam U The raw value's integral type
     * @param other The other value, which must be in this specialization's interval
     * @exception Throws an Exception if other is outside [Minimum, Maximum]
     * @return True iff this value is strictly smaller
     */
    template <std::integral U>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr bool smaller_than(const U other) const {
        return smaller_than(BoundedInteger{ other });
    }

    /**
     * @brief Checks whether this value is larger than another bounded integer.
     * @tparam U The other value's underlying type
     * @tparam OtherMinimum The other value's minimum
     * @tparam OtherMaximum The other value's maximum
     * @param other The other bounded integer
     * @return True iff this value is strictly larger
     */
    template <std::integral U, U OtherMinimum, U OtherMaximum>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr bool larger_than(const BoundedInteger<U, OtherMinimum, OtherMaximum>& other) const noexcept {
        return detail::bounded_integer_less(other.get(), value_);
    }

    /**
     * @brief Checks whether this value is larger than a raw integral value.
     * @tparam U The raw value's integral type
     * @param other The other value, which must be in this specialization's interval
     * @exception Throws an Exception if other is outside [Minimum, Maximum]
     * @return True iff this value is strictly larger
     */
    template <std::integral U>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr bool larger_than(const U other) const {
        return larger_than(BoundedInteger{ other });
    }

    /**
     * @brief Returns the smaller of this and another value of the same bounded type.
     * @param other The other value
     * @return The smaller bounded integer
     */
    [[nodiscard]] constexpr BoundedInteger min(const BoundedInteger& other) const noexcept {
        return BoundedInteger{ std::min(value_, other.value_) };
    }

    /**
     * @brief Returns the smaller of this value and a raw integral value.
     * @tparam U The raw value's integral type
     * @param other The other value, which must be in this specialization's interval
     * @exception Throws an Exception if other is outside [Minimum, Maximum]
     * @return The smaller bounded integer
     */
    template <std::integral U>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr BoundedInteger min(const U other) const {
        return min(BoundedInteger{ other });
    }

    /**
     * @brief Returns the larger of this and another value of the same bounded type.
     * @param other The other value
     * @return The larger bounded integer
     */
    [[nodiscard]] constexpr BoundedInteger max(const BoundedInteger& other) const noexcept {
        return BoundedInteger{ std::max(value_, other.value_) };
    }

    /**
     * @brief Returns the larger of this value and a raw integral value.
     * @tparam U The raw value's integral type
     * @param other The other value, which must be in this specialization's interval
     * @exception Throws an Exception if other is outside [Minimum, Maximum]
     * @return The larger bounded integer
     */
    template <std::integral U>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr BoundedInteger max(const U other) const {
        return max(BoundedInteger{ other });
    }

    /**
     * @brief Clamps this value to the closed subinterval [lower, upper].
     * @param lower The inclusive lower bound
     * @param upper The inclusive upper bound
     * @exception Throws an Exception if lower is larger than upper
     * @return The clamped bounded integer
     */
    [[nodiscard]] constexpr BoundedInteger clamp(const BoundedInteger& lower, const BoundedInteger& upper) const {
        Exception::check(lower.value_ <= upper.value_,
                         "BoundedInteger::clamp: The lower bound {} must not be larger than the upper bound {}",
                         printable(lower.value_), printable(upper.value_));
        return BoundedInteger{ std::clamp(value_, lower.value_, upper.value_) };
    }

    /**
     * @brief Clamps this value to raw integral bounds.
     * @tparam LowerType The lower bound's integral type
     * @tparam UpperType The upper bound's integral type
     * @param lower The inclusive lower bound, which must be in this specialization's interval
     * @param upper The inclusive upper bound, which must be in this specialization's interval
     * @exception Throws an Exception if a bound is invalid or lower is larger than upper
     * @return The clamped bounded integer
     */
    template <std::integral LowerType, std::integral UpperType>
        requires(!std::same_as<std::remove_cv_t<LowerType>, bool>
                 && !std::same_as<std::remove_cv_t<UpperType>, bool>)
    [[nodiscard]] constexpr BoundedInteger clamp(const LowerType lower, const UpperType upper) const {
        return clamp(BoundedInteger{ lower }, BoundedInteger{ upper });
    }

    /**
     * @brief Checks whether two bounded integers have the same numeric value, regardless of type and bounds.
     * @tparam U The other value's underlying type
     * @tparam OtherMinimum The other value's minimum
     * @tparam OtherMaximum The other value's maximum
     * @param other The other bounded integer
     * @return True iff both numeric values are equal
     */
    template <std::integral U, U OtherMinimum, U OtherMaximum>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr bool operator==(const BoundedInteger<U, OtherMinimum, OtherMaximum>& other) const noexcept {
        return detail::bounded_integer_equal(value_, other.get());
    }

    /**
     * @brief Compares two bounded integers numerically, regardless of type and bounds.
     * @tparam U The other value's underlying type
     * @tparam OtherMinimum The other value's minimum
     * @tparam OtherMaximum The other value's maximum
     * @param other The other bounded integer
     * @return The strong ordering of the two numeric values
     */
    template <std::integral U, U OtherMinimum, U OtherMaximum>
        requires(!std::same_as<std::remove_cv_t<U>, bool>)
    [[nodiscard]] constexpr std::strong_ordering
    operator<=>(const BoundedInteger<U, OtherMinimum, OtherMaximum>& other) const noexcept {
        if (detail::bounded_integer_less(value_, other.get())) {
            return std::strong_ordering::less;
        }
        if (detail::bounded_integer_less(other.get(), value_)) {
            return std::strong_ordering::greater;
        }
        return std::strong_ordering::equal;
    }

    /**
     * @brief Prints the stored value.
     * @param output_stream The stream to which the value should be printed
     * @param value The bounded integer that should be printed
     * @return A reference to output_stream
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const BoundedInteger& value) {
        return output_stream << printable(value.value_);
    }

private:
    template <std::integral U>
    [[nodiscard]] static constexpr auto printable(const U value) noexcept {
        using printable_type = std::conditional_t<std::is_signed_v<U>, std::intmax_t, std::uintmax_t>;
        return static_cast<printable_type>(value);
    }

    const value_type value_;
};

} // namespace utility

/**
 * @brief Formats a BoundedInteger using the formatter of its underlying type.
 * @tparam T The bounded integer's underlying type
 * @tparam Minimum The smallest valid value
 * @tparam Maximum The largest valid value
 */
template <std::integral T, T Minimum, T Maximum>
    requires(!std::same_as<std::remove_cv_t<T>, bool> && std::same_as<T, std::remove_cv_t<T>>)
struct fmt::formatter<utility::BoundedInteger<T, Minimum, Maximum>>
    : fmt::formatter<std::conditional_t<std::is_signed_v<T>, std::intmax_t, std::uintmax_t>> {
    /**
     * @brief Formats the bounded integer into the output of the format context.
     * @param value The bounded integer that should be formatted
     * @param ctx The format context that provides the output iterator
     * @return The output iterator past the formatted value
     */
    auto format(const utility::BoundedInteger<T, Minimum, Maximum>& value, fmt::format_context& ctx) const {
        using printable_type = std::conditional_t<std::is_signed_v<T>, std::intmax_t, std::uintmax_t>;
        return fmt::formatter<printable_type>::format(static_cast<printable_type>(value.get()), ctx);
    }
};

/**
 * @brief Hashes a BoundedInteger with the standard hash of its underlying value
 * @tparam T The bounded integer's underlying type
 * @tparam Minimum The smallest valid value
 * @tparam Maximum The largest valid value
 */
template <std::integral T, T Minimum, T Maximum>
    requires(!std::same_as<std::remove_cv_t<T>, bool> && std::same_as<T, std::remove_cv_t<T>>)
struct std::hash<utility::BoundedInteger<T, Minimum, Maximum>> {
    [[nodiscard]] std::size_t operator()(const utility::BoundedInteger<T, Minimum, Maximum>& value) const
        noexcept(noexcept(std::hash<T>{}(value.get()))) {
        return std::hash<T>{}(value.get());
    }
};
