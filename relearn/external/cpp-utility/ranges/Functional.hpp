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

#include "cpp-utility/Concepts.hpp"

#include <range/v3/functional/arithmetic.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/range/traits.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace utility {

/**
 * @brief Returns tuple-like element I through ADL-enabled std::get or a member get<I>(), preserving value category.
 */
template <std::size_t I>
inline constexpr auto element = []<typename T>
    requires detail::has_adl_get<T> || detail::has_member_get<T>
(T&& tuple)
    -> decltype(auto) {
    if constexpr (detail::has_adl_get<T>) {
        using std::get;
        return get<I>(std::forward<T>(tuple));
    } else if constexpr (detail::has_member_get<T>) {
        return std::forward<T>(tuple).template get<I>();
    }
};

/** @brief Predicate that accepts non-null raw pointers. */
inline constexpr auto not_nullptr = [](const auto* const ptr) noexcept { return ptr != nullptr; };

/** @brief Curries the right operand of addition; the returned callable evaluates lhs + rhs transparently. */
inline constexpr auto plus = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr { return std::plus<>{}(lhs, rhs); };
};

/** @brief Curries the right operand of subtraction; the returned callable evaluates lhs - rhs transparently. */
inline constexpr auto minus = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr { return std::minus<>{}(lhs, rhs); };
};

/** @brief Curries the right operand of multiplication; the returned callable evaluates lhs * rhs transparently. */
inline constexpr auto multiplies = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr {
        return std::multiplies<>{}(lhs, rhs);
    };
};

/** @brief Curries the right operand of division; division-by-zero behavior is that of the operand types. */
inline constexpr auto divides = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::divides<>{}(lhs, rhs); };
};

/** @brief Curries the right operand of remainder; invalid/zero divisors retain the operand types' behavior. */
inline constexpr auto modulus = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::modulus<>{}(lhs, rhs); };
};

/** @brief Applies unary arithmetic negation. */
inline constexpr auto negate = []<std::regular T>(const T& val) constexpr {
    return std::negate<>{}(val);
};

/** @brief Curries the right operand of logical AND and returns a bool. */
inline constexpr auto logical_and = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr {
        return std::logical_and<>{}(lhs, rhs);
    };
};

/** @brief Curries the right operand of logical OR and returns a bool. */
inline constexpr auto logical_or = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr {
        return std::logical_or<>{}(lhs, rhs);
    };
};

/** @brief Applies logical negation and returns a bool. */
inline constexpr auto logical_not = []<std::regular T>(const T& val) constexpr {
    return std::logical_not<>{}(val);
};

/** @brief Curries the right operand of bitwise AND. */
inline constexpr auto bit_and = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::bit_and<>{}(lhs, rhs); };
};

/** @brief Curries the right operand of bitwise OR. */
inline constexpr auto bit_or = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::bit_or<>{}(lhs, rhs); };
};

/** @brief Curries the right operand of bitwise XOR. */
inline constexpr auto bit_xor = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::bit_xor<>{}(lhs, rhs); };
};

/** @brief Applies bitwise complement. */
inline constexpr auto bit_not = []<std::regular T>(const T& val) constexpr {
    return std::bit_not<>{}(val);
};

/** @brief Curries the right operand of equality comparison. */
inline constexpr auto equal_to = []<std::equality_comparable T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>
        requires std::equality_comparable_with<U, T>
    (const U& lhs) constexpr {
        return ranges::equal_to{}(lhs, rhs);
    };
};

/** @brief Curries the right operand of inequality comparison. */
inline constexpr auto not_equal_to = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>
        requires std::equality_comparable_with<U, T>
    (const U& lhs) constexpr {
        return ranges::not_equal_to{}(lhs, rhs);
    };
};

/** @brief Curries rhs and tests lhs > rhs. */
inline constexpr auto greater = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>
        requires std::totally_ordered_with<U, T>
    (const U& lhs) constexpr {
        return ranges::greater{}(lhs, rhs);
    };
};

/** @brief Curries rhs and tests lhs < rhs. */
inline constexpr auto less = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>
        requires std::totally_ordered_with<U, T>
    (const U& lhs) constexpr {
        return ranges::less{}(lhs, rhs);
    };
};

/** @brief Curries rhs and tests lhs >= rhs. */
inline constexpr auto greater_equal = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>
        requires std::totally_ordered_with<U, T>
    (const U& lhs) constexpr {
        return ranges::greater_equal{}(lhs, rhs);
    };
};

/** @brief Curries rhs and tests lhs <= rhs. */
inline constexpr auto less_equal = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>
        requires std::totally_ordered_with<U, T>
    (const U& lhs) constexpr {
        return ranges::less_equal{}(lhs, rhs);
    };
};

/** @brief Returns std::abs(val), retaining std::abs overload and domain behavior. */
inline constexpr auto as_abs = [](const auto& val) constexpr {
    return std::abs(val);
};

/** @brief Brace-constructs T from one value; narrowing conversions are rejected during constraint checking. */
#if defined(__GNUC__)
#pragma GCC diagnostic push
// The caller names the target type, so a widening conversion such as construct<double> from a float is
// asked for explicitly and not the accidental promotion this warning looks for. Conversions that lose
// data stay rejected by the braces in the requires-clause below, which is the actual guard here.
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif
template <typename T>
inline constexpr auto construct = []<typename ValueType>(ValueType&& val)
    requires requires { T{ std::forward<ValueType>(val) }; }
{
    return T{ std::forward<ValueType>(val) };
};
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/**
 * @brief Creates a callable that ignores all of its arguments and always returns a copy of the given value.
 *
 * Handy for satisfying a callback interface with a fixed result, e.g. a function that must accept a
 * neuron index but should yield the same value for every one. The zero case is simply constant(T{}).
 */
inline constexpr auto constant = []<std::copy_constructible T>(T value) constexpr {
    return [value = std::move(value)](auto&&...) constexpr -> T { return value; };
};

/** @brief Applies Comparator to the two elements of a pair-like value. */
template <typename Comparator = std::equal_to<>>
inline constexpr auto pairwise_comparison =
    [comp = Comparator{}]<typename PairLikeType>
    requires detail::has_tuple_size<PairLikeType> && (std::tuple_size_v<PairLikeType> == 2) && std::relation<Comparator, std::tuple_element_t<0, PairLikeType>, std::tuple_element_t<1, PairLikeType>>
(const PairLikeType& pair) {
    return comp(element<0>(pair), element<1>(pair));
};

namespace detail {

// Subscripting a std-style container converts the index to the unsigned size type;
// do the conversion explicitly for signed indices so that -Wsign-conversion stays clean for callers
inline constexpr auto as_lookup_index = []<typename IndexType>(IndexType&& index) constexpr -> decltype(auto) {
    if constexpr (std::signed_integral<std::remove_cvref_t<IndexType>>) {
        return static_cast<std::make_unsigned_t<std::remove_cvref_t<IndexType>>>(index);
    } else {
        return std::forward<IndexType>(index);
    }
};

} // namespace detail

/**
 * @brief Creates an unchecked subscript lookup callable, optionally projecting each input to an index.
 *
 * Lvalue ranges are referenced and must outlive the callable. Rvalue ranges are owned by the callable, which avoids
 * dangling references even for temporary borrowed ranges. Projected indices must be valid; negative signed indices
 * are converted to the range's unsigned lookup convention and therefore remain out of bounds.
 */
inline constexpr auto lookup =
    []<ranges::random_access_range T, typename Projection = std::identity>(
        T&& lookup_range_ref, Projection proj = {})
    requires std::copy_constructible<Projection> && (std::is_lvalue_reference_v<T> || std::movable<std::remove_cvref_t<T>>)
{
    if constexpr (!std::is_lvalue_reference_v<T>) {
        return [lookup_range = std::forward<T>(lookup_range_ref), proj = std::move(proj)]<typename IndexType>
            requires std::regular_invocable<Projection&, const IndexType&>
        (
            const IndexType& index) constexpr mutable -> decltype(auto) {
            return lookup_range[detail::as_lookup_index(std::invoke(proj, index))];
        };
    } else if constexpr (std::is_const_v<std::remove_reference_t<T>>) {
        return [&lookup_range_ref, proj]<typename IndexType>
            requires std::regular_invocable<const Projection&, const IndexType&>
        (
            const IndexType& index) constexpr -> decltype(auto) {
            return lookup_range_ref[detail::as_lookup_index(std::invoke(proj, index))];
        };
    } else {
        return [&lookup_range_ref, proj]<typename IndexType>
            requires std::regular_invocable<Projection&, const IndexType&>
        (
            const IndexType& index) constexpr mutable -> decltype(auto) {
            return lookup_range_ref[detail::as_lookup_index(std::invoke(proj, index))];
        };
    }
};

} // namespace utility
