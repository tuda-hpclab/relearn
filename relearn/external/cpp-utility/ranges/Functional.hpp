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

#include "cpp-utility/Concepts.hpp"

#include <range/v3/functional/arithmetic.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/range/traits.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace utility {

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

constexpr auto not_nullptr = [](const auto* const ptr) { return ptr != nullptr; };

inline constexpr auto plus = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr { return std::plus<T>{}(lhs, rhs); };
};

inline constexpr auto minus = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr { return std::minus<T>{}(lhs, rhs); };
};

inline constexpr auto multiplies = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr {
        return std::multiplies<T>{}(lhs, rhs);
    };
};

inline constexpr auto divides = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::divides<T>{}(lhs, rhs); };
};

inline constexpr auto modulus = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::modulus<T>{}(lhs, rhs); };
};

inline constexpr auto negate = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::negate<T>{}(lhs, rhs); };
};

inline constexpr auto logical_and = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr {
        return std::logical_and<T>{}(lhs, rhs);
    };
};

inline constexpr auto logical_or = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr {
        return std::logical_or<T>{}(lhs, rhs);
    };
};

inline constexpr auto logical_not = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>(const U& lhs) constexpr {
        return std::logical_not<T>{}(lhs, rhs);
    };
};

inline constexpr auto bit_and = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::bit_and<T>{}(lhs, rhs); };
};

inline constexpr auto bit_or = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::bit_or<T>{}(lhs, rhs); };
};

inline constexpr auto bit_xor = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::bit_xor<T>{}(lhs, rhs); };
};

inline constexpr auto bit_not = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>(const U& lhs) constexpr { return std::bit_not<T>{}(lhs, rhs); };
};

inline constexpr auto equal_to = []<std::equality_comparable T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>
        requires std::equality_comparable_with<U, T>
    (const U& lhs) constexpr {
        return ranges::equal_to{}(lhs, rhs);
    };
};

inline constexpr auto not_equal_to = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>
        requires std::equality_comparable_with<U, T>
    (const U& lhs) constexpr {
        return ranges::not_equal_to{}(lhs, rhs);
    };
};

inline constexpr auto greater = []<std::regular T>(T rhs) constexpr {
    return
        [rhs]<std::regular U>
        requires std::totally_ordered_with<U, T>
    (const U& lhs) constexpr {
        return ranges::greater{}(lhs, rhs);
    };
};

inline constexpr auto less = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>
        requires std::totally_ordered_with<U, T>
    (const U& lhs) constexpr {
        return ranges::less{}(lhs, rhs);
    };
};

inline constexpr auto greater_equal = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>
        requires std::totally_ordered_with<U, T>
    (const U& lhs) constexpr {
        return ranges::greater_equal{}(lhs, rhs);
    };
};

inline constexpr auto less_equal = []<std::regular T>(T rhs) constexpr {
    return [rhs]<std::regular U>
        requires std::totally_ordered_with<U, T>
    (const U& lhs) constexpr {
        return ranges::less_equal{}(lhs, rhs);
    };
};

inline constexpr auto as_abs = [](const auto& val) constexpr {
    return std::abs(val);
};

template <typename T>
inline constexpr auto construct = []<typename ValueType>(ValueType&& val)
    requires std::constructible_from<T, ValueType>
{
    return T{ std::forward<ValueType>(val) };
};

template <typename Comparator = std::equal_to<>>
inline constexpr auto pairwise_comparison =
    [comp = Comparator{}]<typename PairLikeType>
    requires(std::tuple_size_v<PairLikeType> == 2) && std::relation<Comparator, std::tuple_element_t<0, PairLikeType>, std::tuple_element_t<1, PairLikeType>>
(const PairLikeType& pair) {
    return comp(element<0>(pair), element<1>(pair));
};

inline constexpr auto lookup =
    []<ranges::random_access_range T, typename Projection = std::identity>(
        T && lookup_range_ref, Projection proj = {})
    requires std::is_trivially_copy_constructible_v<Projection> && (std::is_lvalue_reference_v<T> || ranges::borrowed_range<T>)
{
    if constexpr (std::is_const_v<T>) {
        return [&lookup_range_ref, proj]<typename IndexType>
            requires std::regular_invocable<Projection, IndexType>
        (
            const IndexType& index) constexpr -> decltype(auto) {
            return lookup_range_ref[std::invoke(proj, index)];
        };
    } else {
        return [&lookup_range_ref, proj]<typename IndexType>
            requires std::regular_invocable<Projection, IndexType>
        (
            const IndexType& index) constexpr mutable -> decltype(auto) {
            return lookup_range_ref[std::invoke(proj, index)];
        };
    }
};

} // namespace utility
