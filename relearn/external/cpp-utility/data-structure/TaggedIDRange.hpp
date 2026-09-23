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
#include "cpp-utility/data-structure/TaggedID.hpp"
#include "cpp-utility/ranges/Functional.hpp"

#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include <concepts>
#include <utility>

namespace utility {

/**
 * @brief The range factories for a TaggedID, provided as static methods.
 *
 * They live in this separate class so that TaggedID.hpp itself stays free of range-v3:
 * only translation units that actually iterate over ids pay for those includes.
 *
 * The ids are constructed via the value constructor and therefore carry exactly the flags that the id's traits
 * declare as TaggedIDTraits::constructed_flags, just like every other id that was constructed from a value.
 *
 * Example: using NeuronID = TaggedID<std::uint64_t, 2>;
 *      using NeuronIDRange = TaggedIDRange<NeuronID>;
 *      NeuronIDRange::range(3)                 the ids with the values 0, 1, 2
 *      NeuronIDRange::range(2, 5)              the ids with the values 2, 3, 4
 *      NeuronIDRange::range_values(3)          the plain values 0, 1, 2
 *
 * @tparam IDType The TaggedID specialization whose ids (or values) are generated
 */
template <TaggedIDType IDType>
class TaggedIDRange {
public:
    using id_type = IDType;
    using value_type = typename IDType::value_type;

    static constexpr value_type min_value = IDType::min_value;
    static constexpr value_type max_value = IDType::max_value;

    /**
     * @brief Creates a range of TaggedIDs with the values [begin, end) and the id's constructed flags
     *
     * The bounds are accepted in any integral type and are checked before they are converted to value_type,
     * so that the call sites need no cast and a negative bound is reported instead of wrapping around
     *
     * @param begin The first value of the range
     * @param end The past-the-end value of the range
     * @exception Throws an Exception if begin < 0, if end < 0, if begin > end, or if end > max_value + 1
     * @return The range of TaggedIDs
     */
    [[nodiscard]] static auto range(const std::integral auto begin, const std::integral auto end) {
        check_bounds("TaggedIDRange::range", begin, end);

        return ranges::views::iota(static_cast<value_type>(begin), static_cast<value_type>(end))
            | ranges::views::transform(construct<IDType>);
    }

    /**
     * @brief Creates a range of TaggedIDs with the values [0, count) and the id's constructed flags
     * @param count The number of ids
     * @exception Throws an Exception if count < 0 or if count > max_value + 1
     * @return The range of TaggedIDs
     */
    [[nodiscard]] static auto range(const std::integral auto count) {
        return range(min_value, count);
    }

    /**
     * @brief Creates a range of the plain values [begin, end), i.e., of type value_type instead of TaggedID
     *
     * The bounds are accepted in any integral type, see range()
     *
     * @param begin The first value of the range
     * @param end The past-the-end value of the range
     * @exception Throws an Exception if begin < 0, if end < 0, if begin > end, or if end > max_value + 1
     * @return The range of values
     */
    [[nodiscard]] static auto range_values(const std::integral auto begin, const std::integral auto end) {
        check_bounds("TaggedIDRange::range_values", begin, end);

        return ranges::views::iota(static_cast<value_type>(begin), static_cast<value_type>(end));
    }

    /**
     * @brief Creates a range of the plain values [0, count), i.e., of type value_type instead of TaggedID
     * @param count The number of values
     * @exception Throws an Exception if count < 0 or if count > max_value + 1
     * @return The range of values
     */
    [[nodiscard]] static auto range_values(const std::integral auto count) {
        return range_values(min_value, count);
    }

private:
    // The largest admissible past-the-end value. max_value + 1 never overflows value_type
    // because TaggedID reserves at least one bit for the flags
    static constexpr value_type past_max_value = static_cast<value_type>(max_value + value_type{ 1 });

    /**
     * @brief Checks that [begin, end) is a valid range of values. Compares without converting the arguments
     *      so that neither a signed bound wraps around nor the comparison itself needs a cast
     * @param function_name The name of the calling function, used as the prefix of the message
     * @param begin The first value of the range
     * @param end The past-the-end value of the range
     * @exception Throws an Exception if begin < 0, if end < 0, if begin > end, or if end > max_value + 1
     */
    static void check_bounds(const char* const function_name, const std::integral auto begin, const std::integral auto end) {
        Exception::check(std::cmp_greater_equal(begin, min_value), "{}: begin ({}) must not be negative", function_name, begin);
        Exception::check(std::cmp_greater_equal(end, min_value), "{}: end ({}) must not be negative", function_name, end);
        Exception::check(std::cmp_less_equal(begin, end), "{}: begin ({}) must not be larger than end ({})", function_name, begin, end);
        Exception::check(std::cmp_less_equal(end, past_max_value), "{}: end ({}) must not be larger than {}", function_name, end, past_max_value);
    }
};

} // namespace utility
