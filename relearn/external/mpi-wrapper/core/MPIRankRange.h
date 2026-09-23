#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/core/MPIRank.h"
#include "mpi-wrapper/core/MPIRankTraits.h"

#include <cpp-utility/data-structure/TaggedIDRange.hpp>
#include <cpp-utility/ranges/Functional.hpp>

#include <range/v3/view/transform.hpp>

namespace mpiPP {

/**
 * @brief The range factories for MPIRank, provided as static methods.
 *
 * They live in this separate header so that MPIRank.h itself stays free of range-v3: only translation units that
 * actually iterate over ranks pay for those includes. This mirrors utility::TaggedIDRange, to which the bounds
 * checking and the construction of the ranks are delegated.
 *
 * Example: MPIRankRange::range(3)          the ranks 0, 1, 2
 *      MPIRankRange::range(2, 5)           the ranks 2, 3, 4
 *      MPIRankRange::range_values(3)       the plain rank ids 0, 1, 2, e.g., for an MPI call that wants ints
 */
class MPIRankRange {
public:
    using id_type = MPIRank;

    /** @brief The plain values are ints, which is the type in which MPI itself expects a rank */
    using value_type = int;

    static constexpr value_type min_value = 0;
    static constexpr value_type max_value = static_cast<value_type>(MPIRankID::max_value);

    /**
     * @brief Creates a range of the MPIRanks [begin, end), all of them initialized
     *
     * As end is past-the-end, it may be max_value + 1, which is what makes the largest rank reachable
     *
     * @param begin The first rank of the range
     * @param end The past-the-end rank of the range
     * @exception Throws an Exception if begin < 0, if end < 0, if begin > end, or if end > max_value + 1
     * @return The range of MPIRanks
     */
    [[nodiscard]] static auto range(const value_type begin, const value_type end) {
        return utility::TaggedIDRange<MPIRankID>::range(begin, end) | ranges::views::transform(utility::construct<MPIRank>);
    }

    /**
     * @brief Creates a range of the MPIRanks [0, count), all of them initialized
     * @param count The number of ranks, a count of 0 yields an empty range
     * @exception Throws an Exception if count < 0 or if count > max_value + 1
     * @return The range of MPIRanks
     */
    [[nodiscard]] static auto range(const value_type count) {
        return range(min_value, count);
    }

    /**
     * @brief Creates a range of the plain rank ids [begin, end), i.e., of ints instead of MPIRanks
     * @param begin The first rank id of the range
     * @param end The past-the-end rank id of the range
     * @exception Throws an Exception if begin < 0, if end < 0, if begin > end, or if end > max_value + 1
     * @return The range of rank ids
     */
    [[nodiscard]] static auto range_values(const value_type begin, const value_type end) {
        return utility::TaggedIDRange<MPIRankID>::range_values(begin, end)
               | ranges::views::transform([](const MPIRankID::value_type value) { return static_cast<value_type>(value); });
    }

    /**
     * @brief Creates a range of the plain rank ids [0, count), i.e., of ints instead of MPIRanks
     * @param count The number of rank ids, a count of 0 yields an empty range
     * @exception Throws an Exception if count < 0 or if count > max_value + 1
     * @return The range of rank ids
     */
    [[nodiscard]] static auto range_values(const value_type count) {
        return range_values(min_value, count);
    }
};

} // namespace mpiPP
