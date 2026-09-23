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

#include <cpp-utility/data-structure/TaggedID.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace mpiPP {

/**
 * @brief The TaggedID customization that reproduces the semantics of MPIRank.
 *
 * An id built from these traits behaves exactly like the hand-written MPIRank: it is uninitialized when default
 * constructed, initialized when constructed from a value, it prints as "MPIRank: 3" respectively
 * "MPIRank: uninitialized", and it rejects the same values as MPIRank's constructor does.
 *
 * TaggedIDTraits::unset_text is deliberately not re-declared, its default is already the expected "uninitialized".
 */
struct MPIRankTraits : utility::TaggedIDTraits {
    /** @brief The index of the flag that marks a rank as carrying an actual rank id, i.e., as being initialized */
    static constexpr std::size_t initialized_flag = 0;

    /** @brief Ranks print as "<name>: <value>" instead of the generic [value: ..., flags: ...] */
    static constexpr std::string_view name = "MPIRank";

    /** @brief Constructing a rank from a value marks it as initialized, default constructing one does not */
    static constexpr std::size_t constructed_flags = std::size_t{ 1 } << initialized_flag;

    /** @brief An uninitialized rank prints as "MPIRank: uninitialized" instead of printing its (meaningless) value */
    static constexpr std::size_t unset_flag = initialized_flag;

    /**
     * @brief The largest admissible rank. MPIRank rejects everything that is not smaller than
     *      std::numeric_limits<int>::max() / 2, and as TaggedID rejects everything that is larger than max_value,
     *      the limit is that bound minus one
     */
    static constexpr std::uint64_t max_value_limit
        = static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 2 - 1);
};

/**
 * @brief The TaggedID that carries MPIRank's semantics, see MPIRankTraits.
 *
 * 31 value bits and one flag, which makes it the same size as the hand-written MPIRank. The value type is unsigned
 * because TaggedID requires it to be; the int that MPI expects is obtained via get_value_as<int>().
 */
using MPIRankID = utility::TaggedID<std::uint32_t, 1, MPIRankTraits>;

static_assert(MPIRankID::max_value == static_cast<std::uint32_t>(std::numeric_limits<int>::max() / 2 - 1),
              "MPIRankTraits: The largest admissible rank does not match the one of MPIRank");
static_assert(MPIRankID::min_value == 0, "MPIRankTraits: The smallest admissible rank does not match the one of MPIRank");
static_assert(MPIRankID::has_name, "MPIRankTraits: The rank does not print with its name");
static_assert(MPIRankID::has_unset_flag, "MPIRankTraits: The rank does not distinguish an uninitialized value");

} // namespace mpiPP
