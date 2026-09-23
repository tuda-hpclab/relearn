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

#include "mpi-wrapper/core/MPIRankTraits.h"

#include <cpp-utility/Exception.hpp>

#include <fmt/format.h>

#include <compare>
#include <cstddef>
#include <functional>
#include <ostream>

namespace mpiPP {

/**
 * This type reflects an MPI rank in a type-safe manner.
 *
 * It is a thin wrapper around MPIRankID, i.e., around a utility::TaggedID configured with MPIRankTraits: the bit
 * layout, the admissible values, the ordering, the hash and the printed text all come from there, this class adds
 * the names of the MPI domain and the check that guards the access to an uninitialized rank.
 *
 * The factories for ranges of ranks are static methods of MPIRankRange in mpi-wrapper/core/MPIRankRange.h,
 * so that this header stays free of range-v3.
 */
class MPIRank {
public:
    /** @brief The tagged id that carries the rank, see MPIRankTraits */
    using id_type = MPIRankID;

    /**
     * @brief Returns the uninitialized rank that can be used for debugging purposes.
     * @return The uninitialized rank
     */
    [[nodiscard]] static constexpr MPIRank uninitialized_rank() noexcept { return MPIRank{}; }

    /**
     * @brief Returns the root rank (0) in an initialized state.
     * @return The root rank
     */
    [[nodiscard]] static constexpr MPIRank root_rank() { return MPIRank{ 0 }; }

    /**
     * @brief Default constructs an uninitialized rank
     */
    constexpr MPIRank() noexcept = default;

    /**
     * @brief Constructs an initialized rank with rank
     * @exception Throws an Exception if rank < 0 or if it is larger than (1<<30)-2
     */
    constexpr explicit MPIRank(const int rank)
        : id_{ rank } { }

    /**
     * @brief Constructs a rank from an already built tagged id, which is initialized iff the id's flag is set
     * @param id The tagged id that carries the rank
     */
    constexpr explicit MPIRank(const MPIRankID id) noexcept
        : id_{ id } { }

    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const MPIRank& first, const MPIRank& second) = default;

    /**
     * @brief Check if the rank is initialized
     * @return true iff the rank is initialized
     */
    [[nodiscard]] constexpr bool is_initialized() const noexcept { return id_.get_flag<MPIRankTraits::initialized_flag>(); }

    /**
     * @brief Returns the stored MPI rank as int
     * @exception Throws an Exception if the object is not initialized
     * @return The stored MPI rank
     */
    [[nodiscard]] constexpr int get_rank() const {
        utility::Exception::check(is_initialized(), "MPIRank::get_rank: This MPIRank is not initialized.");
        return id_.get_value_as<int>();
    }

    /**
     * @brief Returns the stored MPI rank as std::size_t
     * @exception Throws an Exception if the object is not initialized
     * @return The stored MPI rank
     */
    [[nodiscard]] constexpr std::size_t get_rank_cast() const {
        utility::Exception::check(is_initialized(), "MPIRank::get_rank_cast: This MPIRank is not initialized.");
        return id_.get_value_as<std::size_t>();
    }

    /**
     * @brief Returns the tagged id that carries the rank, e.g., to reuse the algorithms of utility
     * @return The tagged id
     */
    [[nodiscard]] constexpr MPIRankID get_id() const noexcept { return id_; }

    /**
     * @brief Prints the object's represented MPI rank
     * @param os The out-stream in which the object is printed
     * @return The argument os to allow chaining
     */
    friend std::ostream& operator<<(std::ostream& os, const MPIRank& rank) {
        return os << rank.id_;
    }

private:
    MPIRankID id_{};
};

static_assert(sizeof(MPIRank) == sizeof(MPIRankID), "MPIRank grew beyond the tagged id it wraps");

} // namespace mpiPP

/**
 * @brief Formats an MPIRank exactly like the tagged id it wraps, i.e., as "MPIRank: 3"
 *      respectively "MPIRank: uninitialized"
 */
template <>
struct fmt::formatter<mpiPP::MPIRank> : fmt::formatter<mpiPP::MPIRankID> {
    auto format(const mpiPP::MPIRank& rank, fmt::format_context& ctx) const {
        return fmt::formatter<mpiPP::MPIRankID>::format(rank.get_id(), ctx);
    }
};

namespace std {
template <>
struct hash<mpiPP::MPIRank> {
    using argument_type = mpiPP::MPIRank;
    using result_type = std::size_t;

    result_type operator()(const argument_type& mpi_rank) const noexcept {
        return mpi_rank.get_id().hash_value();
    }
};
} // namespace std
