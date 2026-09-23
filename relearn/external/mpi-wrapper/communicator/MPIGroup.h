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

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <cstddef>
#include <span>
#include <utility>

namespace mpiPP {

/**
 * @brief A type-safe, move-only RAII wrapper around an MPI process group (MPI_Group).
 *
 * A group is an ordered set of process identifiers. Groups are obtained from a
 * communicator (MPICommunicator::get_group) and reshaped with include(...)/exclude(...)
 * to build the subset that a new communicator should span (MPICommunicator::create).
 * Copying is forbidden. Move construction leaves the source null; move assignment swaps the two
 * resource states.
 */
class MPIGroup {
public:
    /**
     * @brief Default-constructs a null group (MPI_GROUP_NULL).
     */
    constexpr MPIGroup() noexcept = default;

    MPIGroup(const MPIGroup& other) = delete;
    MPIGroup& operator=(const MPIGroup& other) = delete;

    /**
     * @brief Move-constructs from another MPIGroup by taking ownership of its handle
     * @param other The other MPIGroup, left in a null state
     */
    MPIGroup(MPIGroup&& other) noexcept {
        std::swap(group_, other.group_);
    }

    /**
     * @brief Move-assigns from another MPIGroup by taking ownership of its handle
     * @param other The other MPIGroup, left holding this one's previous handle
     */
    MPIGroup& operator=(MPIGroup&& other) noexcept {
        std::swap(group_, other.group_);

        return *this;
    }

    /**
     * @brief Frees a non-null, non-empty group; MPI cleanup errors are suppressed
     */
    ~MPIGroup() noexcept {
        if (group_ != MPI_GROUP_NULL && group_ != MPI_GROUP_EMPTY) {
            static_cast<void>(MPI_Group_free(&group_));
        }
    }

    /**
     * @brief Produces the subgroup consisting of the specified ranks, in the specified order
     * @param ranks The ranks (of this group) to include; must be distinct and valid
     * @exception Throws an Exception if MPI returns an error code
     * @return The included subgroup, owning its handle
     */
    [[nodiscard]] MPIGroup include(const std::span<const int> ranks) const {
        auto new_group = MPI_Group{ MPI_GROUP_NULL };
        const auto count = utility::safe_cast<int>(ranks.size());
        const auto error_code = MPI_Group_incl(group_, count, ranks.data(), &new_group);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPIGroup::include: Including ranks returned the error: {}", error_code);
        return MPIGroup{ new_group };
    }

    /**
     * @brief Produces the subgroup consisting of all ranks except the specified ones
     * @param ranks The ranks (of this group) to exclude; must be distinct and valid
     * @exception Throws an Exception if MPI returns an error code
     * @return The remaining subgroup, owning its handle
     */
    [[nodiscard]] MPIGroup exclude(const std::span<const int> ranks) const {
        auto new_group = MPI_Group{ MPI_GROUP_NULL };
        const auto count = utility::safe_cast<int>(ranks.size());
        const auto error_code = MPI_Group_excl(group_, count, ranks.data(), &new_group);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPIGroup::exclude: Excluding ranks returned the error: {}", error_code);
        return MPIGroup{ new_group };
    }

    /**
     * @brief Returns the number of ranks in this group
     * @exception Throws an Exception if MPI returns an error code
     * @return The number of ranks
     */
    [[nodiscard]] int get_size() const {
        auto size = int{ 0 };
        const auto error_code = MPI_Group_size(group_, &size);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPIGroup::get_size: returned the error: {}", error_code);
        return size;
    }

    /**
     * @brief Returns this rank's id within this group, or an uninitialized rank if not a member
     * @exception Throws an Exception if MPI returns an error code
     * @return This rank's id, uninitialized iff the calling rank is not part of the group
     */
    [[nodiscard]] MPIRank get_my_rank() const {
        auto rank = int{ 0 };
        const auto error_code = MPI_Group_rank(group_, &rank);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPIGroup::get_my_rank: returned the error: {}", error_code);
        if (rank == MPI_UNDEFINED) {
            return MPIRank::uninitialized_rank();
        }
        return MPIRank{ rank };
    }

    /**
     * @brief Checks if this group is the null group (MPI_GROUP_NULL)
     * @return True iff this group is null
     */
    [[nodiscard]] bool is_null() const noexcept {
        return group_ == MPI_GROUP_NULL;
    }

    /**
     * @brief Returns the underlying MPI_Group handle to hand to raw MPI calls
     * @return The underlying handle
     */
    [[nodiscard]] MPI_Group get() const noexcept {
        return group_;
    }

private:
    friend class MPICommunicator;

    /**
     * @brief Constructs a group wrapping (and taking ownership of) the given handle
     * @param group The handle to wrap
     */
    constexpr explicit MPIGroup(const MPI_Group group) noexcept
        : group_(group) { }

    MPI_Group group_{ MPI_GROUP_NULL };
};

} // namespace mpiPP
