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

#include "mpi-wrapper/communicator/MPIGroup.h"
#include "mpi-wrapper/core/MPIRank.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <cstddef>
#include <utility>

namespace mpiPP {

/**
 * @brief A type-safe, move-only RAII wrapper around an MPI communicator (MPI_Comm).
 *
 * An MPICommunicator either owns its underlying handle (created via split(...) or
 * duplicate(...) and freed on destruction) or merely references a predefined
 * communicator (World, Self) that must not be freed. Copying is forbidden. Move construction
 * leaves the source null; move assignment swaps the two resource states.
 *
 * The predefined MPICommunicator::World is provided statically so it can serve as a
 * default argument for the wrapping functions.
 */
class MPICommunicator {
public:
    /**
     * @brief A non-owning wrapper around MPI_COMM_WORLD, retrievable statically.
     *      Meant as the default communicator for the wrapping functions.
     */
    static const MPICommunicator World;

    /**
     * @brief A non-owning wrapper around MPI_COMM_SELF (the single-rank self communicator).
     */
    static const MPICommunicator Self;

    /**
     * @brief Default-constructs a null, non-owning communicator (MPI_COMM_NULL).
     */
    constexpr MPICommunicator() noexcept = default;

    MPICommunicator(const MPICommunicator& other) = delete;
    MPICommunicator& operator=(const MPICommunicator& other) = delete;

    /**
     * @brief Move-constructs from another MPICommunicator by taking ownership of its handle
     * @param other The other MPICommunicator, left in a null, non-owning state
     */
    MPICommunicator(MPICommunicator&& other) noexcept {
        std::swap(communicator_, other.communicator_);
        std::swap(is_owning_, other.is_owning_);
    }

    /**
     * @brief Move-assigns from another MPICommunicator by taking ownership of its handle
     * @param other The other MPICommunicator, left holding this one's previous handle
     */
    MPICommunicator& operator=(MPICommunicator&& other) noexcept {
        std::swap(communicator_, other.communicator_);
        std::swap(is_owning_, other.is_owning_);

        return *this;
    }

    /**
     * @brief Frees an owned, non-null communicator; MPI cleanup errors are suppressed
     */
    ~MPICommunicator() noexcept {
        if (is_owning_ && communicator_ != MPI_COMM_NULL) {
            static_cast<void>(MPI_Comm_free(&communicator_));
        }
    }

    /**
     * @brief Splits this communicator into disjoint subcommunicators, one per color.
     *      This is a collective operation over all ranks of this communicator.
     * @param color The subcommunicator this rank joins; MPI_UNDEFINED yields a null communicator
     * @param key Controls the rank ordering within the new subcommunicator (ties broken by old rank)
     * @exception Throws an Exception if MPI returns an error code
     * @return The new subcommunicator, owning its handle (or null if color is MPI_UNDEFINED)
     */
    [[nodiscard]] MPICommunicator split(const int color, const int key = 0) const {
        auto new_communicator = MPI_Comm{ MPI_COMM_NULL };
        const auto error_code = MPI_Comm_split(communicator_, color, key, &new_communicator);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunicator::split: Splitting the communicator returned the error: {}", error_code);
        return MPICommunicator{ new_communicator, new_communicator != MPI_COMM_NULL };
    }

    /**
     * @brief Duplicates this communicator, yielding an independent communicator with the same group.
     *      This is a collective operation over all ranks of this communicator.
     * @exception Throws an Exception if MPI returns an error code
     * @return The duplicated communicator, owning its handle
     */
    [[nodiscard]] MPICommunicator duplicate() const {
        auto new_communicator = MPI_Comm{ MPI_COMM_NULL };
        const auto error_code = MPI_Comm_dup(communicator_, &new_communicator);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunicator::duplicate: Duplicating the communicator returned the error: {}", error_code);
        return MPICommunicator{ new_communicator, true };
    }

    /**
     * @brief Returns the process group underlying this communicator
     * @exception Throws an Exception if MPI returns an error code
     * @return The group, owning its handle
     */
    [[nodiscard]] MPIGroup get_group() const {
        auto group = MPI_Group{ MPI_GROUP_NULL };
        const auto error_code = MPI_Comm_group(communicator_, &group);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunicator::get_group: Retrieving the group returned the error: {}", error_code);
        return MPIGroup{ group };
    }

    /**
     * @brief Creates a new communicator spanning the given group, which must be a subset of this
     *      communicator's group. This is a collective operation over all ranks of this communicator.
     * @param group The group the new communicator spans
     * @exception Throws an Exception if MPI returns an error code
     * @return The new communicator, owning its handle (or null on ranks that are not in the group)
     */
    [[nodiscard]] MPICommunicator create(const MPIGroup& group) const {
        auto new_communicator = MPI_Comm{ MPI_COMM_NULL };
        const auto error_code = MPI_Comm_create(communicator_, group.get(), &new_communicator);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunicator::create: Creating the communicator returned the error: {}", error_code);
        return MPICommunicator{ new_communicator, new_communicator != MPI_COMM_NULL };
    }

    /**
     * @brief Returns the number of ranks in this communicator
     * @exception Throws an Exception if MPI returns an error code
     * @return The number of ranks
     */
    [[nodiscard]] int get_number_ranks() const {
        auto number_ranks = int{ 0 };
        const auto error_code = MPI_Comm_size(communicator_, &number_ranks);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunicator::get_number_ranks: returned the error: {}", error_code);
        return number_ranks;
    }

    /**
     * @brief Returns the number of ranks in this communicator cast to std::size_t
     * @exception Throws an Exception if MPI returns an error code
     * @return The number of ranks
     */
    [[nodiscard]] std::size_t get_number_ranks_cast() const {
        return utility::safe_cast<std::size_t>(get_number_ranks());
    }

    /**
     * @brief Returns this rank's id within this communicator in a type-safe manner
     * @exception Throws an Exception if MPI returns an error code
     * @return This rank's id
     */
    [[nodiscard]] MPIRank get_my_rank() const {
        auto my_rank = int{ 0 };
        const auto error_code = MPI_Comm_rank(communicator_, &my_rank);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunicator::get_my_rank: returned the error: {}", error_code);
        return MPIRank{ my_rank };
    }

    /**
     * @brief Checks if this rank is the root rank (rank 0) within this communicator
     * @exception Throws an Exception if MPI returns an error code
     * @return True iff this rank is the root rank
     */
    [[nodiscard]] bool is_root_rank() const {
        return get_my_rank() == MPIRank::root_rank();
    }

    /**
     * @brief Checks if this communicator is the null communicator (MPI_COMM_NULL)
     * @return True iff this communicator is null
     */
    [[nodiscard]] bool is_null() const noexcept {
        return communicator_ == MPI_COMM_NULL;
    }

    /**
     * @brief Checks if this communicator owns (and will free) its underlying handle
     * @return True iff this communicator owns its handle
     */
    [[nodiscard]] bool owns() const noexcept {
        return is_owning_;
    }

    /**
     * @brief Returns the underlying MPI_Comm handle to hand to raw MPI calls
     * @return The underlying handle
     */
    [[nodiscard]] MPI_Comm get() const noexcept {
        return communicator_;
    }

private:
    // The Cartesian communicator wraps handles produced by MPI_Cart_create/_sub via the private constructor.
    friend class MPICartesianCommunicator;

    /**
     * @brief Constructs a communicator wrapping the given handle
     * @param communicator The handle to wrap
     * @param is_owning Whether this communicator owns (and will free) the handle
     */
    constexpr MPICommunicator(const MPI_Comm communicator, const bool is_owning) noexcept
        : communicator_(communicator)
        , is_owning_(is_owning) { }

    MPI_Comm communicator_{ MPI_COMM_NULL };
    bool is_owning_{ false };
};

inline const MPICommunicator MPICommunicator::World{ MPI_COMM_WORLD, false };
inline const MPICommunicator MPICommunicator::Self{ MPI_COMM_SELF, false };

} // namespace mpiPP
