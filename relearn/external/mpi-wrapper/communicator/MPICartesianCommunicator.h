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

#include "mpi-wrapper/communicator/MPICartesianCoordinates.h"
#include "mpi-wrapper/communicator/MPICommunicator.h"
#include "mpi-wrapper/core/MPIRank.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace mpiPP {

/**
 * @brief A communicator carrying an MPI Cartesian topology (MPI_Cart_create).
 *
 * This ties the plain MPICommunicator together with MPICartesianCoordinates: it owns
 * the Cartesian communicator and translates between linear ranks and grid coordinates
 * through the MPI topology (MPI_Cart_coords / MPI_Cart_rank), offers neighbour lookups
 * (MPI_Cart_shift) and partitioning into lower-dimensional grids (MPI_Cart_sub).
 *
 * Being backed by a move-only MPICommunicator, it is itself move-only.
 */
class MPICartesianCommunicator {
public:
    /**
     * @brief The source and destination ranks of a shift along one grid dimension.
     *      A rank is uninitialized if the shift falls off a non-periodic boundary (MPI_PROC_NULL).
     */
    struct Shift {
        MPIRank source;
        MPIRank destination;
    };

    /**
     * @brief Suggests a balanced grid shape for the given number of ranks (MPI_Dims_create).
     * @param number_ranks The number of ranks to distribute
     * @param number_dimensions The number of grid dimensions, must be > 0
     * @exception Throws an Exception if number_dimensions == 0 or MPI returns an error code
     * @return A balanced dimension vector whose product equals number_ranks
     */
    [[nodiscard]] static std::vector<int> create_dimensions(const int number_ranks, const std::size_t number_dimensions) {
        utility::Exception::check(number_dimensions > 0, "MPICartesianCommunicator::create_dimensions: The number of dimensions must be > 0.");
        auto dimensions = std::vector<int>(number_dimensions, 0);
        const auto error_code = MPI_Dims_create(number_ranks, utility::safe_cast<int>(number_dimensions), dimensions.data());
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICartesianCommunicator::create_dimensions: returned the error: {}", error_code);
        return dimensions;
    }

    /**
     * @brief Creates a Cartesian topology on top of a base communicator (MPI_Cart_create).
     *      This is a collective operation over all ranks of the base communicator.
     * @param dimensions The extent of the grid in each dimension, must be non-empty
     * @param periods Whether each dimension wraps around (non-zero) or not (zero); same size as dimensions
     * @param reorder Whether MPI may reorder the ranks to match the hardware topology
     * @param base The communicator to build on top of, defaults to MPICommunicator::World
     * @exception Throws an Exception if the arguments are inconsistent or MPI returns an error code
     * @return The Cartesian communicator, owning its handle; null on ranks excluded when the grid
     *      contains fewer positions than the base communicator has ranks
     */
    [[nodiscard]] static MPICartesianCommunicator create(const std::vector<int>& dimensions, const std::vector<int>& periods, const bool reorder = true, const MPICommunicator& base = MPICommunicator::World) {
        utility::Exception::check(!dimensions.empty(), "MPICartesianCommunicator::create: The grid must have at least one dimension.");
        utility::Exception::check(dimensions.size() == periods.size(), "MPICartesianCommunicator::create: dimensions and periods must have the same size: {} vs {}.", dimensions.size(), periods.size());

        const auto number_dimensions = utility::safe_cast<int>(dimensions.size());
        auto new_communicator = MPI_Comm{ MPI_COMM_NULL };
        const auto error_code = MPI_Cart_create(base.get(), number_dimensions, dimensions.data(), periods.data(), reorder ? 1 : 0, &new_communicator);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICartesianCommunicator::create: Creating the Cartesian communicator returned the error: {}", error_code);

        return MPICartesianCommunicator{ MPICommunicator{ new_communicator, new_communicator != MPI_COMM_NULL } };
    }

    /**
     * @brief Default-constructs a null Cartesian communicator.
     */
    MPICartesianCommunicator() noexcept = default;

    /**
     * @brief Returns the number of dimensions of the grid (MPI_Cartdim_get)
     * @exception Throws an Exception if MPI returns an error code
     * @return The number of dimensions
     */
    [[nodiscard]] std::size_t get_number_dimensions() const {
        auto number_dimensions = int{ 0 };
        const auto error_code = MPI_Cartdim_get(communicator_.get(), &number_dimensions);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICartesianCommunicator::get_number_dimensions: returned the error: {}", error_code);
        return utility::safe_cast<std::size_t>(number_dimensions);
    }

    /**
     * @brief Returns the extent of the grid in each dimension (MPI_Cart_get)
     * @exception Throws an Exception if MPI returns an error code
     * @return The per-dimension extents
     */
    [[nodiscard]] std::vector<int> get_dimensions() const {
        const auto number_dimensions = get_number_dimensions();
        auto dimensions = std::vector<int>(number_dimensions);
        auto periods = std::vector<int>(number_dimensions);
        auto coordinates = std::vector<int>(number_dimensions);
        const auto error_code = MPI_Cart_get(communicator_.get(), utility::safe_cast<int>(number_dimensions), dimensions.data(), periods.data(), coordinates.data());
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICartesianCommunicator::get_dimensions: returned the error: {}", error_code);
        return dimensions;
    }

    /**
     * @brief Returns the calling rank's coordinate within the grid (MPI_Cart_get)
     * @exception Throws an Exception if MPI returns an error code
     * @return This rank's Cartesian coordinate
     */
    [[nodiscard]] MPICartesianCoordinates get_my_coordinates() const {
        const auto number_dimensions = get_number_dimensions();
        auto dimensions = std::vector<int>(number_dimensions);
        auto periods = std::vector<int>(number_dimensions);
        auto coordinates = std::vector<int>(number_dimensions);
        const auto error_code = MPI_Cart_get(communicator_.get(), utility::safe_cast<int>(number_dimensions), dimensions.data(), periods.data(), coordinates.data());
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICartesianCommunicator::get_my_coordinates: returned the error: {}", error_code);
        return MPICartesianCoordinates{ std::move(coordinates) };
    }

    /**
     * @brief Converts a linear rank into its grid coordinate (MPI_Cart_coords)
     * @param rank The rank to translate, must be a valid rank of this communicator
     * @exception Throws an Exception if MPI returns an error code
     * @return The Cartesian coordinate of rank
     */
    [[nodiscard]] MPICartesianCoordinates get_coordinates(const MPIRank rank) const {
        const auto number_dimensions = get_number_dimensions();
        auto coordinates = std::vector<int>(number_dimensions);
        const auto error_code = MPI_Cart_coords(communicator_.get(), rank.get_rank(), utility::safe_cast<int>(number_dimensions), coordinates.data());
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICartesianCommunicator::get_coordinates: returned the error: {}", error_code);
        return MPICartesianCoordinates{ std::move(coordinates) };
    }

    /**
     * @brief Converts a grid coordinate into its linear rank (MPI_Cart_rank)
     * @param coordinates The coordinate to translate, must have get_number_dimensions() entries
     * @exception Throws an Exception if the dimensionality does not match or MPI returns an error code
     * @return The linear rank at coordinates
     */
    [[nodiscard]] MPIRank get_rank(const MPICartesianCoordinates& coordinates) const {
        const auto& values = coordinates.get_coordinates();
        utility::Exception::check(values.size() == get_number_dimensions(), "MPICartesianCommunicator::get_rank: The coordinate has {} dimensions, but the grid has {}.", values.size(), get_number_dimensions());
        auto rank = int{ 0 };
        const auto error_code = MPI_Cart_rank(communicator_.get(), values.data(), &rank);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICartesianCommunicator::get_rank: returned the error: {}", error_code);
        return MPIRank{ rank };
    }

    /**
     * @brief Returns the neighbours reached by shifting along one dimension (MPI_Cart_shift)
     * @param direction The dimension to shift along, in [0, get_number_dimensions())
     * @param displacement The shift distance (positive: upwards, negative: downwards)
     * @exception Throws an Exception if MPI returns an error code
     * @return The source and destination ranks; uninitialized where the shift leaves a non-periodic grid
     */
    [[nodiscard]] Shift shift(const int direction, const int displacement) const {
        auto source = int{ MPI_PROC_NULL };
        auto destination = int{ MPI_PROC_NULL };
        const auto error_code = MPI_Cart_shift(communicator_.get(), direction, displacement, &source, &destination);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICartesianCommunicator::shift: returned the error: {}", error_code);
        return Shift{ rank_or_uninitialized(source), rank_or_uninitialized(destination) };
    }

    /**
     * @brief Partitions the grid into lower-dimensional Cartesian subgrids (MPI_Cart_sub).
     *      This is a collective operation over all ranks of this communicator.
     * @param keep_dimensions One flag per dimension: non-zero keeps the dimension in the subgrid, zero drops it; size must equal get_number_dimensions()
     * @exception Throws an Exception if the size does not match or MPI returns an error code
     * @return The Cartesian subgrid this rank belongs to, owning its handle
     */
    [[nodiscard]] MPICartesianCommunicator sub(const std::vector<int>& keep_dimensions) const {
        utility::Exception::check(keep_dimensions.size() == get_number_dimensions(), "MPICartesianCommunicator::sub: keep_dimensions has {} entries, but the grid has {} dimensions.", keep_dimensions.size(), get_number_dimensions());
        auto new_communicator = MPI_Comm{ MPI_COMM_NULL };
        const auto error_code = MPI_Cart_sub(communicator_.get(), keep_dimensions.data(), &new_communicator);
        utility::Exception::check(error_code == MPI_SUCCESS, "MPICartesianCommunicator::sub: returned the error: {}", error_code);
        return MPICartesianCommunicator{ MPICommunicator{ new_communicator, new_communicator != MPI_COMM_NULL } };
    }

    /**
     * @brief Returns the number of ranks in this communicator
     * @exception Throws an Exception if MPI returns an error code
     * @return The number of ranks
     */
    [[nodiscard]] int get_number_ranks() const {
        return communicator_.get_number_ranks();
    }

    /**
     * @brief Returns the calling rank's id within this communicator
     * @exception Throws an Exception if MPI returns an error code
     * @return This rank's id
     */
    [[nodiscard]] MPIRank get_my_rank() const {
        return communicator_.get_my_rank();
    }

    /**
     * @brief Checks if this communicator is null
     * @return True iff this communicator is null
     */
    [[nodiscard]] bool is_null() const noexcept {
        return communicator_.is_null();
    }

    /**
     * @brief Returns the underlying plain communicator, e.g. to hand to communicator-aware functions
     * @return The underlying communicator
     */
    [[nodiscard]] const MPICommunicator& as_communicator() const noexcept {
        return communicator_;
    }

    /**
     * @brief Returns the underlying MPI_Comm handle to hand to raw MPI calls
     * @return The underlying handle
     */
    [[nodiscard]] MPI_Comm get() const noexcept {
        return communicator_.get();
    }

private:
    /**
     * @brief Constructs a Cartesian communicator adopting the given (already owning) communicator
     * @param communicator The Cartesian communicator to adopt
     */
    explicit MPICartesianCommunicator(MPICommunicator communicator) noexcept
        : communicator_(std::move(communicator)) { }

    /**
     * @brief Wraps a raw MPI rank, mapping the negative sentinels (MPI_PROC_NULL/MPI_UNDEFINED) to an uninitialized rank
     * @param rank The raw rank
     * @return The wrapped rank, uninitialized iff rank is negative
     */
    [[nodiscard]] static MPIRank rank_or_uninitialized(const int rank) {
        if (rank < 0) {
            return MPIRank::uninitialized_rank();
        }
        return MPIRank{ rank };
    }

    MPICommunicator communicator_{};
};

} // namespace mpiPP
