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

#include <cpp-utility/Exception.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
#include <fmt/ostream.h>
#pragma GCC diagnostic pop

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <string_view>
#include <utility>
#include <vector>

namespace mpiPP {

/**
 * This type reflects a point in a Cartesian topology in a type-safe manner. A
 * coordinate consists of one non-negative index per dimension of the grid, in
 * the same order MPI uses for MPI_Cart_coords / MPI_Cart_rank.
 */
class MPICartesianCoordinates {
public:
    /**
     * @brief Returns the uninitialized coordinate that can be used for debugging purposes.
     * @return The uninitialized coordinate
     */
    [[nodiscard]] static MPICartesianCoordinates uninitialized() { return MPICartesianCoordinates{}; }

    /**
     * @brief Returns the origin (all-zero) coordinate for a grid of the given dimensionality.
     * @param number_dimensions The number of dimensions of the grid, must be > 0
     * @exception Throws an Exception if number_dimensions == 0
     * @return The origin coordinate with number_dimensions zero-entries
     */
    [[nodiscard]] static MPICartesianCoordinates origin(const std::size_t number_dimensions) {
        utility::Exception::check(number_dimensions > 0, "MPICartesianCoordinates::origin: The number of dimensions must be > 0.");
        return MPICartesianCoordinates{ std::vector<int>(number_dimensions, 0) };
    }

    /**
     * @brief Converts a linear rank into its Cartesian coordinate.
     *
     * Uses the row-major ordering MPI applies for Cartesian topologies, i.e. the
     * last dimension varies fastest. This is the inverse of to_rank(...).
     * @param dimensions The extent of the grid in each dimension, must be non-empty and every entry > 0
     * @param rank The linear rank, must be initialized and in [0, product(dimensions))
     * @exception Throws an Exception if dimensions is empty, any extent is <= 0, the grid does not fit into an MPIRank, or rank is out of range
     * @return The coordinate corresponding to rank
     */
    [[nodiscard]] static MPICartesianCoordinates from_rank(const std::vector<int>& dimensions, const MPIRank rank) {
        const auto number_ranks = number_ranks_in_grid(dimensions, "from_rank");

        auto remainder = std::int64_t{ rank.get_rank() };
        utility::Exception::check(remainder < number_ranks, "MPICartesianCoordinates::from_rank: The rank {} is out of range for a grid with {} ranks.", remainder, number_ranks);

        auto coordinates = std::vector<int>(dimensions.size());
        for (auto dimension = dimensions.size(); dimension-- > 0;) {
            coordinates[dimension] = static_cast<int>(remainder % dimensions[dimension]);
            remainder /= dimensions[dimension];
        }
        return MPICartesianCoordinates{ std::move(coordinates) };
    }

    /**
     * @brief Default constructs an uninitialized coordinate
     */
    MPICartesianCoordinates() noexcept = default;

    /**
     * @brief Constructs an initialized coordinate from the per-dimension indices
     * @param coordinates The index in each dimension, every entry must be >= 0
     * @exception Throws an Exception if any entry is < 0
     */
    explicit MPICartesianCoordinates(std::vector<int> coordinates)
        : coordinates_{ std::move(coordinates) }
        , is_initialized_{ true } {

        for (const auto coordinate : coordinates_) {
            utility::Exception::check(coordinate >= 0, "MPICartesianCoordinates::MPICartesianCoordinates: Every coordinate must be >= 0: {}.", coordinate);
        }
    }

    /**
     * @brief Constructs an initialized coordinate from the per-dimension indices
     * @param coordinates The index in each dimension, every entry must be >= 0
     * @exception Throws an Exception if any entry is < 0
     */
    MPICartesianCoordinates(const std::initializer_list<int> coordinates)
        : MPICartesianCoordinates{ std::vector<int>{ coordinates } } { }

    [[nodiscard]] friend std::strong_ordering operator<=>(const MPICartesianCoordinates& first, const MPICartesianCoordinates& second) = default;

    /**
     * @brief Check if the coordinate is initialized
     * @return true iff the coordinate is initialized
     */
    [[nodiscard]] bool is_initialized() const noexcept { return is_initialized_; }

    /**
     * @brief Returns the number of dimensions of the grid this coordinate lives in
     * @return The number of dimensions, 0 for an uninitialized coordinate
     */
    [[nodiscard]] std::size_t get_number_dimensions() const noexcept { return coordinates_.size(); }

    /**
     * @brief Returns the index in a single dimension
     * @param dimension The dimension to query, must be < get_number_dimensions()
     * @exception Throws an Exception if the coordinate is not initialized or dimension is out of range
     * @return The index in the requested dimension, >= 0
     */
    [[nodiscard]] int get_coordinate(const std::size_t dimension) const {
        utility::Exception::check(is_initialized_, "MPICartesianCoordinates::get_coordinate: This coordinate is not initialized.");
        utility::Exception::check(dimension < coordinates_.size(), "MPICartesianCoordinates::get_coordinate: The dimension {} is out of range for {} dimensions.", dimension, coordinates_.size());
        return coordinates_[dimension];
    }

    /**
     * @brief Returns all per-dimension indices
     * @exception Throws an Exception if the coordinate is not initialized
     * @return The stored coordinates
     */
    [[nodiscard]] const std::vector<int>& get_coordinates() const {
        utility::Exception::check(is_initialized_, "MPICartesianCoordinates::get_coordinates: This coordinate is not initialized.");
        return coordinates_;
    }

    /**
     * @brief Converts this coordinate into its linear rank.
     *
     * Uses the row-major ordering MPI applies for Cartesian topologies, i.e. the
     * last dimension varies fastest. This is the inverse of from_rank(...).
     * @param dimensions The extent of the grid in each dimension, must match get_number_dimensions() and every entry > 0
     * @exception Throws an Exception if the coordinate is not initialized, the dimensionality does not match, any extent is <= 0, an index is >= its extent, or the grid does not fit into an MPIRank
     * @return The linear rank corresponding to this coordinate
     */
    [[nodiscard]] MPIRank to_rank(const std::vector<int>& dimensions) const {
        utility::Exception::check(is_initialized_, "MPICartesianCoordinates::to_rank: This coordinate is not initialized.");
        utility::Exception::check(dimensions.size() == coordinates_.size(), "MPICartesianCoordinates::to_rank: The grid has {} dimensions, but the coordinate has {}.", dimensions.size(), coordinates_.size());
        static_cast<void>(number_ranks_in_grid(dimensions, "to_rank"));

        auto linear = std::int64_t{ 0 };
        for (std::size_t dimension = 0; dimension < dimensions.size(); ++dimension) {
            utility::Exception::check(coordinates_[dimension] < dimensions[dimension], "MPICartesianCoordinates::to_rank: The index {} is out of range for dimension {} with extent {}.", coordinates_[dimension], dimension, dimensions[dimension]);
            linear = linear * dimensions[dimension] + coordinates_[dimension];
        }
        return MPIRank{ static_cast<int>(linear) };
    }

    /**
     * @brief Prints the object's represented Cartesian coordinate
     * @param os The out-stream in which the object is printed
     * @param coordinates The coordinate to print
     * @return The argument os to allow chaining
     */
    friend std::ostream& operator<<(std::ostream& os, const MPICartesianCoordinates& coordinates) {
        if (!coordinates.is_initialized_) {
            os << "MPICartesianCoordinates: uninitialized";
            return os;
        }

        os << "MPICartesianCoordinates: (";
        for (std::size_t dimension = 0; dimension < coordinates.coordinates_.size(); ++dimension) {
            if (dimension != 0) {
                os << ", ";
            }
            os << coordinates.coordinates_[dimension];
        }
        os << ")";
        return os;
    }

private:
    /**
     * @brief Validates a grid description and returns the number of ranks it contains.
     * @param dimensions The extent of the grid in each dimension, must be non-empty and every entry > 0
     * @param caller The calling function's name, only used for the exception message
     * @exception Throws an Exception if dimensions is empty, any extent is <= 0, or the grid does not fit into an MPIRank
     * @return The product of all extents, guaranteed to fit into an int
     */
    [[nodiscard]] static std::int64_t number_ranks_in_grid(const std::vector<int>& dimensions, const std::string_view caller) {
        utility::Exception::check(!dimensions.empty(), "MPICartesianCoordinates::{}: The grid must have at least one dimension.", caller);

        // The largest valid rank must stay within the range an MPIRank accepts
        constexpr auto maximum_ranks = std::int64_t{ std::numeric_limits<int>::max() / 2 };

        auto number_ranks = std::int64_t{ 1 };
        for (const auto extent : dimensions) {
            utility::Exception::check(extent > 0, "MPICartesianCoordinates::{}: Every grid extent must be > 0: {}.", caller, extent);
            number_ranks *= extent;
            utility::Exception::check(number_ranks <= maximum_ranks, "MPICartesianCoordinates::{}: The grid is too large to fit into an MPIRank.", caller);
        }
        return number_ranks;
    }

    std::vector<int> coordinates_{};
    bool is_initialized_{ false };
};

} // namespace mpiPP

template <>
struct fmt::formatter<mpiPP::MPICartesianCoordinates> : ostream_formatter { };

namespace std {
template <>
struct hash<mpiPP::MPICartesianCoordinates> {
    using argument_type = mpiPP::MPICartesianCoordinates;
    using result_type = std::size_t;

    result_type operator()(const argument_type& coordinates) const {
        if (!coordinates.is_initialized()) {
            // All bits are set
            return std::numeric_limits<result_type>::max();
        }

        // Boost-style hash combine over the per-dimension indices
        auto seed = result_type{ 0 };
        for (const auto coordinate : coordinates.get_coordinates()) {
            const auto value = std::hash<int>{}(coordinate);
            seed ^= value + result_type{ 0x9e3779b9 } + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};
} // namespace std
