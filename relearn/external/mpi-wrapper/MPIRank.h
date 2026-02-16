#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cpp-utility/Exception.hpp"
#include "cpp-utility/ranges/Functional.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
#include <fmt/ostream.h>
#pragma GCC diagnostic pop

#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include <compare>
#include <cstddef>
#include <functional>
#include <limits>

namespace mpiPP {

/**
 * This type reflects an MPI rank in a type-safe manner
 */
class MPIRank {
    static constexpr auto id_bit_count = sizeof(int) * 8 - 1;

public:
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
     * @brief Create a range of MPIRanks within the range [begin, end)
     *
     * @param begin begin of the range, must be >= 0
     * @param end end of the range, must be >= begin and < (1<<30)-1
     * @exception Throws a Exception if begin < 0 or end < begin or if end is larger than (1<<30)-1
     * @return constexpr auto range of MPIRanks
     */
    [[nodiscard]] static auto range(const int begin, const int end) {
        utility::Exception::check(begin >= 0, "MPIRank::range(int, int): begin must be larger than or egal to 0, was {}", begin);
        utility::Exception::check(begin <= end, "MPIRank::range(int, int): begin must be smaller than end, were {}, {}", begin, end);

        constexpr auto maximum_rank = std::numeric_limits<int>::max() / 2;
        utility::Exception::check(end < maximum_rank, "MPIRank::range(int, int): end must be smaller than {}: {}.", maximum_rank, end);
        return ranges::views::iota(begin, end) | ranges::views::transform(utility::construct<MPIRank>);
    }

    /**
     * @brief Create a range of MPIRanks within the range [0, size)
     *
     * @param size size of the range, must be > 0
     * @exception Throws a Exception if size <= 0
     * @return auto range of MPIRanks
     */
    [[nodiscard]] static auto range(const int size) {
        utility::Exception::check(size > 0, "MPIRank::range(int): size must be larger than 0, was {}", size);
        return range(0, size);
    }

    /**
     * @brief Default constructs an uninitialized rank
     */
    constexpr MPIRank() noexcept = default;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
    /**
     * @brief Constructs an initialized rank with rank
     * @exception Throws a Exception if rank < 0 or if it is larger than (1<<30)-1
     */
    constexpr explicit MPIRank(const int rank)
        : actual_rank_{ rank }
        , is_initialized_{ true } {

        constexpr auto maximum_rank = std::numeric_limits<int>::max() / 2;
        utility::Exception::check(rank >= 0, "MPIRank::MPIRank: The actual rank must be >= 0: {}.", rank);
        utility::Exception::check(rank < maximum_rank, "MPIRank::MPIRank: The actual rank must be < {}: {}.", maximum_rank, rank);
    }
#pragma GCC diagnostic pop

    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const MPIRank& first, const MPIRank& second) = default;

    /**
     * @brief Check if the rank is initialized
     * @return true iff the rank is initialized
     */
    [[nodiscard]] constexpr bool is_initialized() const noexcept { return is_initialized_; }

    /**
     * @brief Returns the stored MPI rank as int
     * @exception Throws a Exception if the object is not initialized
     * @return The stored MPI rank
     */
    [[nodiscard]] constexpr int get_rank() const {
        utility::Exception::check(is_initialized_, "MPIRank::get_rank: This MPIRank is not initialized.");
        return actual_rank_;
    }

    /**
     * @brief Returns the stored MPI rank as int
     * @exception Throws a Exception if the object is not initialized
     * @return The stored MPI rank
     */
    [[nodiscard]] constexpr std::size_t get_rank_cast() const {
        utility::Exception::check(is_initialized_, "MPIRank::get_rank_cast: This MPIRank is not initialized.");
        return static_cast<std::size_t>(actual_rank_);
    }

    /**
     * @brief Prints the object's represented MPI rank
     * @param os The out-stream in which the object is printed
     * @return The argument os to allow chaining
     */
    friend std::ostream& operator<<(std::ostream& os, const MPIRank& rank) {
        if (rank.is_initialized_) {
            os << "MPIRank: " << rank.actual_rank_;
        } else {
            os << "MPIRank: uninitialized";
        }
        return os;
    }

private:
    int actual_rank_ : id_bit_count = -1;
    bool is_initialized_ : 1 = false;
};

} // namespace mpiPP

template <>
struct fmt::formatter<mpiPP::MPIRank> : ostream_formatter { };

namespace std {
template <>
struct hash<mpiPP::MPIRank> {
    using argument_type = mpiPP::MPIRank;
    using result_type = std::size_t;

    result_type operator()(const argument_type& mpi_rank) const {
        constexpr auto max = std::numeric_limits<result_type>::max();
        if (!mpi_rank.is_initialized()) {
            // All bits are set
            return max;
        }

        const auto rank = mpi_rank.get_rank();
        return result_type(rank);
    }
};
} // namespace std
