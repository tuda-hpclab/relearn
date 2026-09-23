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

#include "mpi-wrapper/communicator/MPICommunicator.h"
#include "mpi-wrapper/core/MPICounters.h"
#include "mpi-wrapper/core/MPIRank.h"
#include "mpi-wrapper/core/MPITypes.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <utility>
#include <vector>

namespace mpiPP {

namespace detail {
/**
 * @brief Wire-format pair for MPI_MINLOC / MPI_MAXLOC reductions. Its layout (a value followed by
 *      an int location) matches the predefined MPI_<T>_INT datatypes, so it can be reduced with the
 *      built-in location operations. The location is a raw int rather than an MPIRank on purpose:
 *      MPIRank is a bit-field and cannot alias the plain int that MPI writes into this slot.
 */
template <typename T>
struct value_location_pair {
    T value;
    int location;
};
} // namespace detail

/**
 * @brief Value types for which MPI provides a predefined value-plus-location datatype, and which
 *      can therefore take part in a MINLOC / MAXLOC reduction.
 */
template <typename T>
concept MPILocatable = std::same_as<T, short> || std::same_as<T, int> || std::same_as<T, long> || std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, long double>;

/**
 * @brief The result of a MINLOC / MAXLOC reduction: the extreme value together with the rank that
 *      provided it (the smallest such rank in case of ties).
 */
template <MPILocatable T>
struct ValueLocation {
    T value;          ///< The reduced minimum or maximum value
    MPIRank location; ///< The rank that provided that value
};

namespace MPIReductions {
/**
 * @brief Maps a locatable value type to the predefined MPI datatype pairing it with an int
 *      location, for use with MPI_MINLOC / MPI_MAXLOC.
 * @tparam T The locatable value type
 * @return The matching MPI_<T>_INT datatype
 */
template <MPILocatable T>
[[nodiscard]] inline MPI_Datatype value_location_type() {
    if constexpr (std::same_as<T, short>) {
        return MPI_SHORT_INT;
    } else if constexpr (std::same_as<T, int>) {
        return MPI_2INT;
    } else if constexpr (std::same_as<T, long>) {
        return MPI_LONG_INT;
    } else if constexpr (std::same_as<T, float>) {
        return MPI_FLOAT_INT;
    } else if constexpr (std::same_as<T, double>) {
        return MPI_DOUBLE_INT;
    } else {
        return MPI_LONG_DOUBLE_INT;
    }
}
/**
 * @brief Reduces the provided values and returns the sum on all ranks
 * @tparam T The type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The sum of all local values
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T all_reduce_sum(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_SUM, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the minimum on all ranks
 * @tparam T The type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The minimum of all local values
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T all_reduce_min(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_MIN, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the maximum on all ranks
 * @tparam T The type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The maximum of all local values
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T all_reduce_max(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_MAX, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the product on all ranks
 * @tparam T The type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The product of all local values
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T all_reduce_prod(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 1 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_PROD, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns logical or
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The logical or of all local values
 */
[[nodiscard]] inline bool all_reduce_or(const bool value, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto val = value ? std::uint8_t{ 1 } : std::uint8_t{ 0 };
    const auto total_value = all_reduce_max(val, communicator);
    return total_value == 1;
}

/**
 * @brief Reduces the provided values and returns logical and
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The logical and of all local values
 */
[[nodiscard]] inline bool all_reduce_and(const bool value, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto val = value ? std::uint8_t{ 1 } : std::uint8_t{ 0 };
    const auto total_value = all_reduce_min(val, communicator);
    return total_value == 1;
}

/**
 * @brief Reduces the provided values and returns the bitwise and on all ranks
 * @tparam T The integral type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The bitwise and of all local values
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] T all_reduce_bitwise_and(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_BAND, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the bitwise or on all ranks
 * @tparam T The integral type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The bitwise or of all local values
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] T all_reduce_bitwise_or(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_BOR, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the bitwise exclusive or on all ranks
 * @tparam T The integral type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The bitwise exclusive or of all local values
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] T all_reduce_bitwise_xor(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_BXOR, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the sum on the root rank
 * @tparam T The type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The sum of all local values on the root rank, T(0) on all other ranks
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T reduce_sum(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_SUM, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the minimum on the root rank
 * @tparam T The type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The minimum of all local values on the root rank, T(0) on all other ranks
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T reduce_min(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_MIN, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the maximum on the root rank
 * @tparam T The type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The maximum of all local values on the root rank, T(0) on all other ranks
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T reduce_max(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_MAX, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the product on the root rank
 * @tparam T The type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The product of all local values on the root rank, T(1) on all other ranks
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T reduce_prod(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 1 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_PROD, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns logical or on the root rank
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The logical or on the root rank; false on every other rank
 */
[[nodiscard]] inline bool reduce_or(const bool value, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto val = value ? std::uint8_t{ 1 } : std::uint8_t{ 0 };
    const auto total_value = reduce_max(val, communicator);
    return total_value == 1;
}

/**
 * @brief Reduces the provided values and returns logical and on the root rank
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The logical and on the root rank; false on every other rank
 */
[[nodiscard]] inline bool reduce_and(const bool value, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto val = value ? std::uint8_t{ 1 } : std::uint8_t{ 0 };
    const auto total_value = reduce_min(val, communicator);
    return total_value == 1;
}

/**
 * @brief Reduces the provided values and returns the bitwise and on the root rank
 * @tparam T The integral type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The bitwise and of all local values on the root rank, T(0) on all other ranks
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] T reduce_bitwise_and(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_BAND, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the bitwise or on the root rank
 * @tparam T The integral type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The bitwise or of all local values on the root rank, T(0) on all other ranks
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] T reduce_bitwise_or(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_BOR, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the bitwise exclusive or on the root rank
 * @tparam T The integral type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The bitwise exclusive or of all local values on the root rank, T(0) on all other ranks
 */
template <MPIReductionArithmetic T>
    requires std::integral<T>
[[nodiscard]] T reduce_bitwise_xor(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_BXOR, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the maximum together with the rank holding it on
 *      all ranks
 * @tparam T The locatable type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The maximum of all local values and the (smallest) rank that provided it
 */
template <MPILocatable T>
[[nodiscard]] ValueLocation<T> all_reduce_max_location(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto local = detail::value_location_pair<T>{ value, communicator.get_my_rank().get_rank() };
    auto result = detail::value_location_pair<T>{};

    const auto error_code = MPI_Allreduce(&local, &result, 1, value_location_type<T>(), MPI_MAXLOC, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (maxloc) all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(local));
    MPICounters::add_to_received(sizeof(result));

    return ValueLocation<T>{ result.value, MPIRank{ result.location } };
}

/**
 * @brief Reduces the provided values and returns the minimum together with the rank holding it on
 *      all ranks
 * @tparam T The locatable type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The minimum of all local values and the (smallest) rank that provided it
 */
template <MPILocatable T>
[[nodiscard]] ValueLocation<T> all_reduce_min_location(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto local = detail::value_location_pair<T>{ value, communicator.get_my_rank().get_rank() };
    auto result = detail::value_location_pair<T>{};

    const auto error_code = MPI_Allreduce(&local, &result, 1, value_location_type<T>(), MPI_MINLOC, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (minloc) all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(local));
    MPICounters::add_to_received(sizeof(result));

    return ValueLocation<T>{ result.value, MPIRank{ result.location } };
}

/**
 * @brief Reduces the provided values and returns the maximum together with the rank holding it on
 *      the root rank
 * @tparam T The locatable type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The maximum and its (smallest) rank on the root rank, nothing of interest on all other ranks
 */
template <MPILocatable T>
[[nodiscard]] ValueLocation<T> reduce_max_location(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto local = detail::value_location_pair<T>{ value, communicator.get_my_rank().get_rank() };
    auto result = detail::value_location_pair<T>{};

    const auto error_code = MPI_Reduce(&local, &result, 1, value_location_type<T>(), MPI_MAXLOC, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (maxloc) all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(local));
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(local));
    }

    return ValueLocation<T>{ result.value, MPIRank{ result.location } };
}

/**
 * @brief Reduces the provided values and returns the minimum together with the rank holding it on
 *      the root rank
 * @tparam T The locatable type of data to reduce
 * @param value The local value
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The minimum and its (smallest) rank on the root rank, nothing of interest on all other ranks
 */
template <MPILocatable T>
[[nodiscard]] ValueLocation<T> reduce_min_location(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto local = detail::value_location_pair<T>{ value, communicator.get_my_rank().get_rank() };
    auto result = detail::value_location_pair<T>{};

    const auto error_code = MPI_Reduce(&local, &result, 1, value_location_type<T>(), MPI_MINLOC, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (minloc) all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(local));
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(local));
    }

    return ValueLocation<T>{ result.value, MPIRank{ result.location } };
}

/**
 * @brief Collectively selects the median of all elements provided across the communicator and
 *      returns it on the root rank. A value is a median if at least half of all elements are
 *      less than or equal to it and at least half of all elements are greater than or equal to
 *      it; for an even total count the lower of the two middle elements is returned. The local
 *      containers may differ in size and may be empty on some ranks. No rank transmits its
 *      elements as a whole: the candidate range is narrowed iteratively around weighted
 *      median-of-medians pivots, so each of the O(log n) rounds exchanges only one pivot
 *      proposal per rank.
 * @tparam Range A contiguous range with an element type accepted by arithmetic MPI reductions
 * @param values The local elements, not modified; the elements must be totally ordered, so
 *      floating-point NaNs are not allowed
 * @param root The rank that receives the median, defaults to MPIRank::root_rank()
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if root is not a rank of the communicator, if the containers
 *      of all ranks are empty, or if mpi reports an error
 * @return The median of all elements on the root rank, T(0) on all other ranks
 */
template <MPIReductionArithmeticRange Range>
[[nodiscard]] std::ranges::range_value_t<Range> reduce_median(const Range& values, const MPIRank root = MPIRank::root_rank(), const MPICommunicator& communicator = MPICommunicator::World) {
    using T = std::ranges::range_value_t<Range>;

    const auto number_ranks = communicator.get_number_ranks();
    utility::Exception::check(root.get_rank() < number_ranks, "MPIReductions::reduce_median: There are {} ranks, but the median was requested on rank {}", number_ranks, root.get_rank());

    // Sorted local working copy: counting elements relative to a pivot is then a binary search,
    // and discarding everything on one side of a pivot only shrinks the window [low, high)
    auto candidates = std::vector<T>(std::ranges::begin(values), std::ranges::end(values));
    std::ranges::sort(candidates);

    const auto total_count = all_reduce_sum(utility::safe_cast<std::uint64_t>(candidates.size()), communicator);
    utility::Exception::check(total_count > 0, "MPIReductions::reduce_median: Cannot take the median when all ranks provide empty containers");

    // The median is the k-th smallest remaining candidate (1-based); ceil(total / 2) selects the
    // lower middle element, which satisfies the median definition for even counts as well
    auto k = total_count / 2 + total_count % 2;

    auto low = std::size_t{ 0 };
    auto high = candidates.size();

    const auto value_type = MPITypes::convert_type_to_mpi_type<T>();
    const auto count_type = MPITypes::convert_type_to_mpi_type<std::uint64_t>();

    auto proposed_pivots = std::vector<T>(utility::safe_cast<std::size_t>(number_ranks));
    auto proposed_weights = std::vector<std::uint64_t>(utility::safe_cast<std::size_t>(number_ranks));

    while (true) {
        // Every rank proposes the middle of its remaining window, weighted by the window size
        const auto local_weight = utility::safe_cast<std::uint64_t>(high - low);
        const auto local_pivot = local_weight > 0 ? candidates[low + (high - low) / 2] : T{ 0 };

        const auto pivots_error = MPI_Allgather(&local_pivot, 1, value_type, proposed_pivots.data(), 1, value_type, communicator.get());
        utility::Exception::check(pivots_error == MPI_SUCCESS, "MPIReductions::reduce_median: Gathering the pivot proposals returned the error: {}", pivots_error);
        const auto weights_error = MPI_Allgather(&local_weight, 1, count_type, proposed_weights.data(), 1, count_type, communicator.get());
        utility::Exception::check(weights_error == MPI_SUCCESS, "MPIReductions::reduce_median: Gathering the pivot weights returned the error: {}", weights_error);

        MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks) * (sizeof(T) + sizeof(std::uint64_t)));
        MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(number_ranks) * (sizeof(T) + sizeof(std::uint64_t)));

        // The pivot is the weighted median of the proposals: every proposal is an actual
        // candidate of some rank, and at least a quarter of the remaining candidates lie on
        // either side of it, so every round discards a constant fraction of the candidates
        auto proposals = std::vector<std::pair<T, std::uint64_t>>{};
        auto remaining_count = std::uint64_t{ 0 };
        for (auto rank = std::size_t{ 0 }; rank < proposed_weights.size(); rank++) {
            if (proposed_weights[rank] > 0) {
                proposals.emplace_back(proposed_pivots[rank], proposed_weights[rank]);
                remaining_count += proposed_weights[rank];
            }
        }
        std::ranges::sort(proposals);

        auto pivot = T{ 0 };
        auto cumulative_weight = std::uint64_t{ 0 };
        for (const auto& [proposal, weight] : proposals) {
            pivot = proposal;
            cumulative_weight += weight;
            if (cumulative_weight >= remaining_count / 2 + remaining_count % 2) {
                break;
            }
        }

        // Count globally how the remaining candidates compare to the pivot
        const auto window_begin = candidates.begin() + utility::safe_cast<std::ptrdiff_t>(low);
        const auto window_end = candidates.begin() + utility::safe_cast<std::ptrdiff_t>(high);
        const auto first_equal = std::lower_bound(window_begin, window_end, pivot);
        const auto first_greater = std::upper_bound(first_equal, window_end, pivot);

        const auto local_less = utility::safe_cast<std::uint64_t>(std::distance(window_begin, first_equal));
        const auto local_equal = utility::safe_cast<std::uint64_t>(std::distance(first_equal, first_greater));

        const auto local_counts = std::array<std::uint64_t, 2>{ local_less, local_equal };
        auto global_counts = std::array<std::uint64_t, 2>{ 0, 0 };
        const auto count_error = MPI_Allreduce(local_counts.data(), global_counts.data(), 2, count_type, MPI_SUM, communicator.get());
        utility::Exception::check(count_error == MPI_SUCCESS, "MPIReductions::reduce_median: Reducing the pivot counts returned the error: {}", count_error);

        MPICounters::add_to_sent(sizeof(local_counts));
        MPICounters::add_to_received(sizeof(global_counts));

        const auto global_less = global_counts[0];
        const auto global_equal = global_counts[1];

        if (k <= global_less) {
            // The median is smaller than the pivot: discard the pivot and everything above it
            high = low + utility::safe_cast<std::size_t>(local_less);
        } else if (k <= global_less + global_equal) {
            // The pivot is the k-th smallest element and therefore the median
            return communicator.get_my_rank() == root ? pivot : T{ 0 };
        } else {
            // The median is larger than the pivot: discard the pivot and everything below it
            low += utility::safe_cast<std::size_t>(local_less + local_equal);
            k -= global_less + global_equal;
        }
    }
}
} // namespace MPIReductions

} // namespace mpiPP
