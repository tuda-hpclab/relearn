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
#include "mpi-wrapper/core/MPITypes.h"
#include "mpi-wrapper/reductions/MPIReductions.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

#include <mpi.h>

#include <algorithm>
#include <limits>

namespace mpiPP {

namespace MPIScans {
/**
 * @brief Computes the inclusive prefix sum: rank i receives the sum of the local values of ranks 0..i
 * @tparam T The type of data to scan
 * @param value The local value
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The sum of the local values of all ranks up to and including this one
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T inclusive_scan_sum(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto result = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Scan(&value, &result, 1, type, MPI_SUM, communicator.get());
    utility::Exception::check(error_code == 0, "Inclusive scan (sum) returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return result;
}

/**
 * @brief Computes the inclusive prefix minimum: rank i receives the minimum of the local values of ranks 0..i
 * @tparam T The type of data to scan
 * @param value The local value
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The minimum of the local values of all ranks up to and including this one
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T inclusive_scan_min(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto result = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Scan(&value, &result, 1, type, MPI_MIN, communicator.get());
    utility::Exception::check(error_code == 0, "Inclusive scan (min) returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return result;
}

/**
 * @brief Computes the inclusive prefix maximum: rank i receives the maximum of the local values of ranks 0..i
 * @tparam T The type of data to scan
 * @param value The local value
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The maximum of the local values of all ranks up to and including this one
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T inclusive_scan_max(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto result = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Scan(&value, &result, 1, type, MPI_MAX, communicator.get());
    utility::Exception::check(error_code == 0, "Inclusive scan (max) returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return result;
}

/**
 * @brief Computes the exclusive prefix sum: rank i receives the sum of the local values of ranks 0..i-1
 * @tparam T The type of data to scan
 * @param value The local value
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The sum of the local values of all preceding ranks, T(0) on the root rank
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T exclusive_scan_sum(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto result = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Exscan(&value, &result, 1, type, MPI_SUM, communicator.get());
    utility::Exception::check(error_code == 0, "Exclusive scan (sum) returned the error: {}", error_code);

    // MPI leaves the result on the root rank undefined, so we set it to the neutral element of the sum.
    if (communicator.is_root_rank()) {
        result = T{ 0 };
    }

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return result;
}

/**
 * @brief Computes the exclusive prefix minimum: rank i receives the minimum of the local values of ranks 0..i-1
 * @tparam T The type of data to scan
 * @param value The local value
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The minimum of the local values of all preceding ranks, the largest representable value on the root rank
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T exclusive_scan_min(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto result = std::numeric_limits<T>::max();

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Exscan(&value, &result, 1, type, MPI_MIN, communicator.get());
    utility::Exception::check(error_code == 0, "Exclusive scan (min) returned the error: {}", error_code);

    // MPI leaves the result on the root rank undefined, so we set it to the neutral element of the minimum.
    if (communicator.is_root_rank()) {
        result = std::numeric_limits<T>::max();
    }

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return result;
}

/**
 * @brief Computes the exclusive prefix maximum: rank i receives the maximum of the local values of ranks 0..i-1
 * @tparam T The type of data to scan
 * @param value The local value
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an Exception if mpi reports an error
 * @return The maximum of the local values of all preceding ranks, the smallest representable value on the root rank
 */
template <MPIReductionArithmetic T>
[[nodiscard]] T exclusive_scan_max(const T value, const MPICommunicator& communicator = MPICommunicator::World) {
    auto result = std::numeric_limits<T>::lowest();

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Exscan(&value, &result, 1, type, MPI_MAX, communicator.get());
    utility::Exception::check(error_code == 0, "Exclusive scan (max) returned the error: {}", error_code);

    // MPI leaves the result on the root rank undefined, so we set it to the neutral element of the maximum.
    if (communicator.is_root_rank()) {
        result = std::numeric_limits<T>::lowest();
    }

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return result;
}

/**
 * @brief Computes the componentwise inclusive prefix sum: rank i receives the componentwise sum of the local values of ranks 0..i
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The local values
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi reports an error
 * @return The componentwise sum of the local values of all ranks up to and including this one
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container inclusive_scan_componentwise_sum(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = MPIReductions::all_reduce_min(number_values, communicator);
    const auto maximum_number_values = MPIReductions::all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Inclusive scan (sum) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Scan(values.data(), buffer.data(), length, type, MPI_SUM, communicator.get());
    utility::Exception::check(error_code == 0, "Inclusive scan (sum) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Computes the componentwise exclusive prefix sum: rank i receives the componentwise sum of the local values of ranks 0..i-1
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The local values
 * @param communicator The communicator to scan within, defaults to the world communicator
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi reports an error
 * @return The componentwise sum of the local values of all preceding ranks, all zeros on the root rank
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container exclusive_scan_componentwise_sum(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = MPIReductions::all_reduce_min(number_values, communicator);
    const auto maximum_number_values = MPIReductions::all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Exclusive scan (sum) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Exscan(values.data(), buffer.data(), length, type, MPI_SUM, communicator.get());
    utility::Exception::check(error_code == 0, "Exclusive scan (sum) componentwise returned the error: {}", error_code);

    // MPI leaves the result on the root rank undefined, so we set it to the neutral element of the sum.
    if (communicator.is_root_rank()) {
        std::fill(buffer.begin(), buffer.end(), value_type{ 0 });
    }

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}
} // namespace MPIScans

} // namespace mpiPP
