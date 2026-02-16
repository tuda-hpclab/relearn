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

#include "mpi-wrapper/MPICounters.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPITypes.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <mpi.h>

namespace mpiPP {

namespace MPIReductions {
/**
 * @brief Reduces the provided values and returns the sum on all ranks
 * @tparam T The type of data to reduce
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The sum of all local values
 */
template <MPICompatible T>
[[nodiscard]] T all_reduce_sum(const T value) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_SUM, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the minimum on all ranks
 * @tparam T The type of data to reduce
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The minimum of all local values
 */
template <MPICompatible T>
[[nodiscard]] T all_reduce_min(const T value) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_MIN, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the maximum on all ranks
 * @tparam T The type of data to reduce
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The maximum of all local values
 */
template <MPICompatible T>
[[nodiscard]] T all_reduce_max(const T value) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Allreduce(&value, &total_value, 1, type, MPI_MAX, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "All-reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    MPICounters::add_to_received(sizeof(T));

    return total_value;
}

/**
 * @brief Reduces the provided values and returns logical or
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The logical or of all local values
 */
[[nodiscard]] inline bool all_reduce_or(const bool value) {
    const auto val = value ? std::uint8_t{ 1 } : std::uint8_t{ 0 };
    const auto total_value = all_reduce_max(val);
    return total_value == 1;
}

/**
 * @brief Reduces the provided values and returns logical and
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The logical and of all local values
 */
[[nodiscard]] inline bool all_reduce_and(const bool value) {
    const auto val = value ? std::uint8_t{ 1 } : std::uint8_t{ 0 };
    const auto total_value = all_reduce_min(val);
    return total_value == 1;
}

/**
 * @brief Reduces the provided values and returns the sum on the root rank
 * @tparam T The type of data to reduce
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The sum of all local values on the root rank, T(0) on all other ranks
 */
template <MPICompatible T>
[[nodiscard]] T reduce_sum(const T value) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_SUM, 0, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (MPIInfo::is_root_rank()) {
        MPICounters::add_to_received(utility::save_cast<std::size_t>(MPIInfo::get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the minimum on the root rank
 * @tparam T The type of data to reduce
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The minimum of all local values on the root rank, T(0) on all other ranks
 */
template <MPICompatible T>
[[nodiscard]] T reduce_min(const T value) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_MIN, 0, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (MPIInfo::is_root_rank()) {
        MPICounters::add_to_received(utility::save_cast<std::size_t>(MPIInfo::get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns the maximum on the root rank
 * @tparam T The type of data to reduce
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The maximum of all local values on the root rank, T(0) on all other ranks
 */
template <MPICompatible T>
[[nodiscard]] T reduce_max(const T value) {
    auto total_value = T{ 0 };

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Reduce(&value, &total_value, 1, type, MPI_MAX, 0, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "Reducing all values returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(T));
    if (MPIInfo::is_root_rank()) {
        MPICounters::add_to_received(utility::save_cast<std::size_t>(MPIInfo::get_number_ranks()) * sizeof(T));
    }

    return total_value;
}

/**
 * @brief Reduces the provided values and returns logical or on the root rank
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The logical or of all local values on the root rank, value on all other ranks
 */
[[nodiscard]] inline bool reduce_or(const bool value) {
    const auto val = value ? std::uint8_t{ 1 } : std::uint8_t{ 0 };
    const auto total_value = reduce_max(val);
    return total_value == 1;
}

/**
 * @brief Reduces the provided values and returns logical and on the root rank
 * @param value The local value
 * @exception Throws an Exception if mpi reports an error
 * @return The logical and of all local values on the root rank, value on all other ranks
 */
[[nodiscard]] inline bool reduce_and(const bool value) {
    const auto val = value ? std::uint8_t{ 1 } : std::uint8_t{ 0 };
    const auto total_value = reduce_min(val);
    return total_value == 1;
}

/**
 * @brief Reduces the provided values and returns the componentwise sum on the root rank
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi retorts an error
 * @return The componentwise sum of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPICompatibleRange container>
[[nodiscard]] container reduce_componentwise_sum(const container& values) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values);
    utility::Exception::check(number_values == minimum_number_values, "Reducing (sum) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::save_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_SUM, 0, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "Reducing (sum) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (MPIInfo::is_root_rank()) {
        MPICounters::add_to_received(utility::save_cast<std::size_t>(MPIInfo::get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise sum on all ranks
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi retorts an error
 * @return The componentwise sum of all local values on all ranks
 */
template <MPICompatibleRange container>
[[nodiscard]] container all_reduce_componentwise_sum(const container& values) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values);
    utility::Exception::check(number_values == minimum_number_values, "All-reducing (sum) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::save_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_SUM, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "All-reducing (sum) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise maximum on the root rank
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi retorts an error
 * @return The componentwise maximum of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPICompatibleRange container>
[[nodiscard]] container reduce_componentwise_max(const container& values) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values);
    utility::Exception::check(number_values == minimum_number_values, "Reducing (max) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::save_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_MAX, 0, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "Reducing (max) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (MPIInfo::is_root_rank()) {
        MPICounters::add_to_received(utility::save_cast<std::size_t>(MPIInfo::get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise maximum on all ranks
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi retorts an error
 * @return The componentwise maximum of all local values on all ranks
 */
template <MPICompatibleRange container>
[[nodiscard]] container all_reduce_componentwise_max(const container& values) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values);
    utility::Exception::check(number_values == minimum_number_values, "All-reducing (max) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::save_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_MAX, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "All-reducing (max) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise minimum on the root rank
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi retorts an error
 * @return The componentwise minimum of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPICompatibleRange container>
[[nodiscard]] container reduce_componentwise_min(const container& values) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values);
    utility::Exception::check(number_values == minimum_number_values, "Reducing (min) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::save_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_MIN, 0, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "Reducing (min) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (MPIInfo::is_root_rank()) {
        MPICounters::add_to_received(utility::save_cast<std::size_t>(MPIInfo::get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise minimum on all ranks
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi retorts an error
 * @return The componentwise minimum of all local values on all ranks
 */
template <MPICompatibleRange container>
[[nodiscard]] container all_reduce_componentwise_min(const container& values) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values);
    utility::Exception::check(number_values == minimum_number_values, "All-reducing (min) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::save_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_MIN, MPI_COMM_WORLD);
    utility::Exception::check(error_code == 0, "All-reducing (min) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}
} // namespace MPIReductions

} // namespace mpiPP
