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

#include <concepts>
#include <vector>

namespace mpiPP {

namespace MPIReductions {
/**
 * @brief Reduces the provided values and returns the componentwise sum on the root rank
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise sum of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container reduce_componentwise_sum(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Reducing (sum) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_SUM, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (sum) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise sum on all ranks
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise sum of all local values on all ranks
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container all_reduce_componentwise_sum(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "All-reducing (sum) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_SUM, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (sum) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise maximum on the root rank
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise maximum of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container reduce_componentwise_max(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Reducing (max) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_MAX, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (max) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise maximum on all ranks
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise maximum of all local values on all ranks
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container all_reduce_componentwise_max(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "All-reducing (max) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_MAX, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (max) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise minimum on the root rank
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise minimum of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container reduce_componentwise_min(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Reducing (min) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_MIN, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (min) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise minimum on all ranks
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise minimum of all local values on all ranks
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container all_reduce_componentwise_min(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "All-reducing (min) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_MIN, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (min) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise product on the root rank
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise product of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container reduce_componentwise_prod(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Reducing (prod) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 1 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_PROD, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (prod) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise product on all ranks
 * @tparam container A range of values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise product of all local values on all ranks
 */
template <MPIReductionArithmeticRange container>
[[nodiscard]] container all_reduce_componentwise_prod(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "All-reducing (prod) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 1 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_PROD, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (prod) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise bitwise and on the root rank
 * @tparam container A range of integral values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise bitwise and of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPIReductionArithmeticRange container>
    requires std::integral<typename container::value_type>
[[nodiscard]] container reduce_componentwise_bitwise_and(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Reducing (bitwise and) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_BAND, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (bitwise and) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise bitwise and on all ranks
 * @tparam container A range of integral values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise bitwise and of all local values on all ranks
 */
template <MPIReductionArithmeticRange container>
    requires std::integral<typename container::value_type>
[[nodiscard]] container all_reduce_componentwise_bitwise_and(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "All-reducing (bitwise and) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_BAND, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (bitwise and) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise bitwise or on the root rank
 * @tparam container A range of integral values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise bitwise or of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPIReductionArithmeticRange container>
    requires std::integral<typename container::value_type>
[[nodiscard]] container reduce_componentwise_bitwise_or(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Reducing (bitwise or) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_BOR, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (bitwise or) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise bitwise or on all ranks
 * @tparam container A range of integral values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise bitwise or of all local values on all ranks
 */
template <MPIReductionArithmeticRange container>
    requires std::integral<typename container::value_type>
[[nodiscard]] container all_reduce_componentwise_bitwise_or(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "All-reducing (bitwise or) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_BOR, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (bitwise or) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise bitwise exclusive or on the root rank
 * @tparam container A range of integral values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise bitwise exclusive or of all local values on the root rank, nothing of interest on all other ranks
 */
template <MPIReductionArithmeticRange container>
    requires std::integral<typename container::value_type>
[[nodiscard]] container reduce_componentwise_bitwise_xor(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Reducing (bitwise xor) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Reduce(values.data(), buffer.data(), length, type, MPI_BXOR, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (bitwise xor) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(value_type) * number_values);
    }

    return buffer;
}

/**
 * @brief Reduces the provided values and returns the componentwise bitwise exclusive or on all ranks
 * @tparam container A range of integral values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an Exception if the number of elements differs between MPI ranks or MPI reports an error
 * @return The componentwise bitwise exclusive or of all local values on all ranks
 */
template <MPIReductionArithmeticRange container>
    requires std::integral<typename container::value_type>
[[nodiscard]] container all_reduce_componentwise_bitwise_xor(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "All-reducing (bitwise xor) componentwise with differently sized vectors!");

    auto buffer = container{};
    if constexpr (detail::is_std_vector_v<container>) {
        buffer.resize(number_values, value_type{ 0 });
    }

    const auto length = utility::safe_cast<int>(number_values);
    const auto type = MPITypes::convert_type_to_mpi_type<value_type>();
    const auto error_code = MPI_Allreduce(values.data(), buffer.data(), length, type, MPI_BXOR, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (bitwise xor) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(value_type) * number_values);
    MPICounters::add_to_received(sizeof(value_type) * number_values);

    return buffer;
}

/**
 * @brief Reduces the provided values componentwise and returns, for each component, the maximum and
 *      the rank holding it on all ranks
 * @tparam container A range of locatable values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi reports an error
 * @return For each component the maximum of all local values and the (smallest) rank that provided it
 */
template <MPIReductionArithmeticRange container>
    requires MPILocatable<typename container::value_type>
[[nodiscard]] std::vector<ValueLocation<typename container::value_type>> all_reduce_componentwise_max_location(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "All-reducing (maxloc) componentwise with differently sized vectors!");

    const auto my_rank = communicator.get_my_rank().get_rank();
    auto input = std::vector<detail::value_location_pair<value_type>>(number_values);
    for (auto i = std::size_t{ 0 }; i < number_values; i++) {
        input[i] = detail::value_location_pair<value_type>{ values.data()[i], my_rank };
    }
    auto output = std::vector<detail::value_location_pair<value_type>>(number_values);

    const auto length = utility::safe_cast<int>(number_values);
    const auto error_code = MPI_Allreduce(input.data(), output.data(), length, value_location_type<value_type>(), MPI_MAXLOC, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (maxloc) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(detail::value_location_pair<value_type>) * number_values);
    MPICounters::add_to_received(sizeof(detail::value_location_pair<value_type>) * number_values);

    auto result = std::vector<ValueLocation<value_type>>(number_values);
    for (auto i = std::size_t{ 0 }; i < number_values; i++) {
        result[i] = ValueLocation<value_type>{ output[i].value, MPIRank{ output[i].location } };
    }
    return result;
}

/**
 * @brief Reduces the provided values componentwise and returns, for each component, the minimum and
 *      the rank holding it on all ranks
 * @tparam container A range of locatable values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi reports an error
 * @return For each component the minimum of all local values and the (smallest) rank that provided it
 */
template <MPIReductionArithmeticRange container>
    requires MPILocatable<typename container::value_type>
[[nodiscard]] std::vector<ValueLocation<typename container::value_type>> all_reduce_componentwise_min_location(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "All-reducing (minloc) componentwise with differently sized vectors!");

    const auto my_rank = communicator.get_my_rank().get_rank();
    auto input = std::vector<detail::value_location_pair<value_type>>(number_values);
    for (auto i = std::size_t{ 0 }; i < number_values; i++) {
        input[i] = detail::value_location_pair<value_type>{ values.data()[i], my_rank };
    }
    auto output = std::vector<detail::value_location_pair<value_type>>(number_values);

    const auto length = utility::safe_cast<int>(number_values);
    const auto error_code = MPI_Allreduce(input.data(), output.data(), length, value_location_type<value_type>(), MPI_MINLOC, communicator.get());
    utility::Exception::check(error_code == 0, "All-reducing (minloc) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(detail::value_location_pair<value_type>) * number_values);
    MPICounters::add_to_received(sizeof(detail::value_location_pair<value_type>) * number_values);

    auto result = std::vector<ValueLocation<value_type>>(number_values);
    for (auto i = std::size_t{ 0 }; i < number_values; i++) {
        result[i] = ValueLocation<value_type>{ output[i].value, MPIRank{ output[i].location } };
    }
    return result;
}

/**
 * @brief Reduces the provided values componentwise and returns, for each component, the maximum and
 *      the rank holding it on the root rank
 * @tparam container A range of locatable values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi reports an error
 * @return For each component the maximum and its (smallest) rank on the root rank, nothing of interest on all other ranks
 */
template <MPIReductionArithmeticRange container>
    requires MPILocatable<typename container::value_type>
[[nodiscard]] std::vector<ValueLocation<typename container::value_type>> reduce_componentwise_max_location(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Reducing (maxloc) componentwise with differently sized vectors!");

    const auto my_rank = communicator.get_my_rank().get_rank();
    auto input = std::vector<detail::value_location_pair<value_type>>(number_values);
    for (auto i = std::size_t{ 0 }; i < number_values; i++) {
        input[i] = detail::value_location_pair<value_type>{ values.data()[i], my_rank };
    }
    auto output = std::vector<detail::value_location_pair<value_type>>(number_values);

    const auto length = utility::safe_cast<int>(number_values);
    const auto error_code = MPI_Reduce(input.data(), output.data(), length, value_location_type<value_type>(), MPI_MAXLOC, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (maxloc) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(detail::value_location_pair<value_type>) * number_values);
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(detail::value_location_pair<value_type>) * number_values);
    }

    auto result = std::vector<ValueLocation<value_type>>(number_values);
    for (auto i = std::size_t{ 0 }; i < number_values; i++) {
        result[i] = ValueLocation<value_type>{ output[i].value, MPIRank{ output[i].location } };
    }
    return result;
}

/**
 * @brief Reduces the provided values componentwise and returns, for each component, the minimum and
 *      the rank holding it on the root rank
 * @tparam container A range of locatable values, for example std::array or std::vector
 * @param values The values to reduce
 * @param communicator The communicator to reduce within, defaults to the world communicator
 * @exception Throws an exception if the number of elements differ on mpi ranks or mpi reports an error
 * @return For each component the minimum and its (smallest) rank on the root rank, nothing of interest on all other ranks
 */
template <MPIReductionArithmeticRange container>
    requires MPILocatable<typename container::value_type>
[[nodiscard]] std::vector<ValueLocation<typename container::value_type>> reduce_componentwise_min_location(const container& values, const MPICommunicator& communicator = MPICommunicator::World) {
    using value_type = typename container::value_type;

    const auto number_values = values.size();
    const auto minimum_number_values = all_reduce_min(number_values, communicator);
    const auto maximum_number_values = all_reduce_max(number_values, communicator);
    utility::Exception::check(minimum_number_values == maximum_number_values, "Reducing (minloc) componentwise with differently sized vectors!");

    const auto my_rank = communicator.get_my_rank().get_rank();
    auto input = std::vector<detail::value_location_pair<value_type>>(number_values);
    for (auto i = std::size_t{ 0 }; i < number_values; i++) {
        input[i] = detail::value_location_pair<value_type>{ values.data()[i], my_rank };
    }
    auto output = std::vector<detail::value_location_pair<value_type>>(number_values);

    const auto length = utility::safe_cast<int>(number_values);
    const auto error_code = MPI_Reduce(input.data(), output.data(), length, value_location_type<value_type>(), MPI_MINLOC, 0, communicator.get());
    utility::Exception::check(error_code == 0, "Reducing (minloc) componentwise returned the error: {}", error_code);

    MPICounters::add_to_sent(sizeof(detail::value_location_pair<value_type>) * number_values);
    if (communicator.is_root_rank()) {
        MPICounters::add_to_received(utility::safe_cast<std::size_t>(communicator.get_number_ranks()) * sizeof(detail::value_location_pair<value_type>) * number_values);
    }

    auto result = std::vector<ValueLocation<value_type>>(number_values);
    for (auto i = std::size_t{ 0 }; i < number_values; i++) {
        result[i] = ValueLocation<value_type>{ output[i].value, MPIRank{ output[i].location } };
    }
    return result;
}
} // namespace MPIReductions

} // namespace mpiPP
