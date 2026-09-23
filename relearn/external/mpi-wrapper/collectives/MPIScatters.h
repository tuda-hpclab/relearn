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
#include <cpp-utility/data/displacement.hpp>

#include <mpi.h>

#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace mpiPP {

namespace MPIScatters {

/**
 * @brief Scatters the provided values from the sender to all ranks, one value per rank.
 *      Before the call:
 *          the sender has values = [a, b, c, ...]
 *      After the call:
 *          rank k has <return> = values[k]
 * @tparam T The type of data to scatter
 * @tparam extent The extent of the input span, is usually deduced automatically
 * @param values The values to scatter, one per rank. Can be of any size on the non-sender ranks
 * @param sender The rank which scatters the values
 * @param communicator The communicator to scatter within, defaults to the world communicator
 * @exception Throws an Exception if sender is too large, if values.size() is not equal to the
 *      number of ranks (checked only on the sender rank), or mpi returns an error code
 * @return The value destined for the current rank
 */
template <MPICompatible T, std::size_t extent>
T scatter(const std::span<const T, extent> values, const MPIRank sender, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();
    const auto sender_rank = sender.get_rank();
    utility::Exception::check(sender_rank < number_ranks, "MPIScatters::scatter: There are {} ranks, but should scatter from {}", number_ranks, sender_rank);

    if (my_rank == sender) {
        const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
        utility::Exception::check(number_ranks_cast == values.size(), "MPIScatters::scatter: Sizes do not match: {} vs {}", number_ranks_cast, values.size());
    }

    auto buffer = T{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Scatter(values.data(), 1, type, &buffer, 1, type, sender_rank, communicator.get());
    utility::Exception::check(error_code == 0, "Scattering one value per rank returned the error: {}", error_code);

    if (my_rank == sender) {
        MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(number_ranks) * sizeof(T));
    }
    MPICounters::add_to_received(sizeof(T));

    return buffer;
}

/**
 * @brief Scatters the provided values from the sender to all ranks, a fixed number of values per rank.
 *      Before the call:
 *          the sender has values = [data for rank 0, data for rank 1, ...], count values per rank
 *      After the call:
 *          rank k has <return> = values[k * count, ..., (k + 1) * count - 1]
 * @tparam T The type of data to scatter
 * @tparam extent The extent of the input span, is usually deduced automatically
 * @param values The values to scatter, count consecutive values per rank. Can be of any size on the non-sender ranks
 * @param count The number of values each rank receives, must be the same on every rank
 * @param sender The rank which scatters the values
 * @param communicator The communicator to scatter within, defaults to the world communicator
 * @exception Throws an Exception if sender is too large, if count is negative, if values.size() is not equal to
 *      count times the number of ranks (checked only on the sender rank), or mpi returns an error code
 * @return A vector with the count values destined for the current rank
 */
template <MPICompatible T, std::size_t extent>
[[nodiscard]] std::vector<T> scatter(const std::span<const T, extent> values, const std::integral auto count, const MPIRank sender, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();
    const auto sender_rank = sender.get_rank();
    utility::Exception::check(sender_rank < number_ranks, "MPIScatters::scatter: There are {} ranks, but should scatter from {}", number_ranks, sender_rank);

    // throws for negative counts, on every rank alike
    const auto count_cast = utility::safe_cast<std::size_t>(count);

    if (my_rank == sender) {
        const auto expected_size = count_cast * utility::safe_cast<std::size_t>(number_ranks);
        utility::Exception::check(expected_size == values.size(), "MPIScatters::scatter: Sizes do not match: {} vs {}", expected_size, values.size());
    }

    auto buffer = std::vector<T>(count_cast);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Scatter(values.data(), utility::safe_cast<int>(count), type, buffer.data(), utility::safe_cast<int>(count), type, sender_rank, communicator.get());
    utility::Exception::check(error_code == 0, "Scattering a fixed number of values per rank returned the error: {}", error_code);

    if (my_rank == sender) {
        MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(values.size()) * sizeof(T));
    }
    MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(count_cast) * sizeof(T));

    return buffer;
}

/**
 * @brief Scatters the provided values from the sender to all ranks, a variable number of values per rank.
 *      Calculates the displacements automatically, the values must be tightly packed in order of ranks.
 *      Before the call:
 *          the sender has values = [data for rank 0, data for rank 1, ...], counts[k] values for rank k
 *      After the call:
 *          rank k has <return> = the counts[k] values destined for it
 * @tparam T The type of data to scatter
 * @tparam data_extent The extent of the input span, is usually deduced automatically
 * @tparam count_extent The extent of the counts span, is usually deduced automatically
 * @param values The values to scatter, counts[k] consecutive values for rank k. Can be of any size on the non-sender ranks
 * @param counts The number of values each rank receives. Can be of any size on the non-sender ranks
 * @param sender The rank which scatters the values
 * @param communicator The communicator to scatter within, defaults to the world communicator
 * @exception Throws an Exception if sender is too large, if counts.size() is not equal to the number of ranks,
 *      if any count is negative, if the counts do not sum up to values.size() (the latter three are checked
 *      only on the sender rank), or mpi returns an error code
 * @return A vector with the values destined for the current rank
 */
template <MPICompatible T, std::size_t data_extent, std::size_t count_extent>
[[nodiscard]] std::vector<T> scatter(const std::span<const T, data_extent> values, const std::span<const int, count_extent> counts, const MPIRank sender, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();

    auto send_displacements = std::vector<int>{};
    auto total_send = std::size_t{ 0 };

    if (my_rank == sender) {
        const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
        utility::Exception::check(number_ranks_cast == counts.size(), "MPIScatters::scatter: Sizes do not match: {} vs {}", number_ranks_cast, counts.size());

        for (const auto count : counts) {
            // throws for negative counts
            total_send += utility::safe_cast<std::size_t>(count);
        }
        utility::Exception::check(total_send == values.size(), "MPIScatters::scatter: The counts sum up to {}, but {} values were provided", total_send, values.size());

        send_displacements = utility::calculate_displacements<int>(counts);
    }

    // scatter the per-rank counts so each rank learns how much it will receive from the sender
    const auto my_count = scatter(counts, sender, communicator);

    auto buffer = std::vector<T>(utility::safe_cast<std::size_t>(my_count));

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Scatterv(values.data(), counts.data(), send_displacements.data(), type,
                                         buffer.data(), my_count, type, sender.get_rank(), communicator.get());
    utility::Exception::check(error_code == 0, "Scattering multiple values returned the error: {}", error_code);

    if (my_rank == sender) {
        MPICounters::add_to_sent(utility::safe_cast<std::uint64_t>(total_send) * sizeof(T));
    }
    MPICounters::add_to_received(utility::safe_cast<std::uint64_t>(my_count) * sizeof(T));

    return buffer;
}

/**
 * @brief Scatters the provided values from the sender to all ranks, a variable number of values per rank.
 *      Calculates the counts and the displacements automatically.
 *      Before the call:
 *          the sender has values = [[data for rank 0], [data for rank 1], ...]
 *      After the call:
 *          rank k has <return> = [data for rank k]
 * @tparam T The type of data to scatter
 * @param values The values to scatter, values[k] holds the data destined for rank k.
 *      Can be of any size on the non-sender ranks
 * @param sender The rank which scatters the values
 * @param communicator The communicator to scatter within, defaults to the world communicator
 * @exception Throws an Exception if sender is too large, if values.size() is not equal to the
 *      number of ranks (checked only on the sender rank), or mpi returns an error code
 * @return A vector with the values destined for the current rank
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> scatter(const std::vector<std::vector<T>>& values, const MPIRank sender, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto my_rank = communicator.get_my_rank();
    const auto number_ranks = communicator.get_number_ranks();

    auto send_counts = std::vector<int>{};
    auto flattened_send = std::vector<T>{};

    if (my_rank == sender) {
        const auto number_ranks_cast = utility::safe_cast<std::size_t>(number_ranks);
        utility::Exception::check(number_ranks_cast == values.size(), "MPIScatters::scatter: Sizes do not match: {} vs {}", number_ranks_cast, values.size());

        send_counts.reserve(values.size());
        for (const auto& part : values) {
            send_counts.emplace_back(utility::safe_cast<int>(part.size()));
        }

        auto total_send = std::size_t{ 0 };
        for (const auto& part : values) {
            total_send += part.size();
        }

        flattened_send.reserve(total_send);
        for (const auto& part : values) {
            flattened_send.insert(flattened_send.end(), part.begin(), part.end());
        }
    }

    return scatter(std::span<const T>{ flattened_send }, std::span<const int>{ send_counts }, sender, communicator);
}

} // namespace MPIScatters

} // namespace mpiPP
