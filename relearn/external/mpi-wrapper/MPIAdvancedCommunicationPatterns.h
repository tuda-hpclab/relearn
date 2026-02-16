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

#include "mpi-wrapper/CommunicationMap.h"
#include "mpi-wrapper/CommunicationVector.h"
#include "mpi-wrapper/MPICollectives.h"
#include "mpi-wrapper/MPICommunication.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"
#include "mpi-wrapper/MPISynchronization.h"
#include "mpi-wrapper/MPITypes.h"

#include "cpp-utility/Exception.hpp"

#include <mpi.h>
#include <range/v3/view/filter.hpp>

#include <span>
#include <unordered_set>
#include <vector>

namespace mpiPP {

namespace MPIAdvancedCommunicationPatterns {
/**
 * @brief Exchanges values between all specified mpi ranks.
 *      Before the call:
 *          rank k has src[i] = v
 *      After the call:
 *          rank i has <ret>[k] = v
 * @tparam T The type of data to exchange
 * @param src The elements to exchange
 * @param in_ranks The ranks that will send data
 * @param out_ranks The ranks that the current rank will send data to
 * @exception Throws an Exception if src.size() is not equal to the number of ranks,
 *      if in_ranks or out_ranks contain a non-valid mpi ranks,
 *      or mpi returns an error code
 * @return The exchanged values. Is T{} at places not specified by in_ranks.
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> all_to_all_sparse(const std::span<const T> src,
                                               const std::unordered_set<MPIRank>& in_ranks, const std::unordered_set<MPIRank>& out_ranks) {
    const auto number_ranks = MPIInfo::get_number_ranks();
    const auto number_ranks_cast = utility::save_cast<std::size_t>(number_ranks);
    utility::Exception::check(src.size() == number_ranks_cast, "all_to_all_sparse: Got {} values but {} mpi ranks.", src.size(), number_ranks_cast);

    const auto number_ranks_sending_to_me = in_ranks.size();
    const auto number_ranks_i_send_to = out_ranks.size();

    auto count_tokens = std::vector<MPI_Request>{};
    count_tokens.reserve(number_ranks_sending_to_me + number_ranks_i_send_to);

    auto incoming_values = std::vector<T>(number_ranks_cast, T{});

    // The order (check sends) -> (start sends) -> (check receives) -> (send receives)
    // produces heap corruption if a check of the receiving ranks fails

    for (const auto& sending_rank : in_ranks) {
        const auto actual_rank = sending_rank.get_rank();
        utility::Exception::check(actual_rank < number_ranks, "all_to_all_sparse: actual rank is too large: {} vs {}", actual_rank, number_ranks);
    }

    for (const auto& receiving_rank : out_ranks) {
        const auto actual_rank = receiving_rank.get_rank();
        utility::Exception::check(actual_rank < number_ranks, "all_to_all_sparse: actual rank is too large: {} vs {}", actual_rank, number_ranks);
    }

    for (const auto& sending_rank : in_ranks) {
        const auto s_rank = utility::save_cast<std::size_t>(sending_rank.get_rank());
        const auto receive_token = MPICommunication::async_receive(std::span<T>{ &incoming_values[s_rank], 1 }, sending_rank);
        count_tokens.emplace_back(receive_token);
    }

    for (const auto& receiving_rank : out_ranks) {
        const auto r_rank = utility::save_cast<std::size_t>(receiving_rank.get_rank());
        const auto send_token = MPICommunication::async_send(std::span<const T>{ &src[r_rank], 1 }, receiving_rank);
        count_tokens.emplace_back(send_token);
    }

    MPISynchronization::wait_all(count_tokens);

    return incoming_values;
}

/**
 * @brief Exchanges values between all specified mpi ranks.
 *      Before the call:
 *          rank k has src[i] = v
 *      After the call:
 *          rank i has tgt[k] = v
 * @tparam T The type of data to exchange
 * @param src The elements to exchange
 * @param tgt The range where to store the received values. Places not specified by in_ranks are not altered
 * @param in_ranks The ranks that will send data
 * @param out_ranks The ranks that the current rank will send data to
 * @exception Throws an Exception if src.size() is not equal to the number of ranks,
 *      if in_ranks or out_ranks contain a non-valid mpi ranks,
 *      or mpi returns an error code
 */
template <MPICompatible T>
void all_to_all_sparse(const std::span<const T> src, const std::span<T> tgt,
                       const std::unordered_set<MPIRank>& in_ranks, const std::unordered_set<MPIRank>& out_ranks) {
    const auto number_ranks = MPIInfo::get_number_ranks();
    const auto number_ranks_cast = utility::save_cast<std::size_t>(number_ranks);
    utility::Exception::check(src.size() == number_ranks_cast, "all_to_all_sparse: Got {} values but {} mpi ranks.", src.size(), number_ranks_cast);
    utility::Exception::check(tgt.size() == number_ranks_cast, "all_to_all_sparse: Got {} values but {} mpi ranks.", tgt.size(), number_ranks_cast);

    const auto number_ranks_sending_to_me = in_ranks.size();
    const auto number_ranks_i_send_to = out_ranks.size();

    auto count_tokens = std::vector<MPI_Request>{};
    count_tokens.reserve(number_ranks_sending_to_me + number_ranks_i_send_to);

    // The order (check sends) -> (start sends) -> (check receives) -> (send receives)
    // produces heap corruption if a check of the receiving ranks fails

    for (const auto& sending_rank : in_ranks) {
        const auto actual_rank = sending_rank.get_rank();
        utility::Exception::check(actual_rank < number_ranks, "all_to_all_sparse: actual rank is too large: {} vs {}", actual_rank, number_ranks);
    }

    for (const auto& receiving_rank : out_ranks) {
        const auto actual_rank = receiving_rank.get_rank();
        utility::Exception::check(actual_rank < number_ranks, "all_to_all_sparse: actual rank is too large: {} vs {}", actual_rank, number_ranks);
    }

    for (const auto& sending_rank : in_ranks) {
        const auto s_rank = utility::save_cast<std::size_t>(sending_rank.get_rank());
        const auto receive_token = MPICommunication::async_receive(std::span<T>{ &tgt[s_rank], 1 }, sending_rank);
        count_tokens.emplace_back(receive_token);
    }

    for (const auto& receiving_rank : out_ranks) {
        const auto r_rank = utility::save_cast<std::size_t>(receiving_rank.get_rank());
        const auto send_token = MPICommunication::async_send(std::span<const T>{ &src[r_rank], 1 }, receiving_rank);
        count_tokens.emplace_back(send_token);
    }

    MPISynchronization::wait_all(count_tokens);
}

/**
 * @brief Exchanges data with all MPI ranks
 * @tparam RequestType The type that should be exchanged
 * @param outgoing_requests The values that should be exchanged. values[i] should be send to MPI rank i (if present)
 * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
 */
template <MPICompatible RequestType>
[[nodiscard]] CommunicationMap<RequestType> exchange_requests(const CommunicationMap<RequestType>& outgoing_requests) {
    using size_type = typename std::vector<RequestType>::size_type;

    const auto number_ranks = MPIInfo::get_number_ranks();

    utility::Exception::check(number_ranks == outgoing_requests.get_number_ranks(),
                              "MPIAdvancedCommunicationPatterns::exchange_requests: There are {} mpi ranks, but the communication map reports {}", number_ranks, outgoing_requests.get_number_ranks());

    const auto& number_requests_outgoing = outgoing_requests.get_request_sizes_vector();
    const auto& number_requests_incoming = MPICollectives::all_to_all<size_type>(number_requests_outgoing);

    const auto size_hint = outgoing_requests.size();
    auto incoming_requests = CommunicationMap<RequestType>(number_ranks, size_hint);
    incoming_requests.resize(number_requests_incoming);

    auto async_tokens = std::vector<MPI_Request>{};

    for (const auto rank : MPIRank::range(number_ranks) | ranges::views::filter([&incoming_requests](const auto& r) { return incoming_requests.contains(r); })) {
        const auto token = MPICommunication::async_receive<RequestType>(incoming_requests.get_span(rank), rank);
        async_tokens.emplace_back(token);
    }

    for (const auto rank : MPIRank::range(number_ranks) | ranges::views::filter([&outgoing_requests](const auto& r) { return outgoing_requests.contains(r); })) {
        const auto token = MPICommunication::async_send<RequestType>(outgoing_requests.get_span(rank), rank);
        async_tokens.emplace_back(token);
    }

    MPISynchronization::wait_all(async_tokens);

    return incoming_requests;
}

/**
 * @brief Exchanges data with all MPI ranks
 * @tparam RequestType The type that should be exchanged
 * @param outgoing_requests The values that should be exchanged. values[i] should be send to MPI rank i (if present)
 * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
 */
template <MPICompatible RequestType>
[[nodiscard]] CommunicationVector<RequestType> exchange_requests(const CommunicationVector<RequestType>& outgoing_requests) {
    using size_type = typename std::vector<RequestType>::size_type;

    const auto number_ranks = MPIInfo::get_number_ranks();

    utility::Exception::check(number_ranks == outgoing_requests.get_number_ranks(),
                              "MPIAdvancedCommunicationPatterns::exchange_requests: There are {} mpi ranks, but the communication map reports {}", number_ranks, outgoing_requests.get_number_ranks());

    const auto& number_requests_outgoing = outgoing_requests.get_request_sizes_vector();
    const auto& number_requests_incoming = MPICollectives::all_to_all<size_type>(number_requests_outgoing);

    auto incoming_requests = CommunicationVector<RequestType>(number_ranks);
    incoming_requests.resize(number_requests_incoming);

    auto async_tokens = std::vector<MPI_Request>{};

    for (const auto rank : MPIRank::range(number_ranks) | ranges::views::filter([&incoming_requests](const auto& r) { return incoming_requests.contains(r); })) {
        const auto token = MPICommunication::async_receive<RequestType>(incoming_requests.get_span(rank), rank);
        async_tokens.emplace_back(token);
    }

    for (const auto rank : MPIRank::range(number_ranks) | ranges::views::filter([&outgoing_requests](const auto& r) { return outgoing_requests.contains(r); })) {
        const auto token = MPICommunication::async_send<RequestType>(outgoing_requests.get_span(rank), rank);
        async_tokens.emplace_back(token);
    }

    MPISynchronization::wait_all(async_tokens);

    return incoming_requests;
}

/**
 * @brief Exchanges data with the specified MPI ranks
 * @tparam RequestType The type that should be exchanged
 * @param outgoing_requests The values that should be exchanged. values[i] should be send to MPI rank i (if present)
 * @param in_ranks The ranks that will send data
 * @param out_ranks The ranks that the current rank will send data to
 * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
 */
template <MPICompatible RequestType>
[[nodiscard]] CommunicationVector<RequestType> exchange_requests(const CommunicationVector<RequestType>& outgoing_requests,
                                                              const std::unordered_set<MPIRank>& in_ranks, const std::unordered_set<MPIRank>& out_ranks) {
    using size_type = typename std::vector<RequestType>::size_type;

    const auto number_ranks = MPIInfo::get_number_ranks();

    utility::Exception::check(number_ranks == outgoing_requests.get_number_ranks(),
                              "MPIAdvancedCommunicationPatterns::exchange_requests: There are {} mpi ranks, but the communication map reports {}", number_ranks, outgoing_requests.get_number_ranks());

    const auto& number_requests_outgoing = outgoing_requests.get_request_sizes_vector();
    const auto& number_requests_incoming = all_to_all_sparse<size_type>(number_requests_outgoing, in_ranks, out_ranks);

    auto incoming_requests = CommunicationVector<RequestType>(number_ranks);
    incoming_requests.resize(number_requests_incoming);

    auto async_tokens = std::vector<MPI_Request>{};

    // in_ranks and out_ranks are checked by all_to_all_sparse for validity
    // so they do not need to be checked again

    for (const auto& rank : in_ranks) {
        if (!incoming_requests.contains(rank)) {
            continue;
        }

        const auto token = MPICommunication::async_receive<RequestType>(incoming_requests.get_span(rank), rank);
        async_tokens.emplace_back(token);
    }

    for (const auto& rank : out_ranks) {
        if (!outgoing_requests.contains(rank)) {
            continue;
        }

        const auto token = MPICommunication::async_send<RequestType>(outgoing_requests.get_span(rank), rank);
        async_tokens.emplace_back(token);
    }

    MPISynchronization::wait_all(async_tokens);

    return incoming_requests;
}

/**
 * @brief Exchanges data with the specified MPI ranks
 * @tparam RequestType The type that should be exchanged
 * @param outgoing_requests The values that should be exchanged. values[i] should be send to MPI rank i (if present)
 * @param in_ranks The ranks that will send data
 * @param out_ranks The ranks that the current rank will send data to
 * @return The values that were received from the MPI ranks. <return>[i] on rank j was values[j] on rank i
 */
template <MPICompatible RequestType>
[[nodiscard]] CommunicationMap<RequestType> exchange_requests(const CommunicationMap<RequestType>& outgoing_requests,
                                                              const std::unordered_set<MPIRank>& in_ranks, const std::unordered_set<MPIRank>& out_ranks) {
    using size_type = typename std::vector<RequestType>::size_type;

    const auto number_ranks = MPIInfo::get_number_ranks();

    utility::Exception::check(number_ranks == outgoing_requests.get_number_ranks(),
                              "MPIAdvancedCommunicationPatterns::exchange_requests: There are {} mpi ranks, but the communication map reports {}", number_ranks, outgoing_requests.get_number_ranks());

    const auto& number_requests_outgoing = outgoing_requests.get_request_sizes_vector();
    const auto& number_requests_incoming = all_to_all_sparse<size_type>(number_requests_outgoing, in_ranks, out_ranks);

    const auto size_hint = outgoing_requests.size();
    auto incoming_requests = CommunicationMap<RequestType>(number_ranks, size_hint);
    incoming_requests.resize(number_requests_incoming);

    auto async_tokens = std::vector<MPI_Request>{};

    // in_ranks and out_ranks are checked by all_to_all_sparse for validity
    // so they do not need to be checked again

    for (const auto& rank : in_ranks) {
        if (!incoming_requests.contains(rank)) {
            continue;
        }

        const auto token = MPICommunication::async_receive<RequestType>(incoming_requests.get_span(rank), rank);
        async_tokens.emplace_back(token);
    }

    for (const auto& rank : out_ranks) {
        if (!outgoing_requests.contains(rank)) {
            continue;
        }

        const auto token = MPICommunication::async_send<RequestType>(outgoing_requests.get_span(rank), rank);
        async_tokens.emplace_back(token);
    }

    MPISynchronization::wait_all(async_tokens);

    return incoming_requests;
}
} // namespace MPIAdvancedCommunicationPatterns

} // namespace mpiPP
