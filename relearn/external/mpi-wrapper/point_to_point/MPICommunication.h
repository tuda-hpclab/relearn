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

#include <optional>
#include <span>
#include <vector>

namespace mpiPP {

namespace MPICommunication {
/**
 * @brief Sends the buffer to the specified rank. Must have a matching call on the other rank
 * @tparam T The type of data to send
 * @param buffer The data to send
 * @param target Where to send the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if target is not in the communicator or MPI reports an error
 */
template <MPICompatible T>
void send(const std::span<const T> buffer, const MPIRank target, const int tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto target_rank = target.get_rank();
    utility::Exception::check(target_rank < number_ranks, "MPICommunication::send: There are {} ranks, but should send to {}", number_ranks, target_rank);

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::safe_cast<int>(buffer.size());
    const auto error_code = MPI_Send(buffer.data(), size_cast, type, target_rank, tag, communicator.get());
    utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunication::send: Error code is: {}", error_code);

    MPICounters::add_to_sent(buffer.size_bytes());
}

/**
 * @brief Receives data into the buffer from the specified rank. Must have a matching call on the other rank
 * @tparam T The type of data to receive
 * @param buffer The buffer for the data
 * @param source From where to receive the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if source is not in the communicator or MPI reports an error
 */
template <MPICompatible T>
void receive(const std::span<T> buffer, const MPIRank source, const int tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto source_rank = source.get_rank();
    utility::Exception::check(source_rank < number_ranks, "MPICommunication::receive: There are {} ranks, but should receive from {}", number_ranks, source_rank);

    auto status = MPI_Status{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::safe_cast<int>(buffer.size());
    const auto error_code = MPI_Recv(buffer.data(), size_cast, type, source_rank, tag, communicator.get(), &status);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunication::receive: Error code is: {}", error_code);

    MPICounters::add_to_received(buffer.size_bytes());
}

/**
 * @brief Sends the data element to the specified rank. Must have a matching call on the other rank
 * @tparam T The type of data to send
 * @param value The data element to send
 * @param target Where to send the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if target is not in the communicator or MPI reports an error
 */
template <MPICompatible T>
void send(const T value, const MPIRank target, const int tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    send(std::span{ &value, 1 }, target, tag, communicator);
}

/**
 * @brief Receives one data element from the specified rank. Must have a matching call on the other rank
 * @tparam T The type of data to receive
 * @param source From where to receive the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if source is not in the communicator or MPI reports an error
 */
template <MPICompatible T>
[[nodiscard]] T receive(const MPIRank source, const int tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    auto value = T{};
    receive<T>(std::span<T>{ &value, 1 }, source, tag, communicator);
    return value;
}

/**
 * @brief Sends the buffer to the specified rank. Must have a matching call on the other rank.
 *      Returns the control flow without necessarily completing the request
 * @tparam T The type of data to send
 * @param buffer The data to send; its storage must remain valid and unmodified until the request completes
 * @param target Where to send the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if target is not in the communicator or MPI reports an error
 * @return Returns a token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_send(const std::span<const T> buffer, const MPIRank target, const int tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto target_rank = target.get_rank();
    utility::Exception::check(target_rank < number_ranks, "MPICommunication::send: There are {} ranks, but should send to {}", number_ranks, target_rank);

    auto request = MPI_Request{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::safe_cast<int>(buffer.size());
    const auto error_code = MPI_Isend(buffer.data(), size_cast, type, target_rank, tag, communicator.get(), &request);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunication::async_send: Error code is: {}", error_code);

    MPICounters::add_to_sent(buffer.size_bytes());

    return request;
}

/**
 * @brief Receives data into the buffer from the specified rank. Must have a matching call on the other rank.
 *      Returns the control flow without necessarily completing the request
 * @tparam T The type of data to receive
 * @param buffer The receive buffer; its storage must remain valid until the request completes and
 *      must not be read before then
 * @param source From where to receive the data
 * @param tag A tag to distinguish data transfers, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if source is not in the communicator or MPI reports an error
 * @return Returns a token that can be waited on
 */
template <MPICompatible T>
[[nodiscard]] MPI_Request async_receive(const std::span<T> buffer, const MPIRank source, const int tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto source_rank = source.get_rank();
    utility::Exception::check(source_rank < number_ranks, "MPICommunication::receive: There are {} ranks, but should receive from {}", number_ranks, source_rank);

    auto request = MPI_Request{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::safe_cast<int>(buffer.size());
    const auto error_code = MPI_Irecv(buffer.data(), size_cast, type, source_rank, tag, communicator.get(), &request);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunication::async_receive: Error code is: {}", error_code);

    MPICounters::add_to_received(buffer.size_bytes());

    return request;
}

/**
 * @brief Resolves an MPIRank into the raw rank id expected by MPI. The uninitialized rank maps
 *      to MPI_PROC_NULL, turning that half of a send/receive exchange into a no-op. This mirrors
 *      MPI_Sendrecv semantics and lets a shift across a domain boundary (see
 *      MPICartesianCommunicator::shift, which yields the uninitialized rank at the boundary) be
 *      passed straight through without a special case at the call site.
 * @param partner The rank to communicate with
 * @param number_ranks The number of ranks in the communicator
 * @exception Throws an Exception if the rank is initialized but not part of the communicator
 * @return The raw rank id, or MPI_PROC_NULL for the uninitialized rank
 */
[[nodiscard]] inline int resolve_partner_rank(const MPIRank partner, const int number_ranks) {
    if (!partner.is_initialized()) {
        return MPI_PROC_NULL;
    }
    const auto partner_rank = partner.get_rank();
    utility::Exception::check(partner_rank < number_ranks, "MPICommunication::resolve_partner_rank: There are {} ranks, but should communicate with {}", number_ranks, partner_rank);
    return partner_rank;
}

/**
 * @brief Sends the send buffer to target while simultaneously receiving into the receive buffer
 *      from source, in a single deadlock-free operation. The uninitialized rank may be passed for
 *      either partner to skip that half of the exchange (MPI_PROC_NULL).
 * @tparam T The type of data to send and receive
 * @param send_buffer The data to send
 * @param target Where to send the data
 * @param receive_buffer The buffer for the received data
 * @param source From where to receive the data
 * @param send_tag A tag to distinguish the send, default is 0
 * @param receive_tag A tag to distinguish the receive, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if a partner is not in the communicator or MPI reports an error
 */
template <MPICompatible T>
void send_receive(const std::span<const T> send_buffer, const MPIRank target, const std::span<T> receive_buffer, const MPIRank source, const int send_tag = 0, const int receive_tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto target_rank = resolve_partner_rank(target, number_ranks);
    const auto source_rank = resolve_partner_rank(source, number_ranks);

    auto status = MPI_Status{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto send_size_cast = utility::safe_cast<int>(send_buffer.size());
    const auto receive_size_cast = utility::safe_cast<int>(receive_buffer.size());
    const auto error_code = MPI_Sendrecv(send_buffer.data(), send_size_cast, type, target_rank, send_tag, receive_buffer.data(), receive_size_cast, type, source_rank, receive_tag, communicator.get(), &status);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunication::send_receive: Error code is: {}", error_code);

    MPICounters::add_to_sent(send_buffer.size_bytes());
    MPICounters::add_to_received(receive_buffer.size_bytes());
}

/**
 * @brief Sends one data element to target while simultaneously receiving one element from source,
 *      in a single deadlock-free operation. The uninitialized rank may be passed for either partner
 *      to skip that half of the exchange (MPI_PROC_NULL); the received value is unchanged then.
 * @tparam T The type of data to send and receive
 * @param value The data element to send
 * @param target Where to send the data
 * @param source From where to receive the data
 * @param send_tag A tag to distinguish the send, default is 0
 * @param receive_tag A tag to distinguish the receive, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if a partner is not in the communicator or MPI reports an error
 * @return The received data element
 */
template <MPICompatible T>
[[nodiscard]] T send_receive(const T value, const MPIRank target, const MPIRank source, const int send_tag = 0, const int receive_tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    auto received = T{};
    send_receive<T>(std::span<const T>{ &value, 1 }, target, std::span<T>{ &received, 1 }, source, send_tag, receive_tag, communicator);
    return received;
}

/**
 * @brief Sends the buffer to target while simultaneously receiving from source into the same
 *      buffer, in a single deadlock-free operation. The buffer must be large enough to hold the
 *      incoming message. The uninitialized rank may be passed for either partner to skip that half
 *      of the exchange (MPI_PROC_NULL); the buffer is left unchanged for a skipped receive.
 * @tparam T The type of data to send and receive
 * @param buffer The data to send, overwritten with the received data
 * @param target Where to send the data
 * @param source From where to receive the data
 * @param send_tag A tag to distinguish the send, default is 0
 * @param receive_tag A tag to distinguish the receive, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if a partner is not in the communicator or MPI reports an error
 */
template <MPICompatible T>
void send_receive_replace(const std::span<T> buffer, const MPIRank target, const MPIRank source, const int send_tag = 0, const int receive_tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto number_ranks = communicator.get_number_ranks();
    const auto target_rank = resolve_partner_rank(target, number_ranks);
    const auto source_rank = resolve_partner_rank(source, number_ranks);

    auto status = MPI_Status{};

    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto size_cast = utility::safe_cast<int>(buffer.size());
    const auto error_code = MPI_Sendrecv_replace(buffer.data(), size_cast, type, target_rank, send_tag, source_rank, receive_tag, communicator.get(), &status);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunication::send_receive_replace: Error code is: {}", error_code);

    MPICounters::add_to_sent(buffer.size_bytes());
    MPICounters::add_to_received(buffer.size_bytes());
}

/**
 * @brief Sends one data element to target while simultaneously receiving one element from source
 *      into the same location, in a single deadlock-free operation. The uninitialized rank may be
 *      passed for either partner to skip that half of the exchange (MPI_PROC_NULL); the value is
 *      returned unchanged then.
 * @tparam T The type of data to send and receive
 * @param value The data element to send
 * @param target Where to send the data
 * @param source From where to receive the data
 * @param send_tag A tag to distinguish the send, default is 0
 * @param receive_tag A tag to distinguish the receive, default is 0
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if a partner is not in the communicator or MPI reports an error
 * @return The received data element
 */
template <MPICompatible T>
[[nodiscard]] T send_receive_replace(const T value, const MPIRank target, const MPIRank source, const int send_tag = 0, const int receive_tag = 0, const MPICommunicator& communicator = MPICommunicator::World) {
    auto buffer = value;
    send_receive_replace<T>(std::span<T>{ &buffer, 1 }, target, source, send_tag, receive_tag, communicator);
    return buffer;
}

/**
 * @brief Wildcard tag that matches a message with any tag when probing or receiving.
 */
inline constexpr int any_tag = MPI_ANY_TAG;

/**
 * @brief Metadata about an incoming message as discovered by probe/iprobe, before the message is
 *      received. The count is expressed in elements of the probed type and is only meaningful when
 *      the message is received with that same type.
 */
struct MessageInfo {
    MPIRank source; ///< The actual rank that sent the message
    int tag;        ///< The actual tag the message was sent with
    int count;      ///< The number of elements of the probed type in the message
};

/**
 * @brief Resolves an optional source rank into the raw rank id expected by MPI. An empty optional
 *      maps to MPI_ANY_SOURCE, so a probe/receive matches a message from any sender.
 * @param source The rank to receive from, or an empty optional for any source
 * @param number_ranks The number of ranks in the communicator
 * @exception Throws an Exception if the source is given but not part of the communicator
 * @return The raw rank id, or MPI_ANY_SOURCE for an empty optional
 */
[[nodiscard]] inline int resolve_source_rank(const std::optional<MPIRank> source, const int number_ranks) {
    if (!source.has_value()) {
        return MPI_ANY_SOURCE;
    }
    const auto source_rank = source->get_rank();
    utility::Exception::check(source_rank < number_ranks, "MPICommunication::resolve_source_rank: There are {} ranks, but should receive from {}", number_ranks, source_rank);
    return source_rank;
}

/**
 * @brief Extracts the message metadata from a probe status, determining the element count for the
 *      given type.
 * @tparam T The type the message is expected to hold
 * @param status The status filled in by a (blocking or non-blocking) probe
 * @exception Throws an Exception if MPI reports an error or the message size is not a multiple of
 *      the size of type T
 * @return The source, tag, and element count of the probed message
 */
template <MPICompatible T>
[[nodiscard]] MessageInfo make_message_info(const MPI_Status& status) {
    auto count = 0;
    const auto type = MPITypes::convert_type_to_mpi_type<T>();
    const auto error_code = MPI_Get_count(&status, type, &count);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunication::make_message_info: Error code is: {}", error_code);
    utility::Exception::check(count != MPI_UNDEFINED, "MPICommunication::make_message_info: The message size is not a multiple of the size of the requested type.");

    return MessageInfo{ MPIRank{ status.MPI_SOURCE }, status.MPI_TAG, count };
}

/**
 * @brief Blocks until a matching message is ready to be received, without receiving it. Reports
 *      the size, sender, and tag of the pending message so that a buffer of the right size can be
 *      allocated before receiving.
 * @tparam T The type the message is expected to hold
 * @param source The rank to probe, or an empty optional to match any source (default)
 * @param tag The tag to match, defaults to any tag
 * @param communicator The communicator to probe within, defaults to the world communicator
 * @exception Throws an Exception if the source is not in the communicator or MPI reports an error
 * @return The source, tag, and element count of the pending message
 */
template <MPICompatible T>
[[nodiscard]] MessageInfo probe(const std::optional<MPIRank> source = std::nullopt, const int tag = any_tag, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto source_rank = resolve_source_rank(source, communicator.get_number_ranks());

    auto status = MPI_Status{};
    const auto error_code = MPI_Probe(source_rank, tag, communicator.get(), &status);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunication::probe: Error code is: {}", error_code);

    return make_message_info<T>(status);
}

/**
 * @brief Checks whether a matching message is ready to be received, without blocking or receiving
 *      it. Reports the size, sender, and tag of the pending message if there is one.
 * @tparam T The type the message is expected to hold
 * @param source The rank to probe, or an empty optional to match any source (default)
 * @param tag The tag to match, defaults to any tag
 * @param communicator The communicator to probe within, defaults to the world communicator
 * @exception Throws an Exception if the source is not in the communicator or MPI reports an error
 * @return The metadata of the pending message, or an empty optional if none is ready
 */
template <MPICompatible T>
[[nodiscard]] std::optional<MessageInfo> iprobe(const std::optional<MPIRank> source = std::nullopt, const int tag = any_tag, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto source_rank = resolve_source_rank(source, communicator.get_number_ranks());

    auto flag = 0;
    auto status = MPI_Status{};
    const auto error_code = MPI_Iprobe(source_rank, tag, communicator.get(), &flag, &status);
    utility::Exception::check(error_code == MPI_SUCCESS, "MPICommunication::iprobe: Error code is: {}", error_code);

    if (flag == 0) {
        return std::nullopt;
    }

    return make_message_info<T>(status);
}

/**
 * @brief Receives a message of unknown size into a freshly allocated vector. Probes for the pending
 *      message first to size the buffer, then receives exactly that message from the sender and tag
 *      the probe matched (so a concurrent message from another sender cannot slip in).
 * @tparam T The type of data to receive
 * @param source The rank to receive from, or an empty optional to receive from any source (default)
 * @param tag The tag to match, defaults to any tag
 * @param communicator The communicator to communicate within, defaults to the world communicator
 * @exception Throws an Exception if the source is not in the communicator or MPI reports an error
 * @return A vector holding the received data
 */
template <MPICompatible T>
[[nodiscard]] std::vector<T> receive_dynamic(const std::optional<MPIRank> source = std::nullopt, const int tag = any_tag, const MPICommunicator& communicator = MPICommunicator::World) {
    const auto info = probe<T>(source, tag, communicator);

    auto buffer = std::vector<T>(utility::safe_cast<std::size_t>(info.count));
    receive<T>(buffer, info.source, info.tag, communicator);

    return buffer;
}
} // namespace MPICommunication

} // namespace mpiPP
