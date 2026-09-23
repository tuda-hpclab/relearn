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

#include <atomic>
#include <cstdint>

namespace mpiPP {

/**
 * @brief Offers basic functionality for a thread-safe counter
 */
class AtomicCounter {
public:
    /**
     * @brief Resets the stored counter to zero without synchronization of other memory cells
     */
    void reset() noexcept {
        counter.store(0, std::memory_order::relaxed);
    }

    /**
     * @brief Returns the stored value without synchronization of other memory cells
     * @return The stored value
     */
    [[nodiscard]] std::uint64_t get_value() noexcept {
        return counter.load(std::memory_order::relaxed);
    }

    /**
     * @brief Adds a given value to the stored value without synchronization of other memory cells
     * @param value The value to add
     */
    void add(const std::uint64_t value) noexcept {
        counter.fetch_add(value, std::memory_order::relaxed);
    }

private:
    std::atomic<std::uint64_t> counter{ 0 };
};

/**
 * @brief Tracks logical byte volumes processed by wrapper operations. These are approximations:
 *      collectives may differ from wire traffic, and RMA counts include operations targeting local memory.
 */
class MPICounters {
public:
    /**
     * @brief Starts measuring the communication that is sent, received, or remotely accessed
     */
    static void start_measuring_communication() noexcept {
        measure_communication.store(true, std::memory_order::relaxed);
    }

    /**
     * @brief Stops measuring the communication that is sent, received, or remotely accessed
     */
    static void stop_measuring_communication() noexcept {
        measure_communication.store(false, std::memory_order::relaxed);
    }

    /**
     * @brief Adds the number of bytes to the send counter if communication is measured
     * @param number_bytes The number of bytes to add
     */
    static void add_to_sent(std::uint64_t number_bytes) noexcept {
        if (measure_communication.load(std::memory_order::relaxed)) {
            send_counter.add(number_bytes);
        }
    }

    /**
     * @brief Adds the number of bytes to the receive counter if communication is measured
     * @param number_bytes The number of bytes to add
     */
    static void add_to_received(std::uint64_t number_bytes) noexcept {
        if (measure_communication.load(std::memory_order::relaxed)) {
            receive_counter.add(number_bytes);
        }
    }

    /**
     * @brief Adds the number of bytes to the rma counter if communication is measured
     * @param number_bytes The number of bytes to add
     */
    static void add_to_remotely_accessed(std::uint64_t number_bytes) noexcept {
        if (measure_communication.load(std::memory_order::relaxed)) {
            rma_counter.add(number_bytes);
        }
    }

    /**
     * @brief Returns an approximation of how many bytes were sent.
     *      E.g., it only counts reduce once, so this is an underapproximation.
     * @return The number of bytes sent
     */
    [[nodiscard]] static std::uint64_t get_number_bytes_sent() noexcept {
        return send_counter.get_value();
    }

    /**
     * @brief Returns an approximation of how many bytes were received.
     *      E.g., it only counts reduce on the root rank, so this is an underapproximation.
     * @return The number of bytes received
     */
    [[nodiscard]] static std::uint64_t get_number_bytes_received() noexcept {
        return receive_counter.get_value();
    }

    /**
     * @brief Returns the logical byte volume processed by RMAWindow operations, including local targets
     * @return The accumulated RMA byte volume
     */
    [[nodiscard]] static std::uint64_t get_number_bytes_remotely_accessed() noexcept {
        return rma_counter.get_value();
    }

    /**
     * @brief Resets the counter for the number of bytes sent
     */
    static void reset_sent_counter() noexcept {
        send_counter.reset();
    }

    /**
     * @brief Resets the counter for the number of bytes received
     */
    static void reset_received_counter() noexcept {
        receive_counter.reset();
    }

    /**
     * @brief Resets the counter for the number of bytes remotely accessed
     */
    static void reset_remotely_accessed_counter() noexcept {
        rma_counter.reset();
    }

private:
    static inline std::atomic<bool> measure_communication{ true };

    static inline AtomicCounter send_counter{};
    static inline AtomicCounter receive_counter{};
    static inline AtomicCounter rma_counter{};
};

} // namespace mpiPP
