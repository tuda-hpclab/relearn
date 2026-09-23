#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceArray.h"

#include <mpi.h>

#include <span>
#include <vector>

/**
 * Manages the lifetime of an outstanding async MPI operation on a pair of device buffers.
 *
 * Holds both the send buffer (device-side) and the receive buffer (device or host-pinned,
 * depending on CUDA-aware MPI availability) and provides wait() to block until the transfer
 * completes and the receive buffer is ready on the device.
 */
template <typename T>
class AsyncRequest {
public:
    ~AsyncRequest() = default;

    AsyncRequest(const AsyncRequest&) = delete;
    AsyncRequest& operator=(const AsyncRequest&) = delete;

    /**
     * @brief Constructs an async request, taking ownership of the send buffer and allocating a receive buffer.
     * @param _d_out_buffer    Send buffer (ownership transferred).
     * @param _recv_buffer_size Number of elements to allocate in the receive buffer.
     */
    AsyncRequest(DeviceArray<T>&& _d_out_buffer, std::size_t _recv_buffer_size)
        : h_out_buffer(std::vector<T>())
        , h_recv_buffer(std::vector<T>())
        , d_out_buffer(std::move(_d_out_buffer))
        , d_recv_buffer_ref(&d_recv_buffer)
        , d_out_buffer_ref(&d_out_buffer)
        , recv_buffer_size(_recv_buffer_size) {
        d_recv_buffer = DeviceArray<T>(Config::cuda_aware_mpi_available ? recv_buffer_size : 0);

        if (Config::cuda_aware_mpi_available) {
            recv_buffer_ptr = d_recv_buffer.device_ptr();
            send_buffer_ptr = d_out_buffer.device_ptr();
        } else {
            h_out_buffer = d_out_buffer.get_device_data();
            send_buffer_ptr = h_out_buffer.data();
            h_recv_buffer = std::vector<T>(recv_buffer_size);
            recv_buffer_ptr = h_recv_buffer.data();
        }
    }

    /**
     * @brief Constructs an async request reusing externally owned send and receive device buffers.
     * @param _d_out_buffer   Send buffer reference (not owned).
     * @param _d_recv_buffer  Receive buffer reference (resized if too small).
     * @param _recv_buffer_size Number of elements expected in the receive buffer.
     */
    AsyncRequest(DeviceArray<T>& _d_out_buffer, DeviceArray<T>& _d_recv_buffer, std::size_t _recv_buffer_size)
        : recv_buffer_size(_recv_buffer_size) {
        if (Config::cuda_aware_mpi_available && _d_recv_buffer.size() < _recv_buffer_size) {
            _d_recv_buffer = DeviceArray<T>(_recv_buffer_size);
        }
        d_recv_buffer_ref = &_d_recv_buffer;
        d_out_buffer_ref = &_d_out_buffer;

        h_recv_buffer = std::vector<T>();
        h_out_buffer = std::vector<T>();

        if (Config::cuda_aware_mpi_available) {
            recv_buffer_ptr = d_recv_buffer_ref->device_ptr();
            send_buffer_ptr = d_out_buffer_ref->device_ptr();
        } else {
            h_out_buffer = d_out_buffer_ref->get_device_data();
            send_buffer_ptr = h_out_buffer.data();
            h_recv_buffer = std::vector<T>(recv_buffer_size);
            recv_buffer_ptr = h_recv_buffer.data();
        }
    }

    // d_recv_buffer_ref / d_out_buffer_ref may either point at externally-owned buffers (the
    // reuse constructor above) or self-reference this object's own d_recv_buffer / d_out_buffer
    // members (the ownership-taking constructor above). The implicit move constructor would
    // bitwise-copy those pointers, leaving a self-referencing case dangling into the moved-from
    // object -- cuda_all_to_all_async/_p2p's ownership-taking overloads construct an AsyncRequest
    // locally and return it by value (a move), so this must be handled explicitly.
    AsyncRequest(AsyncRequest&& other) noexcept
        : request(other.request)
        , multi_requests(std::move(other.multi_requests))
        , h_out_buffer(std::move(other.h_out_buffer))
        , h_recv_buffer(std::move(other.h_recv_buffer))
        , d_recv_buffer(std::move(other.d_recv_buffer))
        , d_out_buffer(std::move(other.d_out_buffer))
        , d_recv_buffer_ref((other.d_recv_buffer_ref == &other.d_recv_buffer) ? &d_recv_buffer : other.d_recv_buffer_ref)
        , d_out_buffer_ref((other.d_out_buffer_ref == &other.d_out_buffer) ? &d_out_buffer : other.d_out_buffer_ref)
        , send_buffer_ptr(other.send_buffer_ptr)
        , recv_buffer_ptr(other.recv_buffer_ptr)
        , recv_buffer_size(other.recv_buffer_size) {
        other.request = MPI_REQUEST_NULL;
    }

    AsyncRequest& operator=(AsyncRequest&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        request = other.request;
        multi_requests = std::move(other.multi_requests);
        h_out_buffer = std::move(other.h_out_buffer);
        h_recv_buffer = std::move(other.h_recv_buffer);
        d_recv_buffer = std::move(other.d_recv_buffer);
        d_out_buffer = std::move(other.d_out_buffer);
        send_buffer_ptr = other.send_buffer_ptr;
        recv_buffer_ptr = other.recv_buffer_ptr;
        recv_buffer_size = other.recv_buffer_size;
        d_recv_buffer_ref = (other.d_recv_buffer_ref == &other.d_recv_buffer) ? &d_recv_buffer : other.d_recv_buffer_ref;
        d_out_buffer_ref = (other.d_out_buffer_ref == &other.d_out_buffer) ? &d_out_buffer : other.d_out_buffer_ref;
        other.request = MPI_REQUEST_NULL;
        return *this;
    }

    /**
     * @brief Stores the MPI_Request for a single-request transfer.
     * @param _request The MPI_Request handle returned by MPI_Isend/MPI_Irecv.
     */
    void set_request(MPI_Request _request) {
        request = _request;
    }

    /**
     * @brief Stores MPI_Request handles for a multi-request (all-to-all) transfer.
     * @param reqs Vector of MPI_Request handles.
     */
    void set_multi_requests(std::vector<MPI_Request> reqs) {
        multi_requests = std::move(reqs);
    }

    /**
     * @brief Waits for the MPI transfer, then copies data to the device if not CUDA-aware.
     * @return Moved device receive buffer (ownership transferred to the caller).
     */
    [[nodiscard]] DeviceArray<T>&& wait_no_ref() {
        if (!multi_requests.empty()) {
            MPI_Waitall(static_cast<int>(multi_requests.size()), multi_requests.data(), MPI_STATUSES_IGNORE);
        } else {
            MPI_Wait(&request, MPI_STATUS_IGNORE); // NOLINT(clang-analyzer-optin.mpi.MPI-Checker) - request is populated externally via set_request()/get_request_ptr(), not visible to the analyzer here
        }
        if (!Config::cuda_aware_mpi_available) {
            *d_recv_buffer_ref = DeviceArray<T>(std::move(h_recv_buffer));
        }
        return std::move(d_recv_buffer);
    }

    /**
     * @brief Waits for the MPI transfer and ensures the receive data is available on the device.
     */
    void wait() {
        if (!multi_requests.empty()) {
            MPI_Waitall(static_cast<int>(multi_requests.size()), multi_requests.data(), MPI_STATUSES_IGNORE);
        } else {
            MPI_Wait(&request, MPI_STATUS_IGNORE); // NOLINT(clang-analyzer-optin.mpi.MPI-Checker) - request is populated externally via set_request()/get_request_ptr(), not visible to the analyzer here
        }
        if (!Config::cuda_aware_mpi_available) {
            *d_recv_buffer_ref = DeviceArray<T>(std::move(h_recv_buffer));
        }
    }

    /**
     * @brief MPI-only wait: does not touch CUDA.
     * Call copy_to_device() on the main thread afterwards.
     */
    void wait_mpi_only() {
        if (!multi_requests.empty()) {
            MPI_Waitall(static_cast<int>(multi_requests.size()), multi_requests.data(), MPI_STATUSES_IGNORE);
        } else {
            MPI_Wait(&request, MPI_STATUS_IGNORE); // NOLINT(clang-analyzer-optin.mpi.MPI-Checker) - request is populated externally via set_request()/get_request_ptr(), not visible to the analyzer here
        }
    }

    /**
     * @brief Host→device copy for non-CUDA-aware MPI; must be called on the main thread after wait_mpi_only().
     */
    void copy_to_device() {
        if (!Config::cuda_aware_mpi_available) {
            *d_recv_buffer_ref = DeviceArray<T>(std::move(h_recv_buffer));
        }
    }

    /**
     * @brief Returns the raw send-buffer pointer (device or host depending on CUDA-aware availability).
     */
    [[nodiscard]] T* get_send_buffer_ptr() const {
        return send_buffer_ptr;
    }

    /**
     * @brief Returns the raw receive-buffer pointer (device or host depending on CUDA-aware availability).
     */
    [[nodiscard]] T* get_recv_buffer_ptr() const {
        return recv_buffer_ptr;
    }

    /**
     * @brief Returns a pointer to the MPI_Request handle (for passing to MPI_Isend / MPI_Irecv).
     */
    [[nodiscard]] MPI_Request* get_request_ptr() {
        return &request;
    }

private:
    MPI_Request request{ MPI_REQUEST_NULL };
    std::vector<MPI_Request> multi_requests{};
    std::vector<T> h_out_buffer{};
    std::vector<T> h_recv_buffer{};
    DeviceArray<T> d_recv_buffer{ 0 };
    DeviceArray<T> d_out_buffer{ 0 };
    DeviceArray<T>* d_recv_buffer_ref;
    DeviceArray<T>* d_out_buffer_ref;
    T* send_buffer_ptr{};
    T* recv_buffer_ptr{};

    std::size_t recv_buffer_size{};
};

/** Result of a synchronous all-to-all/allgather exchange: received/gathered data, per-rank counts, and per-rank displacements. */
template <typename T>
struct AllToAllResult {
    DeviceArray<T> data;
    std::vector<int> sizes{};
    std::vector<int> displacements{};
};

/** Result of an asynchronous all-to-all exchange: per-rank receive counts/displacements, and the AsyncRequest to wait on. */
template <typename T>
struct AsyncAllToAllResult {
    std::vector<int> sizes{};
    std::vector<int> displacements{};
    AsyncRequest<T> request;
};

/**
 * @brief Performs a synchronous MPI_Alltoallv over GPU buffers.
 * @param out_buffer   Device buffer containing data to send; already partitioned by target rank.
 * @param h_sizes      Per-rank send counts.
 * @param number_ranks Total number of MPI ranks.
 * @param stats        If true, record byte-transfer statistics.
 * @return Receive buffer, per-rank receive counts, and per-rank receive displacements.
 */
template <typename T>
[[nodiscard]] AllToAllResult<T> cuda_all_to_all(const DeviceArray<T>& out_buffer, const std::span<const int> h_sizes, std::size_t number_ranks, bool stats = false);

/**
 * @brief Overload of cuda_all_to_all_async that takes ownership of the send buffer.
 * @param out_buffer   Device send buffer (ownership transferred).
 * @param h_sizes      Per-rank send counts.
 * @param number_ranks Total number of MPI ranks.
 * @param stats        If true, record byte-transfer statistics.
 * @return Per-rank receive counts, receive displacements, and an AsyncRequest to wait on.
 */
template <typename T>
[[nodiscard]] AsyncAllToAllResult<T> cuda_all_to_all_async(DeviceArray<T>&& out_buffer, const std::span<const int> h_sizes, std::size_t number_ranks, bool stats);

/**
 * @brief P2P (point-to-point) variant of cuda_all_to_all_async using MPI_Isend/MPI_Irecv.
 * @param out_buffer   Device send buffer.
 * @param recv_buffer  Device receive buffer (resized if too small).
 * @param h_sizes      Per-rank send counts.
 * @param number_ranks Total number of MPI ranks.
 * @param stats        If true, record byte-transfer statistics.
 * @return Per-rank receive counts, receive displacements, and an AsyncRequest to wait on.
 */
template <typename T>
[[nodiscard]] AsyncAllToAllResult<T> cuda_all_to_all_async_p2p(DeviceArray<T>& out_buffer, DeviceArray<T>& recv_buffer, const std::span<const int> h_sizes, std::size_t number_ranks, bool stats);

/**
 * @brief Performs an MPI_Allgather over GPU buffers.
 * @param out_buffer   Device buffer with the data this rank contributes.
 * @param number_ranks Total number of MPI ranks.
 * @param stream       CUDA stream for any device-side preparation.
 * @return Gathered device buffer, per-rank counts, and per-rank displacements.
 */
template <typename T>
[[nodiscard]] AllToAllResult<T> cuda_allgather(const DeviceArray<T>& out_buffer, std::size_t number_ranks, const std::shared_ptr<StreamWrapper>& stream);

/**
 * @brief Returns cumulative bytes sent across all cuda_all_to_all* calls.
 */
[[nodiscard]] std::uint64_t get_send_bytes();

/**
 * @brief Returns cumulative bytes received across all cuda_all_to_all* calls.
 */
[[nodiscard]] std::uint64_t get_recv_bytes();

/**
 * @brief Returns a per-rank breakdown of bytes received in each all-to-all call.
 */
[[nodiscard]] std::vector<std::vector<int>> get_received_bytes_vector();

/**
 * @brief Returns a per-rank breakdown of bytes sent in each all-to-all call.
 */
[[nodiscard]] std::vector<std::vector<int>> get_send_bytes_vector();

/**
 * @brief Verifies that all ranks agree on CUDA-aware MPI availability; aborts on mismatch.
 * @param my_rank      MPI rank of this process.
 * @param number_ranks Total number of MPI ranks.
 */
void check_cuda_awareness(CudaConfig::mpi_rank_type my_rank, CudaConfig::mpi_rank_type number_ranks);
