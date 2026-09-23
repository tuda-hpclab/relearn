/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda/mpi/MPICuda.h"

#include "CudaTypes.h"

#include "cuda/memory/DeviceArray.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "util/Timers.h"

#include <mpi-wrapper/core/MPITypes.h>

#include <mpi.h>

#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

static std::uint64_t send_bytes{ 0 };
static std::uint64_t received_bytes{ 0 };
static std::vector<std::vector<int>> send_bytes_vector{};
static std::vector<std::vector<int>> received_bytes_vector{};

template <typename T>
[[nodiscard]] AllToAllResult<T> cuda_all_to_all(const DeviceArray<T>& out_buffer, const std::span<const int> h_sizes, const std::size_t number_ranks, bool stats) {
    //   Timers::start(TimerRegion::CUDA_ALLTOALL);

    std::vector<int> h_displ(number_ranks, 0);
    h_displ[0] = 0;
    for (auto i = 1U; i < number_ranks; i++) {
        h_displ[i] = h_displ[i - 1] + h_sizes[i - 1];
    }

    const auto send_size = std::accumulate(h_sizes.begin(), h_sizes.end(), 0);
    RelearnException::check(out_buffer.size() >= static_cast<std::size_t>(send_size), "cuda_all_to_all {} != {}", out_buffer.size(), send_size);

    std::vector<int> received_sizes(number_ranks, 0);
    // Timers::start(TimerRegion::MPI_ALLGATHER);
    MPI_Alltoall(h_sizes.data(), 1, MPI_INT, received_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    // Timers::stop_and_add(TimerRegion::MPI_ALLGATHER);

    std::vector<int> recv_displs(number_ranks, 0);
    for (auto i = 1U; i < number_ranks; i++) {
        recv_displs[i] = recv_displs[i - 1] + received_sizes[i - 1];
    }
    const auto recv_buffer_size = std::accumulate(received_sizes.cbegin(), received_sizes.cend(), std::size_t{ 0 });
    DeviceArray<T> recv_buffer(Config::cuda_aware_mpi_available ? static_cast<std::size_t>(recv_buffer_size) : 0);
    const auto mpi_datatype = mpiPP::MPITypes::convert_type_to_mpi_type<T>();

    T* sendbuf = nullptr;
    T* recv_buffer_ptr = nullptr;
    std::vector<T> h_out_buffer{};
    std::vector<T> h_recv_buffer{};
    if (Config::cuda_aware_mpi_available) {
        recv_buffer_ptr = recv_buffer.device_ptr();
        sendbuf = out_buffer.device_ptr();
    } else {
        h_out_buffer = out_buffer.get_device_data();
        sendbuf = h_out_buffer.data();
        h_recv_buffer = std::vector<T>(recv_buffer_size);
        recv_buffer_ptr = h_recv_buffer.data();
    }
    const auto* send_counts_ptr = h_sizes.data();
    auto* send_displs_ptr = h_displ.data();

    auto* recv_sizes_ptr = received_sizes.data();
    auto* recv_displs_ptr = recv_displs.data();

    if (Config::cuda_aware_mpi_available) {
        cudaDeviceSynchronize_bridge();
    }
    //   Timers::start(TimerRegion::MPI_ALLTOALL);
    MPI_Alltoallv(sendbuf, send_counts_ptr, send_displs_ptr, mpi_datatype,
                  recv_buffer_ptr, recv_sizes_ptr, recv_displs_ptr, mpi_datatype, MPI_COMM_WORLD);
    // Timers::stop_and_add(TimerRegion::MPI_ALLTOALL);

    if (!Config::cuda_aware_mpi_available) {
        recv_buffer = DeviceArray<T>(std::move(h_recv_buffer));
    }
    // Timers::stop_and_add(TimerRegion::CUDA_ALLTOALL);

    if (stats) {
        send_bytes += out_buffer.size();
        send_bytes_vector.emplace_back(h_sizes.begin(), h_sizes.end());
        received_bytes_vector.emplace_back(received_sizes);
        received_bytes += recv_buffer_size;
    }

    return AllToAllResult<T>{ std::move(recv_buffer), received_sizes, recv_displs };
}

template <typename T>
[[nodiscard]] AsyncAllToAllResult<T> cuda_all_to_all_async(DeviceArray<T>&& out_buffer, const std::span<const int> h_sizes, const std::size_t number_ranks, bool stats) {
    Timers::start(TimerRegion::CUDA_ALLTOALL);

    std::vector<int> h_displ(number_ranks, 0);
    h_displ[0] = 0;
    for (auto i = 1U; i < number_ranks; i++) {
        h_displ[i] = h_displ[i - 1] + h_sizes[i - 1];
    }

    const auto send_size = std::accumulate(h_sizes.begin(), h_sizes.end(), 0);
    RelearnException::check(out_buffer.size() >= static_cast<std::size_t>(send_size), "cuda_all_to_all {} != {}", out_buffer.size(), send_size);

    std::vector<int> received_sizes(number_ranks, 0);
    Timers::start(TimerRegion::MPI_ALLGATHER);
    MPI_Alltoall(h_sizes.data(), 1, MPI_INT, received_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    Timers::stop_and_add(TimerRegion::MPI_ALLGATHER);

    std::vector<int> recv_displs(number_ranks, 0);
    for (auto i = 1U; i < number_ranks; i++) {
        recv_displs[i] = recv_displs[i - 1] + received_sizes[i - 1];
    }
    const auto mpi_datatype = mpiPP::MPITypes::convert_type_to_mpi_type<T>();
    const auto* send_counts_ptr = h_sizes.data();
    auto* send_displs_ptr = h_displ.data();
    auto* recv_sizes_ptr = received_sizes.data();
    auto* recv_displs_ptr = recv_displs.data();

    const auto recv_buffer_size = std::accumulate(received_sizes.cbegin(), received_sizes.cend(), std::size_t{ 0 });
    AsyncRequest<T> async_request(std::move(out_buffer), recv_buffer_size);
    auto* send_buffer_ptr = async_request.get_send_buffer_ptr();
    const auto recv_buffer_ptr = async_request.get_recv_buffer_ptr();

    Timers::start(TimerRegion::MPI_ALLTOALL);
    MPI_Ialltoallv(send_buffer_ptr, send_counts_ptr, send_displs_ptr, mpi_datatype,
                   recv_buffer_ptr, recv_sizes_ptr,
                   recv_displs_ptr, mpi_datatype, MPI_COMM_WORLD, async_request.get_request_ptr());
    Timers::stop_and_add(TimerRegion::MPI_ALLTOALL);

    Timers::stop_and_add(TimerRegion::CUDA_ALLTOALL);

    if (stats) {
        send_bytes += static_cast<uint64_t>(send_size);
        send_bytes_vector.emplace_back(h_sizes.begin(), h_sizes.end());
        received_bytes_vector.emplace_back(received_sizes);
        received_bytes += recv_buffer_size;
    }

    return AsyncAllToAllResult<T>{ received_sizes, recv_displs, std::move(async_request) };
}

template <typename T>
[[nodiscard]] AllToAllResult<T> cuda_allgather(const DeviceArray<T>& out_buffer, const std::size_t number_ranks, const std::shared_ptr<StreamWrapper>& stream) {

    Timers::start(TimerRegion::CUDA_ALLTOALL);

    std::vector<int> received_sizes(number_ranks, 0);
    Timers::start(TimerRegion::MPI_ALLGATHER);
    const auto h_size = static_cast<int>(out_buffer.size());
    MPI_Allgather(&h_size, 1, MPI_INT, received_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    Timers::stop_and_add(TimerRegion::MPI_ALLGATHER);

    std::vector<int> recv_displs(number_ranks, 0);
    for (auto i = 1U; i < number_ranks; i++) {
        recv_displs[i] = recv_displs[i - 1] + received_sizes[i - 1];
    }

    const auto recv_buffer_size = std::accumulate(received_sizes.cbegin(), received_sizes.cend(), std::size_t{ 0 });

    DeviceArray<T> recv_buffer(Config::cuda_aware_mpi_available ? static_cast<std::size_t>(recv_buffer_size) : 0, stream);

    T* sendbuf = nullptr;
    T* recv_buffer_ptr = nullptr;
    std::vector<T> h_out_buffer{};
    std::vector<T> h_recv_buffer{};
    if (Config::cuda_aware_mpi_available) {
        recv_buffer_ptr = recv_buffer.device_ptr();
        sendbuf = out_buffer.device_ptr();
    } else {
        h_out_buffer = out_buffer.get_device_data();
        sendbuf = h_out_buffer.data();
        h_recv_buffer = std::vector<T>(recv_buffer_size);
        recv_buffer_ptr = h_recv_buffer.data();
    }

    const auto mpi_datatype = mpiPP::MPITypes::convert_type_to_mpi_type<T>();

    cudaStreamSynchronize_bride(*stream);

    MPI_Barrier(MPI_COMM_WORLD);
    Timers::start(TimerRegion::MPI_ALLTOALL);
    MPI_Allgatherv(sendbuf, h_size, mpi_datatype,
                   recv_buffer_ptr,
                   received_sizes.data(), recv_displs.data(), mpi_datatype, MPI_COMM_WORLD);
    Timers::stop_and_add(TimerRegion::MPI_ALLTOALL);

    if (!Config::cuda_aware_mpi_available) {
        recv_buffer = DeviceArray<T>(std::move(h_recv_buffer));
    }

    Timers::stop_and_add(TimerRegion::CUDA_ALLTOALL);
    return AllToAllResult<T>{ std::move(recv_buffer), received_sizes, recv_displs };
}

std::uint64_t get_send_bytes() {
    return send_bytes;
}
std::uint64_t get_recv_bytes() {
    return received_bytes;
}

std::vector<std::vector<int>> get_received_bytes_vector() {
    return received_bytes_vector;
}

std::vector<std::vector<int>> get_send_bytes_vector() {
    return send_bytes_vector;
}

[[maybe_unused]] void check_cuda_awareness(CudaConfig::mpi_rank_type my_rank, CudaConfig::mpi_rank_type number_ranks) {

    const int h_val = my_rank;
    auto d_data = DeviceArray<int>(std::span<const int>(&h_val, 1));
    auto d_recv = DeviceArray<int>(1);

    MPI_Allreduce(d_data.device_ptr(), d_recv.device_ptr(), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const auto recv_val = d_recv.get_device_data()[0];

    auto gold = 0;
    for (auto i = 0; i < number_ranks; i++) {
        gold += i;
    }
    RelearnException::check(gold == recv_val, "CudaAwareness test failed");
}

template AllToAllResult<SimpleVec3d> cuda_allgather<SimpleVec3d>(const DeviceArray<SimpleVec3d>& out_buffer, std::size_t number_ranks, const std::shared_ptr<StreamWrapper>& stream);
template AllToAllResult<unsigned int> cuda_allgather<unsigned int>(const DeviceArray<unsigned int>& out_buffer, std::size_t number_ranks, const std::shared_ptr<StreamWrapper>& stream);

template AllToAllResult<CudaConfig::bh_index_type> cuda_all_to_all(const DeviceArray<CudaConfig::bh_index_type>& out_buffer, std::span<const int> h_sizes, std::size_t number_ranks, bool stats);
template AllToAllResult<SynapseCreationResponse> cuda_all_to_all(const DeviceArray<SynapseCreationResponse>& out_buffer, std::span<const int> h_sizes, std::size_t number_ranks, bool stats);

template AllToAllResult<std::uint64_t> cuda_all_to_all(const DeviceArray<std::uint64_t>& out_buffer, std::span<const int> h_sizes, std::size_t number_ranks, bool stats);

template AllToAllResult<SimpleVec3d> cuda_all_to_all(const DeviceArray<SimpleVec3d>& out_buffer, std::span<const int> h_sizes, std::size_t number_ranks, bool stats);

template AsyncAllToAllResult<CudaConfig::bh_index_type> cuda_all_to_all_async(DeviceArray<CudaConfig::bh_index_type>&& out_buffer, std::span<const int> h_sizes, std::size_t number_ranks, bool stats);

template <typename T>
[[nodiscard]] AsyncAllToAllResult<T> cuda_all_to_all_async_p2p(DeviceArray<T>& out_buffer, DeviceArray<T>& in_buffer, const std::span<const int> h_sizes, const std::size_t number_ranks, bool stats) {
    Timers::start(TimerRegion::CUDA_ALLTOALL);

    std::vector<int> h_displ(number_ranks, 0);
    for (auto i = 1U; i < number_ranks; i++) {
        h_displ[i] = h_displ[i - 1] + h_sizes[i - 1];
    }

    const auto send_size = std::accumulate(h_sizes.begin(), h_sizes.end(), 0);
    RelearnException::check(out_buffer.size() >= static_cast<std::size_t>(send_size), "cuda_all_to_all {} != {}", out_buffer.size(), send_size);

    std::vector<int> received_sizes(number_ranks, 0);
    Timers::start(TimerRegion::MPI_ALLGATHER);
    MPI_Alltoall(h_sizes.data(), 1, MPI_INT, received_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    Timers::stop_and_add(TimerRegion::MPI_ALLGATHER);

    std::vector<int> recv_displs(number_ranks, 0);
    for (auto i = 1U; i < number_ranks; i++) {
        recv_displs[i] = recv_displs[i - 1] + received_sizes[i - 1];
    }

    const auto mpi_datatype = mpiPP::MPITypes::convert_type_to_mpi_type<T>();
    const auto recv_buffer_size = std::accumulate(received_sizes.cbegin(), received_sizes.cend(), std::size_t{ 0 });

    AsyncRequest<T> async_request(out_buffer, in_buffer, recv_buffer_size);
    auto* send_buffer_ptr = async_request.get_send_buffer_ptr();
    const auto recv_buffer_ptr = async_request.get_recv_buffer_ptr();

    Timers::start(TimerRegion::DEVICE_SYNC);
    cudaDeviceSynchronize_bridge();
    Timers::stop_and_add(TimerRegion::DEVICE_SYNC);

    std::vector<MPI_Request> requests;
    requests.reserve(2 * number_ranks);

    Timers::start(TimerRegion::MPI_ALLTOALL);

    // NOLINTBEGIN(clang-analyzer-optin.mpi.MPI-Checker) - the analyzer only sees this function; the
    // requests collected below are actually waited on inside AsyncRequest::set_multi_requests (via
    // MPI_Waitall), which is called just after this block.
    for (auto rank = 0U; rank < number_ranks; rank++) {
        if (received_sizes[rank] > 0) {
            MPI_Request req = MPI_REQUEST_NULL;
            Timers::start(TimerRegion::IRECV);
            MPI_Irecv(recv_buffer_ptr + recv_displs[rank], received_sizes[rank], mpi_datatype,
                      static_cast<int>(rank), 0, MPI_COMM_WORLD, &req);
            Timers::stop_and_add(TimerRegion::IRECV);
            requests.push_back(req);
        }
    }

    for (auto rank = 0U; rank < number_ranks; rank++) {
        if (h_sizes[rank] > 0) {
            MPI_Request req = MPI_REQUEST_NULL;
            Timers::start(TimerRegion::ISEND);
            MPI_Isend(send_buffer_ptr + h_displ[rank], h_sizes[rank], mpi_datatype,
                      static_cast<int>(rank), 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            Timers::stop_and_add(TimerRegion::ISEND);
        }
    }
    // NOLINTEND(clang-analyzer-optin.mpi.MPI-Checker)

    Timers::stop_and_add(TimerRegion::MPI_ALLTOALL);

    async_request.set_multi_requests(std::move(requests));

    Timers::stop_and_add(TimerRegion::CUDA_ALLTOALL);

    if (stats) {
        send_bytes += out_buffer.size();
        send_bytes_vector.emplace_back(h_sizes.begin(), h_sizes.end());
        received_bytes_vector.emplace_back(received_sizes);
        received_bytes += recv_buffer_size;
    }

    return AsyncAllToAllResult<T>{ received_sizes, recv_displs, std::move(async_request) };
}

template AsyncAllToAllResult<CudaConfig::bh_index_type> cuda_all_to_all_async_p2p(DeviceArray<CudaConfig::bh_index_type>& out_buffer, DeviceArray<CudaConfig::bh_index_type>& in_buffer, std::span<const int> h_sizes, std::size_t number_ranks, bool stats);
