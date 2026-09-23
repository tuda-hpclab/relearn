/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FireStatusCommunicatorGPUUncompressed.h"

#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/mpi/MPICuda.h"
#include "cuda/spikes/SpikePreparation.h"
#include "cuda/util/Util.h"
#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "util/Timers.h"

#include <cpp-utility/data/displacement.hpp>

#include <mpi-wrapper/core/MPITypes.h>

#include <mpi.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

void FireStatusCommunicatorGPUUncompressed::commit_local_fired_status([[maybe_unused]] const step_type step) {
#ifndef RELEARN_CUDA_ENABLED
    CUDA_NOT_SUPPORTED
#else
    const auto _number_ranks = get_number_ranks();
    if (_number_ranks == 1) {
        return;
    }

    Timers::start(TimerRegion::PREPARE_SENDING_SPIKES);
    [[maybe_unused]] const auto number_neurons = extra_infos->get_size();
    const auto* d_fired = get_fired_status_recorder()->get_d_fired_const();
    Timers::start(TimerRegion::BLOCK1);
    auto do_binary_search = Config::do_binary_search;
    auto out_data_pair = prepare_spikes(FiredStatusHandle{ d_fired }, get_network_graph()->get_gpu_handle_const(),
                                        extra_infos->get_gpu_handle(), do_binary_search, spike_sort_stream);
    Timers::stop_and_add(TimerRegion::BLOCK1);

    auto& [h_sizes, fired_neuron_ids] = out_data_pair;
    MPI_Barrier(MPI_COMM_WORLD);
    d_outgoing_data = std::move(fired_neuron_ids);

    h_outgoing_sizes = std::move(h_sizes);

    Timers::stop_and_add(TimerRegion::PREPARE_SENDING_SPIKES);
#endif
}

void FireStatusCommunicatorGPUUncompressed::exchange_fired_status([[maybe_unused]] const step_type step) {
    if (get_number_ranks() == 1) {
        return;
    }
    Timers::start(TimerRegion::EXCHANGE_NEURON_IDS);

    cudaStreamSynchronize_bride(*spike_sort_stream);

    const auto _number_ranks = static_cast<std::size_t>(get_number_ranks());
    const auto mpi_datatype = mpiPP::MPITypes::convert_type_to_mpi_type<CudaConfig::number_neurons_type>();

    std::vector<int> h_displ(_number_ranks, 0);
    for (auto i = 1U; i < _number_ranks; i++) {
        h_displ[i] = h_displ[i - 1] + h_outgoing_sizes[i - 1];
    }

    // Collective — must stay on the main thread with MPI_COMM_WORLD
    Timers::start(TimerRegion::MPI_ALLGATHER);
    std::vector<int> received_sizes(_number_ranks, 0);
    MPI_Alltoall(h_outgoing_sizes.data(), 1, MPI_INT, received_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    Timers::stop_and_add(TimerRegion::MPI_ALLGATHER);

    std::vector<int> recv_displs(_number_ranks, 0);
    for (auto i = 1U; i < _number_ranks; i++) {
        recv_displs[i] = recv_displs[i - 1] + received_sizes[i - 1];
    }
    d_incoming_displ = DeviceArray<int>(utility::calculate_offsets(std::span<const int>{ received_sizes.cbegin(), received_sizes.cend() }));

    const auto recv_buffer_size = std::accumulate(received_sizes.cbegin(), received_sizes.cend(), 0U);
    // AsyncRequest allocates recv buffer and (non-CUDA-aware) copies outgoing data to host —
    // both CUDA operations happen here on the main thread.
    current_async_req_ = std::make_unique<AsyncRequest<CudaConfig::number_neurons_type>>(d_outgoing_data, d_incoming_data, recv_buffer_size);
    auto* send_buffer_ptr = current_async_req_->get_send_buffer_ptr();
    auto* recv_buffer_ptr = current_async_req_->get_recv_buffer_ptr();
    const auto send_sizes = h_outgoing_sizes; // copy: background thread reads after potential next commit

    // Post Isend/Irecv and wait on the background thread — pure MPI, no CUDA calls.
    // copy_to_device() is called on the main thread inside wait_for_exchange_to_finish().
    {
        const std::unique_lock<std::mutex> lock(work_mtx);
        work_item = [this, mpi_datatype, _number_ranks,
                     h_displ = std::move(h_displ), received_sizes = std::move(received_sizes),
                     recv_displs = std::move(recv_displs), send_sizes = send_sizes,
                     send_buffer_ptr, recv_buffer_ptr]() mutable {
            std::vector<MPI_Request> requests;
            requests.reserve(2 * _number_ranks);

            // NOLINTBEGIN(clang-analyzer-optin.mpi.MPI-Checker) - these requests are waited on via
            // current_async_req_->wait_mpi_only() below, not a direct MPI_Wait the analyzer can trace
            for (auto rank = 0U; rank < _number_ranks; rank++) {
                if (received_sizes[rank] > 0) {
                    MPI_Request req = MPI_REQUEST_NULL;
                    MPI_Irecv(recv_buffer_ptr + recv_displs[rank], received_sizes[rank], mpi_datatype,
                              static_cast<int>(rank), 0, p2p_comm, &req);
                    requests.push_back(req);
                }
            }
            for (auto rank = 0U; rank < _number_ranks; rank++) {
                if (send_sizes[rank] > 0) {
                    MPI_Request req = MPI_REQUEST_NULL;
                    MPI_Isend(send_buffer_ptr + h_displ[rank], send_sizes[rank], mpi_datatype,
                              static_cast<int>(rank), 0, p2p_comm, &req);
                    requests.push_back(req);
                }
            }
            // NOLINTEND(clang-analyzer-optin.mpi.MPI-Checker)

            current_async_req_->set_multi_requests(std::move(requests));
            current_async_req_->wait_mpi_only(); // MPI only — no CUDA
        };
        work_done = false;
        work_ready = true;
    }
    work_cv.notify_one();

    Timers::stop_and_add(TimerRegion::EXCHANGE_NEURON_IDS);
}

void FireStatusCommunicatorGPUUncompressed::wait_for_exchange_to_finish() {
    std::exception_ptr ex;
    {
        std::unique_lock<std::mutex> lock(work_mtx);
        done_cv.wait(lock, [this] { return work_done; });
        ex = std::exchange(thread_exception, nullptr);
    }
    // h→d copy must happen on the main thread (CUPTI is not thread-safe for concurrent CUDA tracing).
    if (current_async_req_) {
        if (!ex) {
            current_async_req_->copy_to_device();
        }
        current_async_req_.reset();
    }
    if (ex) {
        std::rethrow_exception(ex);
    }
}
