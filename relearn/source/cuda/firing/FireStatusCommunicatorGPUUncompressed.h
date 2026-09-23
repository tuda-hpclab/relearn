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

#include "FiredStatusCommunicatorGPU.h"

#include "cuda/input/Handle.h"
#include "cuda/mpi/MPICuda.h"
#include "cuda/util/Util.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "types/CommunicationTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <mpi.h>

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

/**
 * This class communicates the fired status of the local neurons
 * via two separate RelearnTypes::comm_map_firing
 */
class FireStatusCommunicatorGPUUncompressed : public FiredStatusCommunicatorGPU<CudaConfig::number_neurons_type> {
public:
    /**
     * @brief Constructs a new object with the given number of ranks and local neurons (mainly used for pre-allocating memory)
     * @param _my_rank The MPI rank of this process
     * @param num_ranks The number of MPI ranks
     * @exception Throws a RelearnException if num_ranks <= 0
     */
    FireStatusCommunicatorGPUUncompressed(const FireStatusCommunicatorGPUUncompressed&) = delete;
    FireStatusCommunicatorGPUUncompressed& operator=(const FireStatusCommunicatorGPUUncompressed&) = delete;
    FireStatusCommunicatorGPUUncompressed(FireStatusCommunicatorGPUUncompressed&&) = delete;
    FireStatusCommunicatorGPUUncompressed& operator=(FireStatusCommunicatorGPUUncompressed&&) = delete;

    explicit FireStatusCommunicatorGPUUncompressed(const mpiPP::MPIRank _my_rank, const int num_ranks)
        : FiredStatusCommunicatorGPU(_my_rank, num_ranks) {
        RelearnException::check(num_ranks > 0, "FiredStatusCommunicationMap::FiredStatusCommunicationMap: num_ranks is too small: {}", num_ranks);
        MPI_Comm_dup(MPI_COMM_WORLD, &p2p_comm);
        mpi_thread = std::thread([this] {
            cudaSetDevice_bridge(CudaConfig::local_gpu_id);
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(work_mtx);
                    work_cv.wait(lock, [this] { return work_ready || stop_thread; });
                    if (stop_thread && !work_ready) {
                        break;
                    }
                    task = std::move(work_item);
                    work_ready = false;
                }
                try {
                    task();
                } catch (...) {
                    const std::unique_lock<std::mutex> lock(work_mtx);
                    thread_exception = std::current_exception();
                }
                {
                    const std::unique_lock<std::mutex> lock(work_mtx);
                    work_done = true;
                }
                done_cv.notify_one();
            }
        });
    }

    // finalize() joins mpi_thread; std::thread's destructor calls std::terminate() if still
    // joinable, so callers that skip the explicit finalize() (e.g. a NeuronModel that never gets
    // torn down via Simulation::finalize()) must not crash. finalize() is idempotent (checks
    // joinable()/p2p_comm != MPI_COMM_NULL), so calling it again here after an explicit call is safe.
    ~FireStatusCommunicatorGPUUncompressed() override {
        // Explicitly qualified to avoid a virtual dispatch from the destructor; this class has no
        // subclasses, so this always resolved to the same override anyway.
        FireStatusCommunicatorGPUUncompressed::finalize();
    }

    void finalize() override {
        {
            const std::unique_lock<std::mutex> lock(work_mtx);
            stop_thread = true;
        }
        work_cv.notify_one();
        if (mpi_thread.joinable()) {
            mpi_thread.join();
        }
        if (p2p_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&p2p_comm);
        }
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or if init(...) has been called before
     */
    void init(const number_neurons_type number_neurons) override {
        FiredStatusCommunicator::init(number_neurons);
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons([[maybe_unused]] const number_neurons_type creation_count) override {
        CUDA_NOT_SUPPORTED
    }

    /**
     * @brief Registers the fired status of the local neurons that are not disabled.
     * @param step The current update step
     * @exception Can throw a RelearnException
     */
    void commit_local_fired_status(step_type step) override;

    /**
     * @brief Exchanges the fired status with all MPI ranks
     * @param step The current update step
     * @exception Can throw a RelearnException
     */
    void exchange_fired_status(step_type step) override;

    /**
     * @brief Checks if the communicator contains the specified neuron of the rank,
     *      i.e., whether that neuron fired in the last update step.
     * @param rank The MPI rank that owns the neuron
     * @param neuron_id The neuron in question
     * @exception Throws a RelearnException if rank is not from [0, number_ranks) or the neuron_id is virtual
     */
    [[nodiscard]] bool contains([[maybe_unused]] mpiPP::MPIRank rank, [[maybe_unused]] NeuronID neuron_id) const override {
        CUDA_NOT_SUPPORTED
        return false;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_easy_footprint = sizeof(*this) - sizeof(FiredStatusCommunicator);

        auto my_hard_footprint = std::uint64_t{ 0 };

        footprint->emplace("FiredStatusCommunicationMap", my_hard_footprint + my_easy_footprint);

        FiredStatusCommunicator::record_memory_footprint(footprint);
    }

    [[nodiscard]] std::unique_ptr<FireStatusCommunicatorHandle> get_handle() const override {
        return std::make_unique<FireStatusCommunicatorUncompressedHandle>(d_incoming_displ.device_ptr(), d_incoming_data.device_ptr());
    }

    void wait_for_exchange_to_finish() override;

private:
    MPI_Comm p2p_comm{ MPI_COMM_NULL };

    std::mutex work_mtx;
    std::condition_variable work_cv;
    std::condition_variable done_cv;
    std::function<void()> work_item;
    std::exception_ptr thread_exception{ nullptr };
    bool work_ready{ false };
    bool work_done{ true };
    bool stop_thread{ false };
    std::thread mpi_thread; // must be last: started after all members above are initialized

    std::shared_ptr<StreamWrapper> spike_sort_stream{ std::make_shared<StreamWrapper>() };

    // Owned on the main thread; background thread calls wait_mpi_only(), main thread calls copy_to_device().
    std::unique_ptr<AsyncRequest<CudaConfig::number_neurons_type>> current_async_req_{};
};
