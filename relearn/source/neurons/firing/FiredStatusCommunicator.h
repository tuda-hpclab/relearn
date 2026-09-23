#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda/input/Handle.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <mpi-wrapper/core/MPIRank.h>

#include <memory>
#include <vector>

class NetworkGraph;

/**
 * This class provides a virtual interface for exchanging the NeuronID of those that fired in the simulation step.
 */
class FiredStatusCommunicator {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Constructs a new object with the given number of ranks
     * @param _my_rank The MPI rank of this process
     * @param _num_ranks The number of MPI ranks, >0
     * @exception Throws a RelearnException if number_ranks <= 0
     */
    explicit FiredStatusCommunicator(const mpiPP::MPIRank _my_rank, const int _num_ranks)
        : number_ranks(_num_ranks)
        , my_rank(_my_rank) {
        RelearnException::check(number_ranks > 0, "FiredStatusCommunicator::FiredStatusCommunicator: num_ranks is too small: {}", number_ranks);
    }

    FiredStatusCommunicator(const FiredStatusCommunicator&) = default;
    FiredStatusCommunicator& operator=(const FiredStatusCommunicator&) = default;

    FiredStatusCommunicator(FiredStatusCommunicator&&) = delete;
    FiredStatusCommunicator& operator=(FiredStatusCommunicator&&) = delete;

    virtual ~FiredStatusCommunicator() = default;

    virtual void finalize() { }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or if init(...) has been called before
     */
    virtual void init(const number_neurons_type number_neurons) {
        RelearnException::check(number_local_neurons == 0, "FiredStatusCommunicator::init: Was already initialized");
        RelearnException::check(number_neurons > 0, "FiredStatusCommunicator::init: Cannot initialize with 0 neurons");

        number_local_neurons = number_neurons;
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    virtual void create_neurons(const number_neurons_type creation_count) {
        RelearnException::check(number_local_neurons > 0, "FiredStatusCommunicator::create_neurons: Was not previously initialized");
        RelearnException::check(creation_count > 0, "FiredStatusCommunicator::create_neurons: Cannot create 0 neurons");

        const auto old_size = number_local_neurons;
        const auto new_size = old_size + creation_count;

        number_local_neurons = new_size;
    }

    /**
     * @brief Sets the extra infos
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) { // NOLINT(performance-unnecessary-value-param) - moved into extra_infos below
        const auto is_filled = new_extra_info != nullptr;
        RelearnException::check(is_filled, "FiredStatusCommunicator::set_extra_infos: new_extra_info is empty");
        extra_infos = std::move(new_extra_info);
    }

    /**
     * @brief Sets the fired status recorder. It is used to determine which neurons fired at the current time
     * @param new_fired_status_recorder The new fired status recorder, must not be empty
     * @exception Throws a RelearnException if new_fired_status_recorder is empty
     */
    void set_fired_status_recorder(std::shared_ptr<FiredStatusRecorder> new_fired_status_recorder) {
        const auto is_filled = new_fired_status_recorder != nullptr;
        RelearnException::check(is_filled, "FiredStatusCommunicator::set_fired_status_recorder: new_fired_status_recorder is empty");
        fired_status_recorder = std::move(new_fired_status_recorder);
    }

    /**
     * @brief Sets the network graph. It is used to determine which neurons to notify in case of a firing one.
     * @param new_network_graph The new network graph, must not be empty
     * @exception Throws a RelearnException if new_network_graph is empty
     */
    void set_network_graph(std::shared_ptr<NetworkGraph> new_network_graph) { // NOLINT(performance-unnecessary-value-param) - moved into network_graph below
        const auto is_filled = new_network_graph != nullptr;
        RelearnException::check(is_filled, "FiredStatusCommunicator::set_network_graph: new_network_graph is empty");
        network_graph = std::move(new_network_graph);
    }

    /**
     * @brief Returns the stored fired status recorder
     * @return The fired status recorder
     */
    [[nodiscard]] const std::shared_ptr<FiredStatusRecorder>& get_fired_status_recorder() const noexcept {
        return fired_status_recorder;
    }

    /**
     * @brief Returns the stored network graph
     * @return The network graph
     */
    [[nodiscard]] const std::shared_ptr<NetworkGraph>& get_network_graph() const noexcept {
        return network_graph;
    }

    /**
     * @brief Checks if the communicator contains the specified neuron of the rank,
     *      i.e., whether that neuron fired in the last update step.
     * @param rank The MPI rank that owns the neuron
     * @param neuron_id The neuron in question
     * @exception Can throw a RelearnException
     */
    [[nodiscard]] virtual bool contains(mpiPP::MPIRank rank, NeuronID neuron_id) const = 0;

    /**
     * @brief Notifies this class and the input calculators that the plasticity has changed.
     *      Some might cache values, which than can be recalculated
     * @param step The current simulation step
     */
    virtual void notify_of_plasticity_change([[maybe_unused]] const step_type step) {
    }

    /**
     * @brief Registers the fired status of the local neurons that are not disabled.
     * @param step The current update step
     * @exception Can throw a RelearnException
     */
    virtual void commit_local_fired_status(step_type step) = 0;

    /**
     * @brief Exchanges the fired status with all MPI ranks
     * @param step The current update step
     * @exception Can throw a RelearnException
     */
    virtual void exchange_fired_status(step_type step) = 0;

    virtual void wait_for_exchange_to_finish() = 0;

    /**
     * @brief Returns the number of MPI ranks
     * @return The number of MPI ranks
     */
    [[nodiscard]] int get_number_ranks() const noexcept {
        return number_ranks;
    }

    [[nodiscard]] mpiPP::MPIRank get_my_rank() const noexcept {
        return my_rank;
    }

    /**
     * @brief Returns the number of local neurons
     * @return The number of local neurons
     */
    [[nodiscard]] number_neurons_type get_number_local_neurons() const noexcept {
        return number_local_neurons;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        const auto my_footprint = sizeof(*this);
        footprint->emplace("FiredStatusCommunicator", my_footprint);
    }

    [[nodiscard]] virtual std::unique_ptr<FireStatusCommunicatorHandle> get_handle() const {
        return nullptr;
    }

protected:
    std::shared_ptr<NeuronsExtraInfo> extra_infos;
    std::shared_ptr<FiredStatusRecorder> fired_status_recorder;
    std::shared_ptr<NetworkGraph> network_graph;

private:
    int number_ranks{ 0 };
    mpiPP::MPIRank my_rank;
    number_neurons_type number_local_neurons{ 0 };
};
