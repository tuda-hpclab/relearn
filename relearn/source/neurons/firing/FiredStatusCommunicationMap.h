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

#include "Types.h"
#include "Types3.h"

#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIRank.h"

#include <algorithm>
#include <memory>
#include <set>
#include <span>
#include <vector>

/**
 * This class communicates the fired status of the local neurons
 * via two separate RelearnTypes::comm_map_firing
 */
class FiredStatusCommunicationMap : public FiredStatusCommunicator {
public:
    /**
     * @brief Constructs a new object with the given number of ranks and local neurons (mainly used for pre-allocating memory)
     * @param num_ranks The number of MPI ranks
     * @param size_hint The size hint for the communication maps
     * @exception Throws a RelearnException if num_ranks <= 0
     */
    explicit FiredStatusCommunicationMap(const int num_ranks, const std::size_t size_hint = 1)
        : FiredStatusCommunicator(num_ranks)
        , outgoing_ids(num_ranks, size_hint)
        , incoming_ids(num_ranks, size_hint) {
        RelearnException::check(num_ranks > 0, "FiredStatusCommunicationMap::FiredStatusCommunicationMap: num_ranks is too small: {}", num_ranks);
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
    void create_neurons(const number_neurons_type creation_count) override {
        FiredStatusCommunicator::create_neurons(creation_count);
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
     * @brief Sets the incoming ids manually without mpi. This method should be only used in tests. Use exchange_fired_status(step) in the simulation code.
     * @param rnis Set of RankNeuronIds containing the neurons that fired in the last step
     */
    void set_incoming_ids(const std::set<RankNeuronId>& rnis);

    /**
     * @brief Checks if the communicator contains the specified neuron of the rank,
     *      i.e., whether that neuron fired in the last update step.
     * @param rank The MPI rank that owns the neuron
     * @param neuron_id The neuron in question
     * @exception Throws a RelearnException if rank is not from [0, number_ranks) or the neuron_id is virtual
     */
    bool contains(mpiPP::MPIRank rank, NeuronID neuron_id) const override;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_easy_footprint = sizeof(*this) - sizeof(FiredStatusCommunicator);

        auto my_hard_footprint = std::uint64_t{ 0 };
        for (const auto& rank : mpiPP::MPIRank::range(outgoing_ids.get_number_ranks())) {
            my_hard_footprint += outgoing_ids.get_size_in_bytes(rank) + incoming_ids.get_size_in_bytes(rank);
        }

        footprint->emplace("FiredStatusCommunicationMap", my_hard_footprint + my_easy_footprint);

        FiredStatusCommunicator::record_memory_footprint(footprint);
    }

private:
    RelearnTypes::comm_map_firing<NeuronID> outgoing_ids;
    RelearnTypes::comm_map_firing<NeuronID> incoming_ids;
};
