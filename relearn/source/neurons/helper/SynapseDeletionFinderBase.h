#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <mpi-wrapper/core/MPIRank.h>

#include <memory>
#include <span>
#include <utility>
#include <vector>

class NetworkGraph;

/**
 * This class encapsulates the logic of finding and deleting synapses
 * based on the synaptic elements. It provides the communication via MPI
 * and other house keeping, as well as a virtual method to implement.
 *      Holds everything that is common to the CPU and GPU deletion finders;
 *      SynapseDeletionFinderCPU and SynapseDeletionFinderGPU add the parts that differ (how
 *      synapses to delete are found and deletions are committed).
 */
class SynapseDeletionFinderBase {
public:
    using counter_type = RelearnTypes::counter_type;
    using number_synapse_type = RelearnTypes::number_synapse_type;

    SynapseDeletionFinderBase() = default;

    SynapseDeletionFinderBase(const SynapseDeletionFinderBase&) = default;
    SynapseDeletionFinderBase& operator=(const SynapseDeletionFinderBase&) = default;

    SynapseDeletionFinderBase(SynapseDeletionFinderBase&&) = delete;
    SynapseDeletionFinderBase& operator=(SynapseDeletionFinderBase&&) = delete;

    virtual ~SynapseDeletionFinderBase() = default;

    virtual void init(RelearnTypes::number_neurons_type number_neurons) {
        size = number_neurons;
    }

    /**
     * @brief Sets the network graph that stores the synapses
     * @param ng The new network graph, must not be empty
     * @exception Throws a RelearnException if ng is empty
     */
    void set_network_graph(std::shared_ptr<NetworkGraph> ng) { // NOLINT(performance-unnecessary-value-param) - moved into network_graph below
        const auto full = ng != nullptr;
        RelearnException::check(full, "SynapseDeletionFinderBase::set_network_graph: The network graph is empty");

        network_graph = std::move(ng);
    }

    void set_synaptic_elements(std::shared_ptr<SynapticElements> synaptic_elements_ptr) { // NOLINT(performance-unnecessary-value-param) - moved into synaptic_elements below
        const auto full = synaptic_elements_ptr != nullptr;
        RelearnException::check(full, "SynapseDeletionFinderBase::set_synaptic_elements: The synaptic elements is empty");

        synaptic_elements = std::move(synaptic_elements_ptr);
    }

    /**
     * @brief Sets the extra information
     * @param new_extra_info The extra information, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info);

    /**
     * @brief Sets the fired status recorder
     * @param new_fired_status_recorder The fired status recorder, must not be empty
     * @exception Throws a RelearnException if new_fired_status_recorder is empty
     */
    void set_fired_status_recorder(std::shared_ptr<FiredStatusRecorder> new_fired_status_recorder);

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        const auto my_footprint = sizeof(*this);
        footprint->emplace("SynapseDeletionFinder", my_footprint);
    }

protected:
    [[nodiscard]] virtual std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, counter_type num_synapses_to_delete) = 0;

    [[nodiscard]] std::vector<RankNeuronId> register_synapses(NeuronID neuron_id, ElementType element_type, SignalType signal_type);

    [[nodiscard]] std::vector<mpiPP::MPIRank> build_rank_whitelist() const;

    std::shared_ptr<SynapticElements> synaptic_elements;
    std::shared_ptr<NetworkGraph> network_graph;
    std::shared_ptr<NeuronsExtraInfo> extra_info;
    std::shared_ptr<FiredStatusRecorder> fired_status_recorder;
    RelearnTypes::number_neurons_type size{};
    std::uint32_t random_key{};
};
