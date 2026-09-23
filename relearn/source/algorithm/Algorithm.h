#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/CombinedAlgorithmsInternal/RequestEnums.h"
#include "algorithm/Kernel/KernelBase.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <tuple>
#include <variant>

class NetworkGraph;

/** Synapses created by a single Algorithm::update_connectivity() call, to be committed to the network graph. */
struct ConnectivityUpdateResult {
    std::uint64_t number_created_synapses{};
    PlasticLocalSynapses local_synapses;
    PlasticDistantInSynapses distant_in_synapses;
    PlasticDistantOutSynapses distant_out_synapses;
};

/**
 * This is a virtual interface for all algorithms that can be used to create new synapses.
 * It provides Algorithm::update_connectivity and Algorithm::prepare_update_connectivity.
 */
class Algorithm {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using counter_type = RelearnTypes::counter_type;

    Algorithm() = default;

    Algorithm(const Algorithm&) = delete;
    Algorithm& operator=(const Algorithm&) = delete;

    Algorithm(Algorithm&&) = default;
    Algorithm& operator=(Algorithm&&) = default;

    virtual ~Algorithm() = default;

    /**
     * @brief Sets the kernel for the algorithm
     * @param _kernel The kernel, not empty
     * @exception Throws a RelearnException if _kernel is empty
     */
    void set_probability_kernel(std::unique_ptr<KernelBase> _kernel) {
        const auto kernel_full = _kernel != nullptr;
        RelearnException::check(kernel_full, "Algorithm::set_probability_kernel: kernel was empty");

        kernel = std::move(_kernel);
    }

    /**
     * @brief Registers the synaptic elements with the algorithm
     * @param _synaptic_elements The model for the synaptic elements, not empty
     * @exception Throws a RelearnException if _synaptic_elements is empty
     */
    void set_synaptic_elements(std::shared_ptr<SynapticElements> _synaptic_elements) {
        const auto synaptic_elements_full = _synaptic_elements != nullptr;
        RelearnException::check(synaptic_elements_full, "Algorithm::set_synaptic_elements: synaptic_elements was empty");

        synaptic_elements = std::move(_synaptic_elements);
    }

    /**
     * @brief Sets the extra infos for the neurons. They hold the positions and update flags for the neurons.
     * @param infos The extra infos, not empty
     * @exception Throws a RelearnException if infos is empty
     */
    virtual void set_neuron_extra_infos(std::shared_ptr<NeuronsExtraInfo> infos) {
        RelearnException::check(infos != nullptr, "Algorithm::set_neuron_extra_infos: infos is empty");
        extra_infos = std::move(infos);
    }

    /**
     * @brief Sets the network graph (used in some algorithms to guide the creation of synapses
     * @param graph The network graph, not empty
     * @exception Throws a RelearnException if graph is empty
     */
    virtual void set_network_graph(std::shared_ptr<NetworkGraph> graph) {
        RelearnException::check(graph != nullptr, "Algorithm::set_network_graph: network_graph is empty");
        network_graph = std::move(graph);
    }

    /**
     * @brief Initializes the algorithm to include number_neurons many local neurons.
     * @param number_neurons The number of local neurons to store in this class
     */
    virtual void init([[maybe_unused]] const number_neurons_type number_neurons) { }

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    virtual void create_neurons([[maybe_unused]] const number_neurons_type creation_count) { }

    /**
     * @brief Performs all required steps to disable all neurons that are specified.
     *      Disables incrementally, i.e., previously disabled neurons are not enabled.
     * @param neuron_ids The local neuron ids that should be disabled
     * @exception Throws a RelearnException if a specified id is too large
     */
    virtual void disable_neurons([[maybe_unused]] const std::span<const NeuronID> neuron_ids) { }

    /**
     * @brief Updates the octree according to the necessities of the algorithm. Updates only those neurons for which the extra infos specify so.
     *      May perform communication via MPI
     * @param signal_types The signal types of the neurons
     * @param vacant_axons The vacant axons
     * @param vacant_excitatory_dendrites The vacant excitatory dendrites
     * @param vacant_inhibitory_dendrites The vacant inhibitory dendrites
     * @exception Can throw a RelearnException
     */
    virtual void prepare_update_connectivity(const std::span<const SignalType> signal_types,
                                             const std::span<const counter_type> vacant_axons,
                                             const std::span<const counter_type> vacant_excitatory_dendrites,
                                             const std::span<const counter_type> vacant_inhibitory_dendrites)
        = 0;

    /**
     * @brief Updates the connectivity with the algorithm. Already updates the synaptic elements, i.e., the axons and dendrites (both excitatory and inhibitory).
     *      Does not update the network graph. Performs communication with MPI
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return A tuple with the created synapses that must be committed to the network graph
     */
    [[nodiscard]] virtual ConnectivityUpdateResult update_connectivity(number_neurons_type number_neurons) = 0;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        const auto my_footprint = sizeof(*this);
        footprint->emplace("Algorithm", my_footprint);
    }

    using ResultType = std::variant<RelearnTypes::comm_map_creation<SynapseCreationRequest>, RelearnTypes::comm_map_creation<DistantNeuronRequest>>;
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron. Used when using multiple algorithms by CombinedAlgorithms::update_connectivity
     * @param neuron_ids The neuron_ids that should find targets
     * @exception Can throw a RelearnException (especially when this is called but the method is not implemented in the algorithm itself)
     * @return a tuple containing: - a map wrapped in a variant, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks. the map can be of type mpiPP::CommunicationMap<SynapseCreationRequest> or
     *                               mpiPP::CommunicationMap<DistantNeuronRequest>
     *                             - the request type of the map, returned as an element from an enum
     *                             - the direction that the algorithm uses, i.e. forward (from axons to dendrites) or backward (from dendrites to axons)
     */
    [[nodiscard]] virtual std::tuple<ResultType, RequestTypeEnum, DirectionEnum> find_target_neurons_for_combined_algorithms([[maybe_unused]] const std::vector<NeuronID>& neuron_ids) {
        RelearnException::fail("Algorithm::find_target_neurons_for_combined_algorithms: Unimplemented find_target_neurons_for_combined_algorithms!");
    }

    [[nodiscard]] KernelBase* get_kernel() const {
        return kernel.get();
    }

    [[nodiscard]] const std::shared_ptr<SynapticElements>& get_synaptic_elements() const {
        return synaptic_elements;
    }

    [[nodiscard]] const std::shared_ptr<NetworkGraph>& get_network_graph() const {
        return network_graph;
    }

    [[nodiscard]] const std::shared_ptr<NeuronsExtraInfo>& get_extra_infos() const {
        return extra_infos;
    }

    [[nodiscard]] virtual AlgorithmEnum get_algorithm_type() const = 0;

protected:
    std::shared_ptr<KernelBase> kernel{};                  // NOLINT
    std::shared_ptr<SynapticElements> synaptic_elements{}; // NOLINT
    std::shared_ptr<NetworkGraph> network_graph{};         // NOLINT
    std::shared_ptr<NeuronsExtraInfo> extra_infos{};       // NOLINT
};
