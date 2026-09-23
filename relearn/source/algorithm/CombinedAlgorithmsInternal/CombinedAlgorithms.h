#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/Algorithm.h"
#include "algorithm/AlgorithmEnum.h"
#include "algorithm/BarnesHutInternal/BarnesHut.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/BarnesHutInternal/BarnesHutInverted.h"
#include "algorithm/BarnesHutInternal/BarnesHutLocationAware.h"
#include "algorithm/CombinedAlgorithmsInternal/RequestEnums.h"
#include "algorithm/Connector.h"
#include "algorithm/FMMInternal/FastMultipoleMethod.h"
#include "algorithm/FMMInternal/FastMultipoleMethodBase.h"
#include "algorithm/NaiveInternal/Naive.h"
#include "io/NeuronToAlgorithmIO.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "structure/SpaceFillingCurve.h"
#include "types/AlgorithmTypes.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "types/SpaceTypes.h"
#include "types/SynapseTypes.h"
#include "util/Timers.h"

#include <mpi-wrapper/patterns/CommunicationMap.h>
#include <mpi-wrapper/patterns/MPIAdvancedCommunicationPatterns.h>

#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <variant>

/**
 * @brief This is the algorithm that is chosen when multiple algorithms should be used. It is used to initialize the used algorithms
 *      properly and invoke the methods used to find target neurons in each algorithm. It manages different types of requests (forward/backward,
 *      SynapseCreationRequest/DistantNeuronRequest). Supported algorithms are: naive, barnes-hut, barnes-hut-inverted, barnes-hut-location-aware
 */
class CombinedAlgorithms : public Algorithm {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using counter_type = RelearnTypes::counter_type;
    using acceptance_criterion_type = RelearnTypes::acceptance_criterion_type;

    /**
     * @brief Constructs a new CombinedAlgorithms object
     * @param bounding_box The bounding_box that the algorithms should use
     * @param _space_filling_curve The space filling curve that the algorithms should use
     * @param algorithm_configs The algorithm configs, i.e. the algorithm types and their kernels to use and their theta if one is given
     * @param indices_and_neurons The vector indicating which neurons use which algorithms, where the indices refer to the algorithm_configs vector
     * @param theta The acceptance criterion to use in barnes-hut type algorithms, where none is given in the config itself
     */
    CombinedAlgorithms(const RelearnTypes::bounding_box_type& bounding_box, const std::shared_ptr<SpaceFillingCurve>& _space_filling_curve,
                       RelearnTypes::AlgorithmConfigs&& algorithm_configs, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - moved into _algorithm_configs below; checker doesn't recognize the move
                       const RelearnTypes::AlgorithmIndexWithNeuronsType& indices_and_neurons, const acceptance_criterion_type theta = Constants::bh_default_theta)
        : Algorithm()
        , _algorithm_configs{ std::move(algorithm_configs) }
        , _indices_and_neurons{ indices_and_neurons } {
        initialize_algorithms(bounding_box, _space_filling_curve, theta);
    }

    /**
     * @brief Updates the octree according to the necessities of the algorithm. Updates only those neurons for which the extra infos specify so.
     *      May perform communication via MPI. Prepares the connectivity update for every algorithm used.
     * @param signal_types The signal types of the neurons
     * @param vacant_axons The vacant axons
     * @param vacant_excitatory_dendrites The vacant excitatory dendrites
     * @param vacant_inhibitory_dendrites The vacant inhibitory dendrites
     * @exception Can throw a RelearnException
     */
    void prepare_update_connectivity(const std::span<const SignalType> signal_types,
                                     const std::span<const counter_type> vacant_axons,
                                     const std::span<const counter_type> vacant_excitatory_dendrites,
                                     const std::span<const counter_type> vacant_inhibitory_dendrites) override {
        for (const auto& alg_ptr : algorithm_ptrs) {
            alg_ptr->prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites);
        }
    }

    /**
     * @brief Updates the connectivity with the algorithm. Already updates the synaptic elements, i.e., the axons and dendrites (both excitatory and inhibitory).
     *      Does not update the network graph. Performs communication with MPI. Uses the algorithms themselves to find target neurons for specified neurons.
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return A tuple with the created synapses that must be committed to the network graph
     */
    [[nodiscard]] ConnectivityUpdateResult update_connectivity(const number_neurons_type number_neurons) override;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        Algorithm::record_memory_footprint(footprint);

        const auto my_footprint = sizeof(*this) - sizeof(Algorithm);
        footprint->emplace("CombinedAlgorithms", my_footprint);
    }

    /**
     * @brief Sets the extra infos for the neurons. They hold the positions and update flags for the neurons. Does this for every algorithm.
     * @param infos The extra infos, not empty
     * @exception Throws a RelearnException if infos is empty
     */
    void set_neuron_extra_infos(std::shared_ptr<NeuronsExtraInfo> infos) override {
        for (const auto& alg_ptr : algorithm_ptrs) {
            alg_ptr->set_neuron_extra_infos(infos);
        }
        extra_infos = infos;
    }

    /**
     * @brief Initializes the algorithm to include number_neurons many local neurons.
     *      Initializes important elements of the algorithms.
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(const number_neurons_type number_neurons) override {
        for (auto& alg_ptr : algorithm_ptrs) {
            alg_ptr->set_synaptic_elements(synaptic_elements);
            alg_ptr->set_network_graph(network_graph);
            alg_ptr->init(number_neurons);
        }
    }

    /**
     * @brief Returns the vector of algorithm pointers.
     * @return The vector of algorithm pointers
     */
    [[nodiscard]] std::vector<std::shared_ptr<Algorithm>> get_algorithm_pointers() const {
        return algorithm_ptrs;
    }

    /**
     * @brief Returns the algorithm_configs by moving them, so the vector is empty afterwards.
     * @return The algorithm_configs of the combined algorithms
     */
    [[nodiscard]] RelearnTypes::AlgorithmConfigs transfer_algorithm_configs() {
        return std::move(_algorithm_configs);
    }

    [[nodiscard]] AlgorithmEnum get_algorithm_type() const override {
        return AlgorithmEnum::CombinedAlgorithms;
    }

protected:
    /**
     * @brief Processes all incoming requests that go from axons to dendrites from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from axons to dendrites from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] ForwardProcessRequestsResult<SynapseCreationResponse>
    process_requests_forward(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests);

    /**
     * @brief Processes all incoming responses that correspond to requests that go from axons to dendrites from the MPI ranks locally
     * @param creation_requests The requests from axons to dendrites from this MPI rank
     * @param creation_responses The responses from the other MPI ranks corresponding to requests going from axons to dendrites
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] PlasticDistantOutSynapses process_responses_forward(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                                                      const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses);

    /**
     * @brief Processes all incoming requests that go from dendrites to axons from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from dendrites to axons from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] BackwardProcessRequestsResult<SynapseCreationResponse>
    process_requests_backward(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests);

    /**
     * @brief Processes all incoming responses that correspond to requests that go from dendrites to axons from the MPI ranks locally
     * @param creation_requests The requests from dendrites to axons from this MPI rank
     * @param creation_responses The responses from the other MPI ranks corresponding to requests going from dendrites to axons
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] PlasticDistantInSynapses process_responses_backward(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                                                      const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses);

    /**
     * @brief Initializes the vector of algorithm pointers so that the algorithms are constructed and reachable
     * @param bounding_box The bounding box that the algorithms use
     * @param _space_filling_curve The space filling curve
     * @param theta The acceptance criterion for barnes-hut algorithms
     */
    void initialize_algorithms(const RelearnTypes::bounding_box_type& bounding_box, const std::shared_ptr<SpaceFillingCurve>& _space_filling_curve, const acceptance_criterion_type theta) {
        for (auto& algorithm_config : _algorithm_configs) {
            const auto algorithm = algorithm_config.get_algorithm_type();
            const auto opt_theta = algorithm_config.get_theta();
            const auto theta_to_use = opt_theta.value_or(theta);
            switch (algorithm) {
            case AlgorithmEnum::Naive:
                algorithm_ptrs.push_back(std::make_shared<Naive>(bounding_box, _space_filling_curve));
                break;
            case AlgorithmEnum::BarnesHut:
                algorithm_ptrs.push_back(std::make_shared<BarnesHut>(bounding_box, _space_filling_curve, theta_to_use));
                break;
            case AlgorithmEnum::BarnesHutInverted:
                algorithm_ptrs.push_back(std::make_shared<BarnesHutInverted>(bounding_box, _space_filling_curve, theta_to_use));
                break;
            case AlgorithmEnum::BarnesHutLocationAware:
                algorithm_ptrs.push_back(std::make_shared<BarnesHutLocationAware>(bounding_box, _space_filling_curve, theta_to_use));
                break;
            default:
                RelearnException::fail("Algorithm {} not yet implemented in CombinedAlgorithms", algorithm);
            }
            algorithm_ptrs[algorithm_ptrs.size() - 1]->set_probability_kernel(std::move(algorithm_config).transfer_kernel());
        }
    }

private:
    RelearnTypes::AlgorithmConfigs _algorithm_configs{};
    std::vector<std::shared_ptr<Algorithm>> algorithm_ptrs{};
    RelearnTypes::AlgorithmIndexWithNeuronsType _indices_and_neurons{};
};