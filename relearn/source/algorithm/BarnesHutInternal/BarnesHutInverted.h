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
#include "Types.h"

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/Algorithm.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "algorithm/CombinedAlgorithmsInternal/RequestEnums.h"
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "algorithm/Internal/OctreeAlgorithm.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/SpaceFillingCurve.h"
#include "util/NeuronID.h"

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

/**
 * This class represents the implementation and adaptation of the Barnes-Hut algorithm.
 * In this instance, dendrites search for axons.
 * It is strongly tied to Octree, which might perform MPI communication via NodeCache::get_children()
 */
class BarnesHutInverted : public BackwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse>, private OctreeAlgorithm<BarnesHutInvertedCell> {
public:
    using AdditionalCellAttributes = BarnesHutInvertedCell;
    using position_type = RelearnTypes::position_type;
    using counter_type = RelearnTypes::counter_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new instance with the given parameter.
     *      Internally constructs an Octree as well.
     * @param bounding_box The bounding box of the whole simulation
     * @param _space_filling_curve The space-filling curve to use, not nullptr
     * @param theta The acceptance criterion for cells in the tree, must be > 0.0. Default is Constants::bh_default_theta
     * @exception Throws a RelearnException if theta <= 0.0 or if _space_filling_curve is nullptr
     */
    BarnesHutInverted(const RelearnTypes::bounding_box_type& bounding_box, std::shared_ptr<SpaceFillingCurve> _space_filling_curve,
                      const double theta = Constants::bh_default_theta)
        : BackwardAlgorithm()
        , OctreeAlgorithm(bounding_box, std::move(_space_filling_curve))
        , acceptance_criterion(theta) {
        RelearnException::check(theta > 0.0, "BarnesHut::BarnesHut: acceptance_criterion was less than or equal to 0 ({})", theta);
    }

    virtual ~BarnesHutInverted() = default;

    /**
     * @brief Sets the extra infos for the neurons. They hold the positions and update flags for the neurons.
     * @param infos The extra infos, not empty
     * @exception throws a RelearnException if infos is empty
     */
    void set_neuron_extra_infos(std::shared_ptr<NeuronsExtraInfo> infos) override {
        BackwardAlgorithm::set_neuron_extra_infos(infos);
        OctreeAlgorithm::set_neuron_extra_infos(infos);
    }

    /**
     * @brief Returns the currently used acceptance criterion
     * @return The currently used acceptance criterion
     */
    [[nodiscard]] double get_acceptance_criterion() const noexcept {
        return acceptance_criterion;
    }

    /**
     * @brief Initializes the algorithm to include number_neurons many local neurons.
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(const number_neurons_type number_neurons) override {
        OctreeAlgorithm::init(number_neurons);
    }

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(const number_neurons_type creation_count) override {
        OctreeAlgorithm::create_neurons(creation_count);
    }

    /**
     * @brief Performs all required steps to disable all neurons that are specified.
     *      Disables incrementally, i.e., previously disabled neurons are not enabled.
     * @param neuron_ids The local neuron ids that should be disabled
     * @exception Throws a RelearnException if a specified id is too large
     */
    void disable_neurons(const std::span<const NeuronID> neuron_ids) override {
        OctreeAlgorithm::disable_neurons(neuron_ids);
    }

    /**
     * @brief Returns the octree that is used by this algorithm
     */
    [[nodiscard]] const std::unique_ptr<Octree<AdditionalCellAttributes>>& get_octree() {
        return OctreeAlgorithm::get_octree();
    }

    /**
     * @brief Updates the octree according to the necessities of the algorithm. Updates only those neurons for which the extra infos specify so.
     *      May perform communication via MPI
     * @param signal_types The signal types of the neurons
     * @param vacant_axons The vacant axons
     * @param vacant_excitatory_dendrites The vacant excitatory dendrites
     * @param vacant_inhibitory_dendrites The vacant inhibitory dendrites
     * @exception Can throw a RelearnException
     */
    void prepare_update_connectivity(const std::span<const SignalType> signal_types,
                                     const std::span<const unsigned int> vacant_axons,
                                     const std::span<const unsigned int> vacant_excitatory_dendrites,
                                     const std::span<const unsigned int> vacant_inhibitory_dendrites) override {
        OctreeAlgorithm::update_tree(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites);
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(BackwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse>);
        footprint->emplace("BarnesHutInverted", my_footprint);

        BackwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse>::record_memory_footprint(footprint);
    }

    /**
    * @brief Returns a collection of proposed synapse creations for each neuron. Used when using multiple algorithms by CombinedAlgorithms::update_connectivity
    * @param neuron_ids The neuron_ids that should find targets
    * @exception Can throw a RelearnException (especially when this is called but the method is not implemented in the algorithm itself)
    * @return a tuple containing: - a map wrapped in a variant, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks. the map can be of type mpiPP::CommunicationMap<SynapseCreationRequest> or
    *                               mpiPP::CommunicationMap<DistantNeuronRequest>
    *                             - the request type of the map, returned as an element from an enum
    *                             - the direction that the algorithm uses, i.e. forward (from axons to dendrites) or backward (from dendrites to axons)
    */
    [[nodiscard]] std::tuple<Algorithm::ResultType, RequestTypeEnum, DirectionEnum> find_target_neurons_for_combined_algorithms(const std::vector<NeuronID>& neuron_ids) override;

    [[nodiscard]] AlgorithmEnum get_algorithm_type() const override {
        return AlgorithmEnum::BarnesHutInverted;
    }
protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] RelearnTypes::comm_map_creation<SynapseCreationRequest> find_target_neurons(number_neurons_type number_neurons) override;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses from the local rank
     */
    [[nodiscard]] std::pair<RelearnTypes::comm_map_creation<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantOutSynapses>>
    process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) override;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses to this MPI rank from other MPI ranks
     */
    [[nodiscard]] PlasticDistantInSynapses process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                                             const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) override;

private:
    double acceptance_criterion{ Constants::bh_default_theta };
};
