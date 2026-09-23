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

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "algorithm/Internal/OctreeAlgorithm.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"

#include <memory>
#include <utility>
#include <vector>

/**
 * This class represents the implementation and adaptation of the FastMultipoleMethodInverted algorithm. The parameters can be set on the fly.
 * It is strongly tied to Octree, which might perform MPI communication via NodeCache::get_children()
 */
class FastMultipoleMethodInverted : public BackwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse>, private OctreeAlgorithm<FastMultipoleMethodCell> {
    friend class FMMTest;

public:
    using AdditionalCellAttributes = FastMultipoleMethodCell;
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using counter_type = RelearnTypes::counter_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param bounding_box The bounding box to use in the octree
     * @param _space_filling_curve The space-filling curve to use, not nullptr
     * @exception Throws a RelearnException if _space_filling_curve is nullptr
     */
    FastMultipoleMethodInverted(const RelearnTypes::bounding_box_type& bounding_box, std::shared_ptr<SpaceFillingCurve> _space_filling_curve) // NOLINT(performance-unnecessary-value-param) - moved into the base-class ctor below
        : OctreeAlgorithm(bounding_box, std::move(_space_filling_curve), true) { }

    ~FastMultipoleMethodInverted() override = default;

    FastMultipoleMethodInverted(const FastMultipoleMethodInverted&) = delete;
    FastMultipoleMethodInverted& operator=(const FastMultipoleMethodInverted&) = delete;
    FastMultipoleMethodInverted(FastMultipoleMethodInverted&&) = default;
    FastMultipoleMethodInverted& operator=(FastMultipoleMethodInverted&&) = default;

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
    [[nodiscard]] const std::shared_ptr<Octree<AdditionalCellAttributes>>& get_octree() {
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
                                     const std::span<const counter_type> vacant_axons,
                                     const std::span<const counter_type> vacant_excitatory_dendrites,
                                     const std::span<const counter_type> vacant_inhibitory_dendrites) override {
        OctreeAlgorithm::update_tree(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites);
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(BackwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse>);
        footprint->emplace("FastMultipoleMethodInverted", my_footprint);

        BackwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse>::record_memory_footprint(footprint);
    }

    [[nodiscard]] AlgorithmEnum get_algorithm_type() const override {
        return AlgorithmEnum::FastMultipoleMethod;
    }

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons.
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank.
     */
    RelearnTypes::comm_map_creation<SynapseCreationRequest> find_target_neurons(number_neurons_type number_neurons) override;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses from the local rank
     */
    [[nodiscard]] BackwardProcessRequestsResult<SynapseCreationResponse>
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
};
