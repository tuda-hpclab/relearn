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
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "algorithm/Internal/OctreeAlgorithm.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/NaiveInternalCUDA/NaiveCUDACell.h"
#include "cuda/CudaTypes.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/SpaceFillingCurve.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "types/SpaceTypes.h"
#include "types/SynapseTypes.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <tuple>
#include <utility>

/**
 * This class represents the implementation of the trivial O(n^2) algorithm.
 * It is strongly tied to Octree, and might perform MPI communication via NodeCache::get_children()
 */
class NaiveCUDA : public ForwardCPUAlgorithm<SynapseCreationRequest, SynapseCreationResponse>, private OctreeAlgorithm<NaiveCUDACell> {
public:
    using AdditionalCellAttributes = NaiveCUDACell;
    using position_type = RelearnTypes::position_type;
    using counter_type = RelearnTypes::counter_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param bounding_box The bounding box of the whole simulation
     * @param _space_filling_curve The space-filling curve to use, not nullptr
     * @exception Throws a RelearnException if octree is nullptr
     */
    NaiveCUDA(const RelearnTypes::bounding_box_type& bounding_box, std::shared_ptr<SpaceFillingCurve> _space_filling_curve)
        : OctreeAlgorithm(bounding_box, std::move(_space_filling_curve), true) {
    }

    ~NaiveCUDA() override = default;

    NaiveCUDA(const NaiveCUDA&) = delete;
    NaiveCUDA& operator=(const NaiveCUDA&) = delete;
    NaiveCUDA(NaiveCUDA&&) = default;
    NaiveCUDA& operator=(NaiveCUDA&&) = default;

    /**
     * @brief Allocate and copy initial data to GPU memory, sets global variables needed for calculating target-neurons on device
     * @param number_neurons The number of positions that are allocated space for, > 0
     */
    void init(number_neurons_type number_neurons) override;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(ForwardCPUAlgorithm<SynapseCreationRequest, SynapseCreationResponse>);
        footprint->emplace("NaiveCuda", my_footprint);

        ForwardCPUAlgorithm::record_memory_footprint(footprint);
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
        update_tree(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites);
    }

    void set_neuron_extra_infos(std::shared_ptr<NeuronsExtraInfo> infos) override {
        ForwardCPUAlgorithm::set_neuron_extra_infos(infos);
        OctreeAlgorithm::set_neuron_extra_infos(infos);
    }

    [[nodiscard]] AlgorithmEnum get_algorithm_type() const override {
        return AlgorithmEnum::NaiveCuda;
    }

protected:
    /**
     * @brief Allocate memory and copy position data of all neurons to GPU memory
     * @param number_neurons The number of local neurons
     * @return Pointer of neuron positions on the device
     */
    [[nodiscard]] DeviceArray<SimpleVec3d> init_cuda_positions(number_neurons_type number_neurons) const;

    /* @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */
    [[nodiscard]] RelearnTypes::comm_map_creation<SynapseCreationRequest> find_target_neurons(number_neurons_type number_neurons) override;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] ForwardProcessRequestsResult<SynapseCreationResponse>
    process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) override;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] PlasticDistantOutSynapses process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                                              const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) override;

private:
    DeviceArray<SimpleVec3d> device_neuron_positions{ 0 };

    [[nodiscard]] static std::tuple<bool, bool> acceptance_criterion_test(
        const position_type& axon_position,
        const OctreeNode<NaiveCUDACell>* node_with_dendrite,
        SignalType dendrite_type_needed);
};