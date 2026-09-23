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

#include "BarnesHutCUDACell.h"
#include "Config.h"
#include "LinearizedTree.h"

#include "algorithm/Algorithm.h"
#include "algorithm/AlgorithmEnum.h"
#include "algorithm/CombinedAlgorithmsInternal/RequestEnums.h"
#include "algorithm/ForwardGPUAlgorithm.h"
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "algorithm/Internal/OctreeAlgorithm.h"
#include "cuda/CudaTypes.h"
#include "cuda/util/LinearizedTreeDeviceHandle.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/SpaceFillingCurve.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "types/SpaceTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#ifdef RELEARN_CUDA_ENABLED
#include "cuda/algorithm/BarnesHutInternalCUDA/BarnesHutCUDA_CU.h"
#endif

/**
 * This class represents the implementation and adaptation of the Barnes-Hut algorithm. The parameters can be set on the fly.
 * In this instance, axons search for dendrites.
 * It is strongly tied to Octree, and might perform MPI communication via NodeCache::get_children()
 */
class BarnesHutCUDA : public ForwardGPUAlgorithm<DistantNeuronRequest, DistantNeuronResponse>, private OctreeAlgorithm<BarnesHutCUDACell> {
public:
    using AdditionalCellAttributes = BarnesHutCUDACell;
    using position_type = RelearnTypes::position_type;
    using counter_type = RelearnTypes::counter_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using acceptance_criterion_type = RelearnTypes::acceptance_criterion_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param bounding_box The bounding box of the whole simulation
     * @param _space_filling_curve The space-filling curve to use, not nullptr
     * @param theta The acceptance criterion for cells in the tree, must be > 0.0. Default is Constants::bh_default_theta
     * @exception Throws a RelearnException if octree is nullptr
     */
    BarnesHutCUDA(const RelearnTypes::bounding_box_type& bounding_box, std::shared_ptr<SpaceFillingCurve> _space_filling_curve,
                  const acceptance_criterion_type theta = Constants::bh_default_theta)
        : OctreeAlgorithm(bounding_box, std::move(_space_filling_curve), false)
        , acceptance_criterion(theta)
        , exc_stream(std::make_shared<StreamWrapper>())
        , inh_stream(std::make_shared<StreamWrapper>()) {
        RelearnException::check(theta > acceptance_criterion_type{ 0 }, "BarnesHutCUDA::BarnesHutCUDA: acceptance_criterion was less than or equal to 0 ({})", theta);
    }

    ~BarnesHutCUDA() override = default;

    BarnesHutCUDA(const BarnesHutCUDA&) = delete;
    BarnesHutCUDA& operator=(const BarnesHutCUDA&) = delete;
    BarnesHutCUDA(BarnesHutCUDA&&) = default;
    BarnesHutCUDA& operator=(BarnesHutCUDA&&) = delete;

    /**
     * @brief Initialize linearized representation of octree, copy linearized tree to GPU
     */
    void init_octree();

    void update_remote_nodes();

    /**
     * @brief Free allocated memory on GPU
     */
    void free();

    /**
     * @brief Sets the acceptance criterion for cells in the tree
     * @param new_acceptance_criterion The acceptance criterion, > 0.0
     * @exception Throws a RelearnException if acceptance_criterion <= 0.0
     */
    void set_acceptance_criterion(acceptance_criterion_type new_acceptance_criterion);

    /**
     * @brief Returns the currently used acceptance criterion
     * @return The currently used acceptance criterion
     */
    [[nodiscard]] acceptance_criterion_type get_acceptance_criterion() const noexcept {
        return acceptance_criterion;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(ForwardCPUAlgorithm<SynapseCreationRequest, SynapseCreationResponse>);
        footprint->emplace("BarnesHut", my_footprint);

        footprint->emplace("BarnesHut GPU", total_mem_usage);

        ForwardGPUAlgorithm<DistantNeuronRequest, DistantNeuronResponse>::record_memory_footprint(footprint);
    }

    void set_neuron_extra_infos(std::shared_ptr<NeuronsExtraInfo> infos) override { // NOLINT(performance-unnecessary-value-param) - part of a virtual override family; signature must match across ~11 overrides
        ForwardGPUAlgorithm::set_neuron_extra_infos(infos);
        OctreeAlgorithm::set_neuron_extra_infos(infos);
    }

    /**
     * @brief Initializes the algorithm to include number_neurons many local neurons.
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(const number_neurons_type number_neurons) override;

    [[nodiscard]] std::tuple<Algorithm::ResultType, RequestTypeEnum, DirectionEnum> find_target_neurons_for_combined_algorithms([[maybe_unused]] const std::vector<NeuronID>& neuron_ids) override {
        RelearnException::fail("Algorithm::find_target_neurons_for_combined_algorithms: Unimplemented find_target_neurons_for_combined_algorithms!");
    }

    [[nodiscard]] AlgorithmEnum get_algorithm_type() const override {
        return AlgorithmEnum::BarnesHutCuda;
    }

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */

    target_neuron_both_result_type find_target_neurons(number_neurons_type number_neurons) override;

    ProcessRequestsAwareResult
    process_requests(const std::vector<int>& counts_full, const std::vector<int>& offsets_full, const DeviceArray<CudaConfig::number_neurons_type>& source_ids, const DeviceArray<SimpleVec3d>& source_positions, const DeviceArray<CudaConfig::number_neurons_type>& target_ids,
                     const SignalType dendrite_type_needed) override;

    void
    process_responses(const DeviceArray<SynapseCreationResponse>& responses, const DeviceArray<CudaConfig::number_neurons_type>& source_ids, const DeviceArray<CudaConfig::number_neurons_type>& target_ids, const std::vector<int>& sizes, const std::vector<int>& offset, SignalType signal_type) override;

    /**
     * @brief Update linearized representation of octree with grown axonal and dendritic elements, updating all necessary information
     */
    void update_linearized_tree();

    /**
     * @brief Update octree on GPU and copy details back to linearized tree on host
     */
    void prepare_update_connectivity(std::span<const SignalType> signal_types,
                                     std::span<const counter_type> vacant_axons,
                                     std::span<const counter_type> vacant_excitatory_dendrites,
                                     std::span<const counter_type> vacant_inhibitory_dendrites) override;

private:
    acceptance_criterion_type acceptance_criterion{ Constants::bh_default_theta };
    // The following members are only read/written from the RELEARN_CUDA_ENABLED branches of
    // BarnesHutCUDA.cpp; every other member function is a CUDA_NOT_SUPPORTED stub, so a
    // CUDA-disabled build never touches them. Only clang has -Wunused-private-field (GCC has
    // no equivalent check), and GCC's [[maybe_unused]] support on a member with a default
    // member initializer is inconsistent across versions (rejected outright by GCC 11, only
    // accepted after the declarator by GCC 12+), so silence it for clang specifically instead
    // of fighting attribute placement across GCC versions.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif
    LinearizedTree linearized_tree{ 0 }; // initialize linearized tree empty
    // Owns the tree's/populations' device buffers (see LinearizedTreeDeviceHandle.h) -- replaces
    // what used to be individual cudaMalloc'd raw pointers, each freed by hand in free(). Move-
    // assigning a freshly-returned storage object over these in init_octree() (which rebuilds the
    // tree from scratch each time it's called) now frees the previous buffers automatically.
    LinearizedTreeDeviceStorage tree_storage{};
    NeuronPopulationDeviceStorage population_ex_storage{};
    NeuronPopulationDeviceStorage population_inh_storage{};
    // Rebuilt from scratch on every prepare_update_connectivity() call (see the helper lambda
    // there); move-assigning a freshly-built DeviceArray over these frees the previous mapping
    // automatically, replacing what used to be manual cudaMallocAsync_bridge/cudaFreeAsync_bridge.
    std::optional<DeviceArray<CudaConfig::bh_index_type>> vacant_exc_axons_mapping;
    std::optional<DeviceArray<CudaConfig::bh_index_type>> vacant_inh_axons_mapping;
    const std::shared_ptr<StreamWrapper> exc_stream;
    const std::shared_ptr<StreamWrapper> inh_stream;

    std::size_t total_mem_usage{};

    bool linearized_tree_initialized{ false };
    CudaConfig::gaussian_type squared_sigma_inv{};
    std::uint64_t counter{};
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
};