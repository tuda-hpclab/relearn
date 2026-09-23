/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NaiveCUDA.h"

#include "algorithm/Connector.h"
#include "algorithm/Internal/OctreeAlgorithm.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/NaiveInternalCUDA/NaiveCUDACell.h"
#include "cuda/CudaTypes.h"
#include "cuda/algorithm/NaiveInternalCUDA/NaiveCUDA_CU.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "types/CommunicationTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/Random.h"
#include "util/RandomHolderKey.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/ranges/Functional.hpp>

#include <mpi-wrapper/core/MPIInfo.h>

#include <range/v3/view/filter.hpp>

#include <algorithm>
#include <tuple>
#include <vector>

void NaiveCUDA::init(const number_neurons_type number_neurons) {
    OctreeAlgorithm::init(number_neurons);
    device_neuron_positions = init_cuda_positions(number_neurons);
}

DeviceArray<SimpleVec3d> NaiveCUDA::init_cuda_positions(const number_neurons_type number_neurons) const {
    auto neuron_pos = std::vector<SimpleVec3d>(number_neurons);

    // map neuron positions from extra_infos to CudaTypes::cuda_real3 vector
    const auto simulation_neuron_pos = extra_infos->get_positions();
    std::ranges::transform(simulation_neuron_pos, neuron_pos.begin(),
                           [](const position_type& pos) { return SimpleVec3d{ utility::cast<double>(pos.get_x()), utility::cast<double>(pos.get_y()), utility::cast<double>(pos.get_z()) }; });

    return DeviceArray<SimpleVec3d>(neuron_pos);
}

RelearnTypes::comm_map_creation<SynapseCreationRequest>
NaiveCUDA::find_target_neurons(number_neurons_type number_neurons) {
    if (device_neuron_positions.device_ptr() == nullptr) {
        RelearnException::fail("Neuron positions not available on CUDA device.");
    }

    const auto gaussian_kernel = std::dynamic_pointer_cast<GaussianDistributionKernel>(kernel);
    RelearnException::check(gaussian_kernel != nullptr, "BarnesHutCUDA::init: Cuda supports only the Gaussian kernel");
    const auto sigma = gaussian_kernel->get_sigma();
    RelearnException::check(sigma > 0, "BarnesHutCUDA::init: Sigma must be > 0");
    const auto squared_sigma_inv = RelearnTypes::acceptance_criterion_type{ 1 } / (sigma * sigma);

    const auto& disable_flags = extra_infos->get_disable_flags();
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<number_neurons_type>(number_ranks), number_neurons);
    auto synapse_creation_requests_outgoing = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks,
                                                                                                      size_hint);

    // provide Arrays for saving excitatory axons, dendrites and target IDs for later connection
    auto vacant_axons_ex = std::vector<RelearnTypes::counter_type>(number_neurons);
    auto vacant_dendrites_ex = std::vector<RelearnTypes::counter_type>(number_neurons);
    auto start_index_ex = std::vector<std::uint64_t>(number_neurons);
    auto vacant_axon_neuron_id_mapping_ex = std::vector<RelearnTypes::counter_type>(0);

    // // provide Arrays for saving inhibitory axons, dendrites and target IDs for later connection
    auto vacant_axons_inh = std::vector<RelearnTypes::counter_type>(number_neurons);
    auto vacant_dendrites_inh = std::vector<RelearnTypes::counter_type>(number_neurons);
    auto start_index_inh = std::vector<std::uint64_t>(number_neurons);
    auto vacant_axon_neuron_id_mapping_inh = std::vector<RelearnTypes::counter_type>(0);

    auto sum_vacant_axons = 0UL;
    auto start_index_ctr_ex = 0UL;
    auto start_index_ctr_inh = 0UL;

    // Distribute Neurons into the previously provided arrays
    const auto& vacant_dendex = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto& vacant_dendinh = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);
    const auto& vacant_axon = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    for (const auto id : NeuronIDRange::range(number_neurons) | ranges::views::filter(utility::equal_to(UpdateStatus::Enabled), utility::lookup(disable_flags, &NeuronID::get_neuron_id))) {
        const auto i = id.get_neuron_id();
        RelearnException::check(i < vacant_dendex.size(), "NaiveCUDA::find_target_neurons: Id {} is too large {}", i,
                                vacant_dendex.size());
        RelearnException::check(i < vacant_dendinh.size(), "NaiveCUDA::find_target_neurons: Id {} is too large {}", i,
                                vacant_dendinh.size());
        RelearnException::check(i < vacant_axon.size(), "NaiveCUDA::find_target_neurons: Id {} is too large {}", i,
                                vacant_axon.size());

        // distribute vacant dendrites in their respective array
        vacant_dendrites_ex[i] = vacant_dendex[id.get_neuron_id()];
        vacant_dendrites_inh[i] = vacant_dendinh[id.get_neuron_id()];

        // if there are no vacant axons for a specific neuron, there is nothing to do
        const auto number_vacant_axons = vacant_axon[id.get_neuron_id()];
        if (number_vacant_axons == 0) {
            start_index_ex[i] = start_index_ctr_ex;
            start_index_ctr_ex++;
            start_index_inh[i] = start_index_ctr_inh;
            start_index_ctr_inh++;
            continue;
        }

        sum_vacant_axons += number_vacant_axons;
        const auto& signal_types = synaptic_elements->get_signal_types();

        // distribute vacant axons in their respective array
        if (signal_types[id.get_neuron_id()] == SignalType::Excitatory) {
            start_index_ctr_inh++;
            start_index_ex[i] = start_index_ctr_ex;
            start_index_ctr_ex += number_vacant_axons > 0 ? number_vacant_axons : 1;

            vacant_axons_ex[i] = number_vacant_axons;

            for (auto j = 0U; j < number_vacant_axons; ++j) {
                vacant_axon_neuron_id_mapping_ex.emplace_back(i);
            }
        }
        if (signal_types[id.get_neuron_id()] == SignalType::Inhibitory) {
            start_index_ctr_ex++;
            start_index_inh[i] = start_index_ctr_inh;
            start_index_ctr_inh += number_vacant_axons > 0 ? number_vacant_axons : 1;

            vacant_axons_inh[i] = number_vacant_axons;

            for (auto j = 0U; j < number_vacant_axons; ++j) {
                vacant_axon_neuron_id_mapping_inh.emplace_back(i);
            }
        }
    }

    // If there is only no axon available for connection, there is nothing to calculate
    if (sum_vacant_axons < 1) {
        return synapse_creation_requests_outgoing;
    }

    const auto target_size_ex = vacant_axon_neuron_id_mapping_ex.size();
    const auto target_size_inh = vacant_axon_neuron_id_mapping_inh.size();
    // target neuron vector
    auto target_neurons_ex = std::vector<uint64_t>(target_size_ex);

    auto target_neurons_inh = std::vector<uint64_t>(target_size_inh);

    // run find target neurons for both excitatory and inhibitory neurons
    if (!vacant_axon_neuron_id_mapping_ex.empty()) {
        // Preprocess random numbers
        auto random_nums_ex = std::vector<double>(target_size_ex);

        RandomHolder::fill(RandomHolderKey::Algorithm, std::begin(random_nums_ex), std::end(random_nums_ex), CudaTypes::cuda_attraction{ 0 }, CudaTypes::cuda_attraction{ 1 });

        NaiveCUDA_CU::find_target_neurons(device_neuron_positions, number_neurons,
                                          NaiveCUDA_CU::NaiveTargetSelectionTask{ vacant_axon_neuron_id_mapping_ex, vacant_dendrites_ex, random_nums_ex, target_size_ex },
                                          target_neurons_ex, utility::cast<double>(squared_sigma_inv));
    }
    if (!vacant_axon_neuron_id_mapping_inh.empty()) {
        // Preprocess random numbers
        auto random_nums_inh = std::vector<double>(target_size_inh);

        RandomHolder::fill(RandomHolderKey::Algorithm, std::begin(random_nums_inh), std::end(random_nums_inh), CudaTypes::cuda_attraction{ 0 }, CudaTypes::cuda_attraction{ 1 });

        NaiveCUDA_CU::find_target_neurons(device_neuron_positions, number_neurons,
                                          NaiveCUDA_CU::NaiveTargetSelectionTask{ vacant_axon_neuron_id_mapping_inh, vacant_dendrites_inh, random_nums_inh, target_size_inh },
                                          target_neurons_inh, utility::cast<double>(squared_sigma_inv));
    }

    for (auto i = 0U; i < vacant_axon_neuron_id_mapping_ex.size(); ++i) {
        const auto found_target = target_neurons_ex[i];
        const auto source_neuron = vacant_axon_neuron_id_mapping_ex[i];
        if (source_neuron != found_target) {
            // When using MPI, this is invalid:
            const auto target_rank = mpiPP::MPIInfo::get_my_rank();
            const auto creation_request = SynapseCreationRequest(NeuronID(found_target), NeuronID(source_neuron),
                                                                 SignalType::Excitatory);
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }
    for (auto i = 0U; i < vacant_axon_neuron_id_mapping_inh.size(); ++i) {
        const auto found_target = target_neurons_inh[i];
        const auto source_neuron = vacant_axon_neuron_id_mapping_inh[i];
        if (source_neuron != found_target) {
            // When using MPI, this is invalid:
            const auto target_rank = mpiPP::MPIInfo::get_my_rank();
            const auto creation_request = SynapseCreationRequest(NeuronID(found_target), NeuronID(source_neuron),
                                                                 SignalType::Inhibitory);
            synapse_creation_requests_outgoing.append(target_rank, creation_request);
        }
    }

    return synapse_creation_requests_outgoing;
}

std::tuple<bool, bool> NaiveCUDA::acceptance_criterion_test(const position_type& /*axon_position*/,
                                                            const OctreeNode<NaiveCUDACell>* const node_with_dendrite,
                                                            const SignalType dendrite_type_needed) {
    RelearnException::check(node_with_dendrite != nullptr,
                            "Naive::acceptance_criterion_test: node_with_dendrite was nullptr");

    const auto& cell = node_with_dendrite->get_cell();
    const auto has_vacant_dendrites = cell.get_number_dendrites_for(dendrite_type_needed) != 0;
    const auto is_parent = node_with_dendrite->is_parent();

    // Accept leaf only
    return std::make_tuple(!is_parent, has_vacant_dendrites);
}

ForwardProcessRequestsResult<SynapseCreationResponse>
NaiveCUDA::process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, synaptic_elements);
}

PlasticDistantOutSynapses
NaiveCUDA::process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                             const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, synaptic_elements);
}