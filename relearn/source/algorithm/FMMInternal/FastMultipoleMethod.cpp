/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FastMultipoleMethod.h"

#include "Config.h"
#include "Types.h"
#include "Types3.h"

#include "algorithm/Connector.h"
#include "algorithm/FMMInternal/FastMultipoleMethodBase.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Kernel/Gaussian.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include <algorithm>
#include <utility>

RelearnTypes::comm_map_creation<SynapseCreationRequest> FastMultipoleMethod::find_target_neurons(const number_neurons_type number_neurons) {
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();

    const auto size_hint = std::min(static_cast<number_neurons_type>(number_ranks), number_neurons);
    auto synapse_creation_requests_outgoing = RelearnTypes::comm_map_creation<SynapseCreationRequest>(number_ranks, size_hint);

    const auto& global_tree = get_octree();
    const auto& node_cache = global_tree->get_cache();

    auto* root = get_octree_root();
    RelearnException::check(root != nullptr, "FastMultipoleMethod::find_target_neurons: root was nullptr");

    // Get number of dendrites
    const auto total_number_dendrites_ex = root->get_cell().get_number_excitatory_dendrites();
    const auto total_number_dendrites_in = root->get_cell().get_number_inhibitory_dendrites();

    const auto& local_branch_nodes = get_octree()->get_local_branch_nodes();
    const auto branch_level = get_level_of_branch_nodes();

    const auto* probability_kernel_ptr = this->kernel.get();
    const auto* gaussian_ptr = dynamic_cast<const GaussianDistributionKernel*>(probability_kernel_ptr);

    RelearnException::check(gaussian_ptr != nullptr, "FastMultipoleMethod::find_target_neurons: kernel was not Gaussian");

    if (total_number_dendrites_ex > 0) {
        FastMultipoleMethodBase::make_creation_request_for(gaussian_ptr->get_sigma(), node_cache, root, local_branch_nodes, branch_level, ElementType::Axon, SignalType::Excitatory, Constants::unpacking + 1, synapse_creation_requests_outgoing);
    }
    if (total_number_dendrites_in > 0) {
        FastMultipoleMethodBase::make_creation_request_for(gaussian_ptr->get_sigma(), node_cache, root, local_branch_nodes, branch_level, ElementType::Axon, SignalType::Inhibitory, Constants::unpacking + 1, synapse_creation_requests_outgoing);
    }

    // Stop Timer and make cache empty for next connectivity update
    Timers::start(TimerRegion::EMPTY_REMOTE_NODES_CACHE);
    global_tree->clear_cache();
    Timers::stop_and_add(TimerRegion::EMPTY_REMOTE_NODES_CACHE);

    return synapse_creation_requests_outgoing;
}

std::pair<RelearnTypes::comm_map_creation<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
FastMultipoleMethod::process_requests(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests) {
    return ForwardConnector::process_requests(creation_requests, synaptic_elements);
}

PlasticDistantOutSynapses FastMultipoleMethod::process_responses(const RelearnTypes::comm_map_creation<SynapseCreationRequest>& creation_requests,
                                                                 const RelearnTypes::comm_map_creation<SynapseCreationResponse>& creation_responses) {
    return ForwardConnector::process_responses(creation_requests, creation_responses, synaptic_elements);
}
