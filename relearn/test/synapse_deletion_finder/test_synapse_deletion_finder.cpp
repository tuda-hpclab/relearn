/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_synapse_deletion_finder.h"

#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/synaptic_elements/Axons.h"
#include "neurons/synaptic_elements/Dendrites.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"

#include "factory/network_graph/network_graph_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

TEST_F(SynapseDeletionFinderTest, testRandom) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }
}

// End-to-end regression test of RandomSynapseDeletionFinder::delete_synapses(): a single local
// axon-to-dendrite synapse whose growth has shrunk below its connected-element count must be
// found, exchanged (a no-op with 1 MPI rank), and committed -- disconnecting both the axon side
// (via SynapticElements::commit_updates(), which the finder calls internally) and the paired
// dendrite side (via SynapseDeletionFinder::commit_deletions()).
TEST_F(SynapseDeletionFinderTest, testDeleteSynapsesRemovesShrunkenAxonConnection) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 2 };
    constexpr auto axon_owner = RelearnTypes::number_neurons_type{ 0 };
    constexpr auto dendrite_owner = RelearnTypes::number_neurons_type{ 1 };

    auto extra_info = std::make_shared<NeuronsExtraInfo>();
    extra_info->init(number_neurons);
    extra_info->set_positions({ RelearnTypes::position_type{ 0.0, 0.0, 0.0 }, RelearnTypes::position_type{ 1.0, 1.0, 1.0 } });

    const auto signal_types = std::vector<SignalType>(number_neurons, SignalType::Excitatory);

    auto synaptic_elements = std::make_shared<SynapticElements>(std::make_shared<Axons>(), std::make_shared<Dendrites>());
    synaptic_elements->init(number_neurons);
    synaptic_elements->set_signal_types(signal_types);
    synaptic_elements->set_extra_infos(extra_info);

    // The axon owner has exactly one connected (and grown) excitatory axon, and the dendrite
    // owner has exactly one connected (and grown) excitatory dendrite -- both sides of the same
    // physical synapse, which is registered in the network graph below.
    synaptic_elements->add_connected_elements(1, axon_owner, SynapticElementType::Axon);
    synaptic_elements->add_connected_elements(1, dendrite_owner, SynapticElementType::DendriteExcitatory);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);
    // PlasticLocalSynapse(target, source, weight); positive weight <=> excitatory source.
    network_graph->add_synapse(PlasticLocalSynapse(NeuronID(dendrite_owner), NeuronID(axon_owner), 1));
    // The factory only synced the (then-empty) graph to the GPU at construction time -- push the
    // synapse just added so the deletion kernels (which read the GPU-side edge storage) see it.
    network_graph->sync_with_gpu();

    // Shrink the axon owner's growth so that commit_updates() must fully disconnect its one
    // connected axon (current_vacant (0) + delta (<= -1) < 0 triggers a full deletion).
    synaptic_elements->add_to_delta(-1.0, axon_owner, SynapticElementType::Axon);

    auto finder = RandomSynapseDeletionFinder();
    finder.set_network_graph(network_graph);
    finder.set_synaptic_elements(synaptic_elements);
    finder.set_extra_infos(extra_info);
    finder.init(number_neurons);

    const auto& [deleted_axons, deleted_dendrites] = finder.delete_synapses();

    ASSERT_EQ(deleted_axons, 1U) << "The one shrunken axon connection must be found and deleted";
    ASSERT_EQ(deleted_dendrites, 0U) << "No dendrite-initiated deletions were requested";

    ASSERT_EQ(synaptic_elements->get_connected_elements(SynapticElementType::Axon)[axon_owner], 0U)
        << "commit_updates() must have fully disconnected the shrunken axon";
    ASSERT_EQ(synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory)[dendrite_owner], 0U)
        << "commit_deletions() must have disconnected the paired dendrite side of the deleted synapse";
}
