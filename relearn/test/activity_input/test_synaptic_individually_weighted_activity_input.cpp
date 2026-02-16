/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"

#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusCommunicationMap.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/helper/Synapse.h"
#include "neurons/input/SynapticIndividuallyWeightedActivityInput.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/fired_status_communicator/fired_status_communicator_factory.h"
#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"
#include "factory/synapses/synapses_factory.h"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "test_activity_input.h"

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testConstructorThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = SynapticIndividuallyWeightedActivityInput({}), RelearnException);
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);

    ASSERT_NO_THROW(std::ignore = SynapticIndividuallyWeightedActivityInput(fired_status_comm));
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_1 = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_2 = NeuronIdFactory::get_random_number_neurons(this->mt);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm);

    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(0), RelearnException);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), 0);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), 0);

    synaptic_activity_input.init(number_neurons_init);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(0), RelearnException);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init);

    synaptic_activity_input.create_neurons(number_neurons_create_1);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(0), RelearnException);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    synaptic_activity_input.create_neurons(number_neurons_create_2);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(0), RelearnException);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testFootprintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init);
    fired_status_comm->set_network_graph(network_graph);

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(synaptic_activity_input.record_memory_footprint(footprint));
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testSetExtraInfoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init + 1);

    const auto neurons_extra_info_too_large = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_large->init(number_neurons_init + 2);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init + 1);

    const auto neurons_extra_info_too_small = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_small->init(number_neurons_init + 0);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init + 1);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init + 1);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init + 1);
    synaptic_activity_input.set_network_graph(network_graph);

    ASSERT_THROW_NO_PRINT(synaptic_activity_input.set_extra_infos({}), RelearnException);

    synaptic_activity_input.set_extra_infos(neurons_extra_info_too_large);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.update_input(102), RelearnException);

    synaptic_activity_input.set_extra_infos(neurons_extra_info_too_small);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.update_input(102), RelearnException);

    synaptic_activity_input.set_extra_infos(neurons_extra_info);
    ASSERT_NO_THROW(synaptic_activity_input.update_input(102));
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testUpdateInputEmptyNetworkGraph) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    // set empty weight map
    SynapticIndividuallyWeightedActivityInput::weight_map_type weight_map{};

    synaptic_activity_input.update_input(102);
    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        ASSERT_EQ(synaptic_activity_input.get_input(neuron_id), 0.0);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
    }
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testUpdateInputFullNetworkGraph) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = 82;
    const auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank(), this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    // get rank
    const auto rank = mpiPP::MPIInfo::get_my_rank();

    // create weight map
    SynapticIndividuallyWeightedActivityInput::weight_map_type weight_map{};

    auto expected_input = std::vector<double>(number_neurons_init, 0.0);
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto& [plastic_edges, _] = network_graph->get_local_in_edges(neuron_id.get_neuron_id());
        fired_status_recorder->set_fired(neuron_id, FiredStatus::Fired);
        for (const auto& [source_neuron_id, synapse_strength] : plastic_edges) {
            auto source_neuron_id_with_rank = RankNeuronId{ rank, source_neuron_id };
            auto pair = std::make_pair(neuron_id.get_neuron_id(), source_neuron_id_with_rank);

            const auto is_inhibitory = synapse_strength < 0;
            const auto synapse_strength_abs = std::abs(synapse_strength);
            for (auto i = 0; i < synapse_strength_abs; i++) {
                auto random_weight = RandomFactory::get_random_double<SynapticIndividuallyWeightedActivityInput::weight_type>(0.1, 10.0, this->mt);
                if (is_inhibitory) {
                    random_weight = -random_weight;
                }
                weight_map[pair].push_back(random_weight);
                expected_input[neuron_id.get_neuron_id()] += random_weight;
            }
        }
    }

    synaptic_activity_input.set_weight_map(std::move(weight_map));

    synaptic_activity_input.update_input(102);
    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_value = expected_input[neuron_id.get_neuron_id()];

        ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected_value, Constants::eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_value, Constants::eps);
    }
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testUpdateInputPartialNetworkGraph) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = 51;
    const auto network_graph = NetworkGraphFactory::construct_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank(), 8, this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    // get rank
    const auto rank = mpiPP::MPIInfo::get_my_rank();

    // create weight map
    SynapticIndividuallyWeightedActivityInput::weight_map_type weight_map{};

    auto expected_input = std::vector<double>(number_neurons_init, 0.0);
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto& [plastic_edges, _] = network_graph->get_local_in_edges(neuron_id.get_neuron_id());
        fired_status_recorder->set_fired(neuron_id, FiredStatus::Fired);
        for (const auto& [source_neuron_id, synapse_strength] : plastic_edges) {
            auto source_neuron_id_with_rank = RankNeuronId{ rank, source_neuron_id };
            auto pair = std::make_pair(neuron_id.get_neuron_id(), source_neuron_id_with_rank);

            const auto is_inhibitory = synapse_strength < 0;
            const auto synapse_strength_abs = std::abs(synapse_strength);
            for (auto i = 0; i < synapse_strength_abs; i++) {
                auto random_weight = RandomFactory::get_random_double<SynapticIndividuallyWeightedActivityInput::weight_type>(0.1, 10.0, this->mt);
                if (is_inhibitory) {
                    random_weight = -random_weight;
                }
                weight_map[pair].push_back(random_weight);
                expected_input[neuron_id.get_neuron_id()] += random_weight;
            }
        }
    }

    synaptic_activity_input.set_weight_map(std::move(weight_map));

    synaptic_activity_input.update_input(102);
    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_value = expected_input[neuron_id.get_neuron_id()];

        ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected_value, Constants::eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_value, Constants::eps);
    }
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testUpdateInputSomeFired) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt) + 10;
    const auto network_graph = NetworkGraphFactory::construct_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank(), 8, this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    auto fired_status = std::vector<FiredStatus>(number_neurons_init, FiredStatus::Inactive);
    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto fired = RandomFactory::get_random_bool(this->mt);
        if (fired) {
            fired_status[neuron_id] = FiredStatus::Fired;
            fired_status_recorder->set_fired(NeuronID(neuron_id), FiredStatus::Fired);
        }
    }

    // get rank
    const auto rank = mpiPP::MPIInfo::get_my_rank();

    // create weight map
    SynapticIndividuallyWeightedActivityInput::weight_map_type weight_map{};

    auto expected_input = std::vector<double>(number_neurons_init, 0.0);
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto& [plastic_edges, _] = network_graph->get_local_in_edges(neuron_id.get_neuron_id());
        for (const auto& [source_neuron_id, synapse_strength] : plastic_edges) {
            auto source_neuron_id_with_rank = RankNeuronId{ rank, source_neuron_id };
            auto pair = std::make_pair(neuron_id.get_neuron_id(), source_neuron_id_with_rank);

            const auto is_inhibitory = synapse_strength < 0;
            const auto synapse_strength_abs = std::abs(synapse_strength);
            for (auto i = 0; i < synapse_strength_abs; i++) {
                auto random_weight = RandomFactory::get_random_double<SynapticIndividuallyWeightedActivityInput::weight_type>(0.1, 10.0, this->mt);
                if (is_inhibitory) {
                    random_weight = -random_weight;
                }
                weight_map[pair].push_back(random_weight);
                if (fired_status[source_neuron_id.get_neuron_id()] == FiredStatus::Fired) {
                    expected_input[neuron_id.get_neuron_id()] += random_weight;
                }
            }
        }
    }

    synaptic_activity_input.set_weight_map(std::move(weight_map));

    synaptic_activity_input.update_input(102);
    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_value = expected_input[neuron_id.get_neuron_id()];

        ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected_value, Constants::eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_value, Constants::eps);
    }
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testUpdateInputWrongNumberWeightsThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }
    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt) + 10;
    const auto network_graph = NetworkGraphFactory::construct_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank(), 8, this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    const auto rank = mpiPP::MPIInfo::get_my_rank();

    SynapticIndividuallyWeightedActivityInput::weight_map_type weight_map{};

    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto& [plastic_edges, _] = network_graph->get_local_in_edges(neuron_id.get_neuron_id());
        fired_status_recorder->set_fired(neuron_id, FiredStatus::Fired);
        for (const auto& [source_neuron_id, synapse_strength] : plastic_edges) {
            auto source_neuron_id_with_rank = RankNeuronId{ rank, source_neuron_id };
            auto pair = std::make_pair(neuron_id.get_neuron_id(), source_neuron_id_with_rank);

            const auto is_inhibitory = synapse_strength < 0;
            const auto synapse_strength_abs = std::abs(synapse_strength);
            const auto number_weights = RandomFactory::get_random_integer(1, 10, synapse_strength_abs, this->mt);
            for (auto i = 0; i < number_weights; i++) {
                auto random_weight = RandomFactory::get_random_double<SynapticIndividuallyWeightedActivityInput::weight_type>(0.1, 10.0, this->mt);
                if (is_inhibitory) {
                    random_weight = -random_weight;
                }
                weight_map[pair].push_back(random_weight);
            }
        }
    }

    synaptic_activity_input.set_weight_map(std::move(weight_map));

    ASSERT_THROW_NO_PRINT(synaptic_activity_input.update_input(102), RelearnException);
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testDistantInput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt) + 10;
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(this->mt) + 1;

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank(), number_ranks);

    const auto number_synapses = SynapsesFactory::get_random_number_synapses(this->mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(this->mt) + 10;

    const auto distant_in_synapses = SynapsesFactory::generate_plastic_distant_in_synapses(number_neurons_init, number_synapses, number_ranks, number_foreign_neurons, this->mt);

    PlasticDistantInSynapses::const_iterator const_iter;
    for (const_iter = distant_in_synapses.begin(); const_iter != distant_in_synapses.end(); const_iter++) {
        network_graph->add_synapse(*const_iter);
    }

    auto distant_neurons_that_fired = std::set<RankNeuronId>{};
    for (const auto& [_1, source, _2] : distant_in_synapses) {
        distant_neurons_that_fired.insert(source);
    }

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicationMap(number_ranks);
    fired_status_comm.init(number_neurons_init);
    fired_status_comm.set_network_graph(network_graph);
    fired_status_comm.set_fired_status_recorder(fired_status_recorder);
    fired_status_comm.set_incoming_ids(distant_neurons_that_fired);
    auto fired_status_comm_ptr = std::make_shared<FiredStatusCommunicationMap>(fired_status_comm);

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm_ptr);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    // create weight map
    SynapticIndividuallyWeightedActivityInput::weight_map_type weight_map{};

    auto expected_input = std::vector<double>(number_neurons_init, 0.0);
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto& [plastic_edges_distant, _] = network_graph->get_distant_in_edges(neuron_id.get_neuron_id());
        for (const auto& [key, synapse_strength] : plastic_edges_distant) {
            auto pair = std::make_pair(neuron_id.get_neuron_id(), key);

            const auto is_inhibitory = synapse_strength < 0;
            const auto synapse_strength_abs = std::abs(synapse_strength);
            for (auto i = 0; i < synapse_strength_abs; i++) {
                auto random_weight = RandomFactory::get_random_double<SynapticIndividuallyWeightedActivityInput::weight_type>(0.1, 10.0, this->mt);
                if (is_inhibitory) {
                    random_weight = -random_weight;
                }
                weight_map[pair].push_back(random_weight);
                expected_input[neuron_id.get_neuron_id()] += random_weight;
            }
        }
    }

    synaptic_activity_input.set_weight_map(std::move(weight_map));

    synaptic_activity_input.update_input(102);
    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_value = expected_input[neuron_id.get_neuron_id()];

        ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected_value, Constants::eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_value, Constants::eps);
    }
}

TEST_F(SynapticIndividuallyWeightedActivityInputTest, testLocalAndDistantInputsPartialNetworkGraphSomeFired) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt) + 10;
    const auto number_ranks = MPIRankFactory::get_random_number_ranks(this->mt) + 1;

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank(), number_ranks);

    const auto number_local_synapses = SynapsesFactory::get_random_number_synapses(this->mt);

    const auto local_synapses = SynapsesFactory::generate_plastic_local_synapses(number_neurons_init, number_local_synapses, this->mt);
    PlasticLocalSynapses::const_iterator local_const_iter;
    for (local_const_iter = local_synapses.begin(); local_const_iter != local_synapses.end(); local_const_iter++) {
        network_graph->add_synapse(*local_const_iter);
    }

    const auto number_distant_in_synapses = SynapsesFactory::get_random_number_synapses(this->mt);
    const auto number_foreign_neurons = NeuronIdFactory::get_random_number_neurons(this->mt) + 10;

    const auto distant_in_synapses = SynapsesFactory::generate_plastic_distant_in_synapses(number_neurons_init, number_distant_in_synapses, number_ranks, number_foreign_neurons, this->mt);

    PlasticDistantInSynapses::const_iterator distant_const_iter;
    for (distant_const_iter = distant_in_synapses.begin(); distant_const_iter != distant_in_synapses.end(); distant_const_iter++) {
        network_graph->add_synapse(*distant_const_iter);
    }

    auto distant_neurons_that_fired = std::set<RankNeuronId>{};
    for (const auto& [_1, source, _2] : distant_in_synapses) {
        const auto fired = RandomFactory::get_random_bool(this->mt);
        if (fired) {
            distant_neurons_that_fired.insert(source);
        }
    }

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicationMap(number_ranks);
    fired_status_comm.init(number_neurons_init);
    fired_status_comm.set_network_graph(network_graph);
    fired_status_comm.set_fired_status_recorder(fired_status_recorder);
    fired_status_comm.set_incoming_ids(distant_neurons_that_fired);
    auto fired_status_comm_ptr = std::make_shared<FiredStatusCommunicationMap>(fired_status_comm);

    auto fired_status = std::vector<FiredStatus>(number_neurons_init, FiredStatus::Inactive);
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto fired = RandomFactory::get_random_bool(this->mt);
        if (fired) {
            fired_status_recorder->set_fired(neuron_id, FiredStatus::Fired);
            fired_status[neuron_id.get_neuron_id()] = FiredStatus::Fired;
        }
    }

    auto synaptic_activity_input = SynapticIndividuallyWeightedActivityInput(fired_status_comm_ptr);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    // get rank
    const auto rank = mpiPP::MPIInfo::get_my_rank();

    // create weight map
    SynapticIndividuallyWeightedActivityInput::weight_map_type weight_map{};

    auto expected_input = std::vector<double>(number_neurons_init, 0.0);
    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto& [plastic_edges_local, _1] = network_graph->get_local_in_edges(neuron_id);
        const auto& [plastic_edges_distant, _2] = network_graph->get_distant_in_edges(neuron_id);
        for (const auto& [source, synapse_strength] : plastic_edges_local) {
            const auto source_neuron_with_rank = RankNeuronId{ rank, source };
            auto pair = std::make_pair(neuron_id, source_neuron_with_rank);

            const auto is_inhibitory = synapse_strength < 0;
            const auto synapse_strength_abs = std::abs(synapse_strength);
            for (auto i = 0; i < synapse_strength_abs; i++) {
                auto random_weight = RandomFactory::get_random_double<SynapticIndividuallyWeightedActivityInput::weight_type>(0.1, 10.0, this->mt);
                if (is_inhibitory) {
                    random_weight = -random_weight;
                }
                weight_map[pair].push_back(random_weight);
                if (fired_status[source.get_neuron_id()] == FiredStatus::Fired) {
                    expected_input[neuron_id] += random_weight;
                }
            }
        }
        for (const auto& [key, synapse_strength] : plastic_edges_distant) {
            auto pair = std::make_pair(neuron_id, key);

            const auto is_inhibitory = synapse_strength < 0;
            const auto synapse_strength_abs = std::abs(synapse_strength);
            for (auto i = 0; i < synapse_strength_abs; i++) {
                auto random_weight = RandomFactory::get_random_double<SynapticIndividuallyWeightedActivityInput::weight_type>(0.1, 10.0, this->mt);
                if (is_inhibitory) {
                    random_weight = -random_weight;
                }
                weight_map[pair].push_back(random_weight);
                if (fired_status_comm_ptr->contains(key.get_rank(), key.get_neuron_id())) {
                    expected_input[neuron_id] += random_weight;
                }
            }
        }
    }

    synaptic_activity_input.set_weight_map(std::move(weight_map));

    synaptic_activity_input.update_input(102);
    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_value = expected_input[neuron_id.get_neuron_id()];

        ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected_value, Constants::eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_value, Constants::eps);
    }
}