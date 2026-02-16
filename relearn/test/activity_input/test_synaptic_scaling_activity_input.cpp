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
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/input/SynapticScalingActivityInput.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/fired_status_communicator/fired_status_communicator_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "test_activity_input.h"

TEST_F(SynapticScalingActivityInputTest, testConstructorThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_THROW_NO_PRINT(std::ignore = SynapticScalingActivityInput({}), RelearnException);
}

TEST_F(SynapticScalingActivityInputTest, testConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);

    ASSERT_NO_THROW(std::ignore = SynapticScalingActivityInput(fired_status_comm));
}

TEST_F(SynapticScalingActivityInputTest, testInitAndCreate) {
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

    auto synaptic_activity_input = SynapticScalingActivityInput(fired_status_comm);

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
    ASSERT_EQ(synaptic_activity_input.get_scales().size(), number_neurons_init);

    synaptic_activity_input.create_neurons(number_neurons_create_1);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(synaptic_activity_input.get_scales().size(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(0), RelearnException);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(synaptic_activity_input.get_scales().size(), number_neurons_init + number_neurons_create_1);

    synaptic_activity_input.create_neurons(number_neurons_create_2);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(synaptic_activity_input.get_scales().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(synaptic_activity_input.init(0), RelearnException);

    ASSERT_EQ(synaptic_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(synaptic_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(synaptic_activity_input.get_scales().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(SynapticScalingActivityInputTest, testFootprintNoThrow) {
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

    auto synaptic_activity_input = SynapticScalingActivityInput(fired_status_comm);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(synaptic_activity_input.record_memory_footprint(footprint));
}

TEST_F(SynapticScalingActivityInputTest, testSetExtraInfoThrow) {
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

    auto synaptic_activity_input = SynapticScalingActivityInput(fired_status_comm);

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

TEST_F(SynapticScalingActivityInputTest, testUpdateInputEmptyNetworkGraph) {
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

    auto synaptic_activity_input = SynapticScalingActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    synaptic_activity_input.update_input(102);

    for (const auto scale : synaptic_activity_input.get_scales()) {
        ASSERT_EQ(scale, 1.0);
    }

    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        ASSERT_EQ(synaptic_activity_input.get_input(neuron_id), 0.0);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
    }
}

TEST_F(SynapticScalingActivityInputTest, testUpdateInputFullNetworkGraph) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto synaptic_activity_input = SynapticScalingActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        fired_status_recorder->set_fired(neuron_id, FiredStatus::Fired);
    }

    synaptic_activity_input.update_input(102);

    auto total_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);
    auto scewed_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);

    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto& [plastic_edges, _1] = network_graph->get_local_in_edges(neuron_id);
        for (const auto& [_, weight] : plastic_edges) {
            scewed_weights[neuron_id] += weight;
            total_weights[neuron_id] += std::abs(weight);
        }
    }

    const auto scales = synaptic_activity_input.get_scales();
    for (auto i = 0U; i < scales.size(); i++) {
        const auto scale = scales[i];
        const auto total = total_weights[i];

        ASSERT_EQ(scale, total == 0 ? 1.0 : 1.0 / static_cast<double>(total));
    }

    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto id = neuron_id.get_neuron_id();
        const auto scale = scales[id];
        const auto expected = static_cast<double>(scewed_weights[id]) * scale;

        ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected, eps);
    }
}

TEST_F(SynapticScalingActivityInputTest, testUpdateInputRangeFullNetworkGraph) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank(), this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto synaptic_activity_input = SynapticScalingActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        fired_status_recorder->set_fired(neuron_id, FiredStatus::Fired);
    }

    synaptic_activity_input.update_input_range(102, first_neuron_id, last_neuron_id);

    auto total_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);
    auto scewed_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);

    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto& [plastic_edges, _1] = network_graph->get_local_in_edges(neuron_id);
        for (const auto& [_, weight] : plastic_edges) {
            scewed_weights[neuron_id] += weight;
            total_weights[neuron_id] += std::abs(weight);
        }
    }

    const auto scales = synaptic_activity_input.get_scales();
    for (auto i = 0U; i < scales.size(); i++) {
        const auto scale = scales[i];
        const auto total = total_weights[i];

        ASSERT_EQ(scale, total == 0 ? 1.0 : 1.0 / static_cast<double>(total));
    }

    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto id = neuron_id.get_neuron_id();
        const auto scale = scales[id];
        const auto expected = static_cast<double>(scewed_weights[id]) * scale;

        if (first_neuron_id <= neuron_id && neuron_id < last_neuron_id) {
            ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected, eps);
            ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected, eps);
        } else {
            ASSERT_EQ(synaptic_activity_input.get_input(neuron_id), 0.0);
            ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
        }
    }
}

TEST_F(SynapticScalingActivityInputTest, testUpdateInputMultipleRangesFullNetworkGraph) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank(), this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons_init);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    fired_status_comm->init(number_neurons_init);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto synaptic_activity_input = SynapticScalingActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        fired_status_recorder->set_fired(neuron_id, FiredStatus::Fired);
    }

    synaptic_activity_input.update_input_range(105, first_neuron_id, last_neuron_id);
    synaptic_activity_input.update_input_range(105, NeuronID{ 0 }, first_neuron_id);
    synaptic_activity_input.update_input_range(105, last_neuron_id, NeuronID{ number_neurons_init });

    auto total_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);
    auto scewed_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);

    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto& [plastic_edges, _1] = network_graph->get_local_in_edges(neuron_id);
        for (const auto& [_, weight] : plastic_edges) {
            scewed_weights[neuron_id] += weight;
            total_weights[neuron_id] += std::abs(weight);
        }
    }

    const auto scales = synaptic_activity_input.get_scales();
    for (auto i = 0U; i < scales.size(); i++) {
        const auto scale = scales[i];
        const auto total = total_weights[i];

        ASSERT_EQ(scale, total == 0 ? 1.0 : 1.0 / static_cast<double>(total));
    }

    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto id = neuron_id.get_neuron_id();
        const auto scale = scales[id];
        const auto expected = static_cast<double>(scewed_weights[id]) * scale;

        ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected, eps);
    }
}

TEST_F(SynapticScalingActivityInputTest, testUpdateInputPartialNetworkGraph) {
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

    auto synaptic_activity_input = SynapticScalingActivityInput(fired_status_comm);

    synaptic_activity_input.init(number_neurons_init);
    synaptic_activity_input.set_network_graph(network_graph);
    synaptic_activity_input.set_extra_infos(neurons_extra_info);

    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        fired_status_recorder->set_fired(neuron_id, FiredStatus::Fired);
    }

    synaptic_activity_input.update_input(102);

    auto total_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);
    auto scewed_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);

    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto& [plastic_edges, _1] = network_graph->get_local_in_edges(neuron_id);
        for (const auto& [_, weight] : plastic_edges) {
            scewed_weights[neuron_id] += weight;
            total_weights[neuron_id] += std::abs(weight);
        }
    }

    const auto scales = synaptic_activity_input.get_scales();
    for (auto i = 0U; i < scales.size(); i++) {
        const auto scale = scales[i];
        const auto total = total_weights[i];

        ASSERT_EQ(scale, total == 0 ? 1.0 : 1.0 / static_cast<double>(total));
    }

    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto id = neuron_id.get_neuron_id();
        const auto scale = scales[id];
        const auto expected = static_cast<double>(scewed_weights[id]) * scale;

        ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected, eps);
    }
}

TEST_F(SynapticScalingActivityInputTest, testUpdateInputSomeFired) {
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

    auto synaptic_activity_input = SynapticScalingActivityInput(fired_status_comm);

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

    auto expected_input = std::vector<double>(number_neurons_init, 0.0);
    auto total_possible_input = std::vector<double>(number_neurons_init, 0.0);

    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto& [plastic_edges, _1] = network_graph->get_local_in_edges(neuron_id);
        for (const auto& [source, weight] : plastic_edges) {
            total_possible_input[neuron_id] += weight;

            if (fired_status[source.get_neuron_id()] == FiredStatus::Inactive) {
                continue;
            }

            expected_input[neuron_id] += weight;
        }
    }

    synaptic_activity_input.update_input(102);

    auto total_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);
    auto scewed_weights = std::vector<RelearnTypes::plastic_synapse_weight>(number_neurons_init, 0);

    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto& [plastic_edges, _1] = network_graph->get_local_in_edges(neuron_id);
        for (const auto& [target_id, weight] : plastic_edges) {
            total_weights[neuron_id] += std::abs(weight);

            if (fired_status[target_id.get_neuron_id()] == FiredStatus::Inactive) {
                continue;
            }

            scewed_weights[neuron_id] += weight;
        }
    }

    const auto scales = synaptic_activity_input.get_scales();
    for (auto i = 0U; i < scales.size(); i++) {
        const auto scale = scales[i];
        const auto total = total_weights[i];

        ASSERT_EQ(scale, total == 0 ? 1.0 : 1.0 / static_cast<double>(total));
    }

    const auto actual_input = synaptic_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto id = neuron_id.get_neuron_id();
        const auto scale = scales[id];
        const auto expected = static_cast<double>(scewed_weights[id]) * scale;

        ASSERT_NEAR(synaptic_activity_input.get_input(neuron_id), expected, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected, eps);
    }
}
