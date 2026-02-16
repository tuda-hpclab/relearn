/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_kernel.h"

#include "Types.h"

#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Kernel.h"
#include "neurons/enums/SynapticElementType.h"
#include "util/Random.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "adapter/kernel/KernelAdapter.h"

#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <gtest/gtest.h>

#include <cpp-utility/Cast.hpp>
#include <range/v3/algorithm/contains.hpp>
#include <range/v3/view/indices.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <vector>

TEST_F(KernelTest, testCalculateAttractivenessSameNode) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id = NeuronID{ 423 };
    const auto position = RelearnTypes::position_type{ 13.2, 14.5, 0.2 };

    auto node = OctreeNode<FastMultipoleMethodCell>{};
    node.set_cell_neuron_id(neuron_id);
    node.set_rank(mpiPP::MPIRank::root_rank());

    const auto kernel = GaussianDistributionKernel{};

    const auto attractiveness_1 = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &node, ElementType::Axon, SignalType::Excitatory);
    const auto attractiveness_2 = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &node, ElementType::Axon, SignalType::Inhibitory);
    const auto attractiveness_3 = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &node, ElementType::Dendrite, SignalType::Excitatory);
    const auto attractiveness_4 = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &node, ElementType::Dendrite, SignalType::Inhibitory);

    ASSERT_EQ(attractiveness_1, 0.0);
    ASSERT_EQ(attractiveness_2, 0.0);
    ASSERT_EQ(attractiveness_3, 0.0);
    ASSERT_EQ(attractiveness_4, 0.0);
}

TEST_F(KernelTest, testCalculateAttractivenessExceptionNoPosition) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id_1 = NeuronID{ 423 };
    const auto neuron_id_2 = NeuronID{ 12 };
    const auto source_position = RelearnTypes::position_type{ 13.2, 14.5, 0.2 };

    auto node = OctreeNode<FastMultipoleMethodCell>{};
    node.set_cell_neuron_id(neuron_id_1);
    node.set_cell_size(SimulationFactory::get_minimum_position(), SimulationFactory::get_maximum_position());

    node.set_cell_excitatory_axons_position({});
    node.set_cell_inhibitory_axons_position({});
    node.set_cell_excitatory_dendrites_position({});
    node.set_cell_inhibitory_dendrites_position({});

    node.set_cell_number_axons(0, 0);
    node.set_cell_number_dendrites(0, 0);

    const auto kernel = GaussianDistributionKernel{};

    using type = Kernel<FastMultipoleMethodCell>;
    ASSERT_THROW_NO_PRINT(std::ignore = type::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Axon, SignalType::Excitatory);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = type::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Axon, SignalType::Inhibitory);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = type::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Dendrite, SignalType::Excitatory);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = type::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Dendrite, SignalType::Inhibitory);, RelearnException);

    node.set_cell_number_axons(1, 1);
    node.set_cell_number_dendrites(1, 1);

    ASSERT_THROW_NO_PRINT(std::ignore = type::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Axon, SignalType::Excitatory);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = type::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Axon, SignalType::Inhibitory);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = type::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Dendrite, SignalType::Excitatory);, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = type::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id_2 }, source_position, &node, ElementType::Dendrite, SignalType::Inhibitory);, RelearnException);
}

TEST_F(KernelTest, testCreateProbabilityIntervalEmptyVector) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& neuron_id = NeuronID{ 423 };
    const auto& position = RelearnTypes::position_type{ 13.2, 14.5, 0.2 };

    const auto kernel = GaussianDistributionKernel{};

    const auto& [sum_1, attrs_1] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                                { mpiPP::MPIRank::root_rank(), neuron_id }, position, {}, ElementType::Axon, SignalType::Excitatory);

    const auto& [sum_2, attrs_2] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                                { mpiPP::MPIRank::root_rank(), neuron_id }, position, {}, ElementType::Axon, SignalType::Inhibitory);

    const auto& [sum_3, attrs_3] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                                { mpiPP::MPIRank::root_rank(), neuron_id }, position, {}, ElementType::Dendrite, SignalType::Excitatory);

    const auto& [sum_4, attrs_4] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                                { mpiPP::MPIRank::root_rank(), neuron_id }, position, {}, ElementType::Dendrite, SignalType::Inhibitory);

    ASSERT_EQ(sum_1, 0.0);
    ASSERT_EQ(0, attrs_1.size());

    ASSERT_EQ(sum_2, 0.0);
    ASSERT_EQ(0, attrs_2.size());

    ASSERT_EQ(sum_3, 0.0);
    ASSERT_EQ(0, attrs_3.size());

    ASSERT_EQ(sum_4, 0.0);
    ASSERT_EQ(0, attrs_4.size());
}

TEST_F(KernelTest, testCreateProbabilityIntervalAutapseVector) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id = NeuronID{ 423 };
    const auto position = RelearnTypes::position_type{ 13.2, 14.5, 0.2 };

    const auto number_nodes = NeuronID::value_type{ 23 };

    auto nodes = std::vector<OctreeNode<FastMultipoleMethodCell>>{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    auto node_pointers = std::vector<OctreeNode<FastMultipoleMethodCell>*>{ number_nodes, nullptr };

    for (const auto i : ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(neuron_id);
        nodes[i].set_rank(mpiPP::MPIRank::root_rank());
        nodes[i].set_cell_size(SimulationFactory::get_minimum_position(), SimulationFactory::get_maximum_position());

        const auto& target_excitatory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto& target_inhibitory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto& target_excitatory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto& target_inhibitory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };

        const auto& number_vacant_excitatory_axons = utility::save_cast<RelearnTypes::counter_type>(i);
        const auto& number_vacant_inhibitory_axons = utility::save_cast<RelearnTypes::counter_type>(i + 1);
        const auto& number_vacant_excitatory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 1) / 2;
        const auto& number_vacant_inhibitory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 3) / 3;

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto kernel = GaussianDistributionKernel{};

    const auto& [sum_1, attrs_1] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                                { mpiPP::MPIRank::root_rank(), neuron_id }, position, node_pointers, ElementType::Axon, SignalType::Excitatory);

    const auto& [sum_2, attrs_2] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                                { mpiPP::MPIRank::root_rank(), neuron_id }, position, node_pointers, ElementType::Axon, SignalType::Inhibitory);

    const auto& [sum_3, attrs_3] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                                { mpiPP::MPIRank::root_rank(), neuron_id }, position, node_pointers, ElementType::Dendrite, SignalType::Excitatory);

    const auto& [sum_4, attrs_4] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                                { mpiPP::MPIRank::root_rank(), neuron_id }, position, node_pointers, ElementType::Dendrite, SignalType::Inhibitory);

    ASSERT_EQ(sum_1, 0.0);
    ASSERT_EQ(0, attrs_1.size());

    ASSERT_EQ(sum_2, 0.0);
    ASSERT_EQ(0, attrs_2.size());

    ASSERT_EQ(sum_3, 0.0);
    ASSERT_EQ(0, attrs_3.size());

    ASSERT_EQ(sum_4, 0.0);
    ASSERT_EQ(0, attrs_4.size());
}

TEST_F(KernelTest, testCreateProbabilityIntervalVectorException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id = NeuronID{ 423 };
    const auto position = RelearnTypes::position_type{ 13.2, 14.5, 0.2 };

    const auto number_nodes = NeuronID::value_type{ 23 };

    auto nodes = std::vector<OctreeNode<FastMultipoleMethodCell>>{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    auto node_pointers = std::vector<OctreeNode<FastMultipoleMethodCell>*>{ number_nodes, nullptr };

    for (const auto i : ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronID{ i });
        nodes[i].set_rank(mpiPP::MPIRank::root_rank());
        nodes[i].set_cell_size(SimulationFactory::get_minimum_position(), SimulationFactory::get_maximum_position());

        const auto target_excitatory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_excitatory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };

        const auto number_vacant_excitatory_axons = utility::save_cast<RelearnTypes::counter_type>(i);
        const auto number_vacant_inhibitory_axons = utility::save_cast<RelearnTypes::counter_type>(i + 1);
        const auto number_vacant_excitatory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 1) / 2;
        const auto number_vacant_inhibitory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 3) / 3;

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto check = [neuron_id, position](const auto& _node_pointers) {
        using tt = Kernel<FastMultipoleMethodCell>;

        const auto kernel = GaussianDistributionKernel{};

        ASSERT_THROW_NO_PRINT(std::ignore = tt::create_probability_interval(kernel,
                                                                            { mpiPP::MPIRank::root_rank(), neuron_id }, position, _node_pointers, ElementType::Axon, SignalType::Excitatory),
                              RelearnException);

        ASSERT_THROW_NO_PRINT(std::ignore = tt::create_probability_interval(kernel,
                                                                            { mpiPP::MPIRank::root_rank(), neuron_id }, position, _node_pointers, ElementType::Axon, SignalType::Inhibitory),
                              RelearnException);

        ASSERT_THROW_NO_PRINT(std::ignore = tt::create_probability_interval(kernel,
                                                                            { mpiPP::MPIRank::root_rank(), neuron_id }, position, _node_pointers, ElementType::Dendrite, SignalType::Excitatory),
                              RelearnException);

        ASSERT_THROW_NO_PRINT(std::ignore = tt::create_probability_interval(kernel,
                                                                            { mpiPP::MPIRank::root_rank(), neuron_id }, position, _node_pointers, ElementType::Dendrite, SignalType::Inhibitory),
                              RelearnException);
    };

    node_pointers[2] = nullptr;
    check(node_pointers);
    node_pointers[2] = &nodes[2];

    node_pointers[0] = nullptr;
    check(node_pointers);
    node_pointers[0] = &nodes[0];

    node_pointers[22] = nullptr;
    check(node_pointers);
    node_pointers[22] = &nodes[22];

    node_pointers[0] = nullptr;
    node_pointers[2] = nullptr;
    node_pointers[22] = nullptr;
    check(node_pointers);
}

TEST_F(KernelTest, testCreateProbabilityIntervalVector) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id = NeuronID{ 423 };
    const auto position = RelearnTypes::position_type{ 13.2, 14.5, 0.2 };

    const auto number_nodes = NeuronID::value_type{ 32 };

    auto nodes = std::vector<OctreeNode<FastMultipoleMethodCell>>{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    auto node_pointers = std::vector<OctreeNode<FastMultipoleMethodCell>*>{ number_nodes, nullptr };

    for (const auto i : ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronID{ i });
        nodes[i].set_cell_size(SimulationFactory::get_minimum_position(), SimulationFactory::get_maximum_position());
        nodes[i].set_rank(mpiPP::MPIRank(0));

        const auto target_excitatory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_excitatory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };

        const auto number_vacant_excitatory_axons = utility::save_cast<RelearnTypes::counter_type>(i);
        const auto number_vacant_inhibitory_axons = utility::save_cast<RelearnTypes::counter_type>(i + 1);
        const auto number_vacant_excitatory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 1) / 2;
        const auto number_vacant_inhibitory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 3) / 3;

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto kernel = GaussianDistributionKernel{};

    auto total_attractiveness = 0.0;
    auto attractivenesses = std::vector<double>{};
    for (const auto i : ranges::views::indices(number_nodes)) {
        const auto attr = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &nodes[i], ElementType::Axon, SignalType::Excitatory);

        attractivenesses.emplace_back(attr);
        total_attractiveness += attr;
    }

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                            { mpiPP::MPIRank::root_rank(), neuron_id }, position, node_pointers, ElementType::Axon, SignalType::Excitatory);

    if (total_attractiveness > 0.0) {
        ASSERT_NEAR(sum, total_attractiveness, eps);
        ASSERT_EQ(attractivenesses.size(), attrs.size());

        for (const auto i : ranges::views::indices(attrs.size())) {
            ASSERT_NEAR(attrs[i], attractivenesses[i], eps);
        }
    }
}

TEST_F(KernelTest, testCreateProbabilityIntervalEdgeCase) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id = NeuronID{ 423 };
    const auto position = RelearnTypes::position_type{ 130000000.2, 140000000.5, 1000000000.2 };

    const auto number_nodes = NeuronID::value_type{ 12 };

    auto nodes = std::vector<OctreeNode<FastMultipoleMethodCell>>{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    auto node_pointers = std::vector<OctreeNode<FastMultipoleMethodCell>*>{ number_nodes, nullptr };

    for (const auto i : ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronID{ i });
        nodes[i].set_cell_size(SimulationFactory::get_minimum_position(), SimulationFactory::get_maximum_position());
        nodes[i].set_rank(mpiPP::MPIRank(0));

        const auto target_excitatory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_excitatory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };

        const auto number_vacant_excitatory_axons = utility::save_cast<RelearnTypes::counter_type>(i);
        const auto number_vacant_inhibitory_axons = utility::save_cast<RelearnTypes::counter_type>(i + 1);
        const auto number_vacant_excitatory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 1) / 2;
        const auto number_vacant_inhibitory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 3) / 3;

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto kernel = GaussianDistributionKernel{};

    for (const auto i : ranges::views::indices(number_nodes)) {
        const auto attr = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, &nodes[i], ElementType::Axon, SignalType::Excitatory);

        ASSERT_NEAR(attr, 0.0, eps) << ' ' << attr << ' ' << i;
    }

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodCell>::create_probability_interval(kernel,
                                                                                            { mpiPP::MPIRank::root_rank(), neuron_id }, position, node_pointers, ElementType::Axon, SignalType::Excitatory);

    auto total_attractiveness = 0.0;
    auto attractivenesses = std::vector<double>{};

    for (const auto i : ranges::views::indices(number_nodes)) {
        const auto number_values = nodes[i].get_cell().get_number_elements_for(ElementType::Axon, SignalType::Excitatory);
        if (number_values == 0) {
            attractivenesses.emplace_back(0);
            continue;
        }

        const auto opt_pos = nodes[i].get_cell().get_position_for(ElementType::Axon, SignalType::Excitatory);
        ASSERT_TRUE(opt_pos.has_value());
        const auto distance = (position - opt_pos.value()).calculate_2_norm();
        const auto attr = number_values / distance;

        attractivenesses.emplace_back(attr);
        total_attractiveness += attr;
    }

    ASSERT_NEAR(sum, total_attractiveness, eps);
    ASSERT_EQ(attractivenesses.size(), attrs.size());

    for (auto i = 0U; i < attrs.size(); i++) {
        ASSERT_NEAR(attrs[i], attractivenesses[i], eps);
    }
}

TEST_F(KernelTest, testPickTargetEmpty2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id = NeuronID{ 423 };
    const auto position = RelearnTypes::position_type{ 13.2, 14.5, 0.2 };

    const auto kernel = GaussianDistributionKernel{};

    auto* result_1 = Kernel<BarnesHutCell>::pick_target(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, {}, ElementType::Dendrite, SignalType::Excitatory);
    auto* result_2 = Kernel<BarnesHutCell>::pick_target(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, {}, ElementType::Dendrite, SignalType::Inhibitory);

    ASSERT_EQ(result_1, nullptr);
    ASSERT_EQ(result_2, nullptr);
}

TEST_F(KernelTest, testPickTargetException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id = NeuronID{ 423 };
    const auto position = RelearnTypes::position_type{ 13.2, 14.5, 0.2 };

    const auto number_nodes = NeuronID::value_type{ 17 };

    auto nodes = std::vector<OctreeNode<FastMultipoleMethodCell>>{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    auto node_pointers = std::vector<OctreeNode<FastMultipoleMethodCell>*>{ number_nodes, nullptr };

    for (const auto i : ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronID{ i });
        nodes[i].set_cell_size(SimulationFactory::get_minimum_position(), SimulationFactory::get_maximum_position());

        const auto target_excitatory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_excitatory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };

        const auto number_vacant_excitatory_axons = utility::save_cast<RelearnTypes::counter_type>(i);
        const auto number_vacant_inhibitory_axons = utility::save_cast<RelearnTypes::counter_type>(i + 1);
        const auto number_vacant_excitatory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 1) / 2;
        const auto number_vacant_inhibitory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 3) / 3;

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto check = [neuron_id, position](const auto& _node_pointers) {
        using TT = Kernel<FastMultipoleMethodCell>;

        const auto kernel = GaussianDistributionKernel{};

        ASSERT_THROW_NO_PRINT(std::ignore = TT::pick_target(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, _node_pointers, ElementType::Axon, SignalType::Excitatory);, RelearnException);
        ASSERT_THROW_NO_PRINT(std::ignore = TT::pick_target(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, _node_pointers, ElementType::Axon, SignalType::Inhibitory);, RelearnException);
        ASSERT_THROW_NO_PRINT(std::ignore = TT::pick_target(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, _node_pointers, ElementType::Dendrite, SignalType::Excitatory);, RelearnException);
        ASSERT_THROW_NO_PRINT(std::ignore = TT::pick_target(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, _node_pointers, ElementType::Dendrite, SignalType::Inhibitory);, RelearnException);
    };

    node_pointers[2] = nullptr;
    check(node_pointers);
    node_pointers[2] = &nodes[2];

    node_pointers[0] = nullptr;
    check(node_pointers);
    node_pointers[0] = &nodes[0];

    node_pointers[16] = nullptr;
    check(node_pointers);
    node_pointers[16] = &nodes[16];

    node_pointers[0] = nullptr;
    node_pointers[2] = nullptr;
    node_pointers[16] = nullptr;
    check(node_pointers);
}

TEST_F(KernelTest, testPickTargetRandom2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto neuron_id = NeuronID{ 423 };
    const auto position = RelearnTypes::position_type{ 13.2, 14.5, 0.2 };

    const auto number_nodes = NeuronID::value_type{ 26 };

    auto nodes = std::vector<OctreeNode<FastMultipoleMethodCell>>{ number_nodes, OctreeNode<FastMultipoleMethodCell>{} };
    auto node_pointers = std::vector<OctreeNode<FastMultipoleMethodCell>*>{ number_nodes, nullptr };

    for (const auto i : ranges::views::indices(number_nodes)) {
        nodes[i].set_cell_neuron_id(NeuronID{ i });
        nodes[i].set_cell_size(SimulationFactory::get_minimum_position(), SimulationFactory::get_maximum_position());

        const auto target_excitatory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_axon_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_excitatory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };
        const auto target_inhibitory_dendrite_position = RelearnTypes::position_type{ 13.2, 14.5, 1.2 };

        const auto number_vacant_excitatory_axons = utility::save_cast<RelearnTypes::counter_type>(i);
        const auto number_vacant_inhibitory_axons = utility::save_cast<RelearnTypes::counter_type>(i + 1);
        const auto number_vacant_excitatory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 1) / 2;
        const auto number_vacant_inhibitory_dendrites = utility::save_cast<RelearnTypes::counter_type>(i + 3) / 3;

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto check = [neuron_id, position, node_pointers](const auto element_type, const auto signal_type) {
        const auto kernel = GaussianDistributionKernel{};

        for ([[maybe_unused]] const auto i : ranges::views::indices(number_nodes)) {
            auto* result = Kernel<FastMultipoleMethodCell>::
                pick_target(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, node_pointers, element_type, signal_type);

            if (result == nullptr) {
                for (auto* ptr : node_pointers) {
                    const auto attraction_1 = Kernel<FastMultipoleMethodCell>::calculate_attractiveness_to_connect(kernel, { mpiPP::MPIRank::root_rank(), neuron_id }, position, ptr, element_type, signal_type);
                    ASSERT_EQ(attraction_1, 0.0);
                }

                continue;
            }

            ASSERT_TRUE(ranges::contains(node_pointers, result));
        }
    };

    check(ElementType::Axon, SignalType::Excitatory);
    check(ElementType::Axon, SignalType::Inhibitory);
    check(ElementType::Dendrite, SignalType::Excitatory);
    check(ElementType::Dendrite, SignalType::Inhibitory);
}
