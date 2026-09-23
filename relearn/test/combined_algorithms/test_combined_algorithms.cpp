/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_combined_algorithms.h"

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/Algorithms.h"
#include "algorithm/BarnesHutInternal/BarnesHut.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/BarnesHutInternal/BarnesHutInverted.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "algorithm/CombinedAlgorithmsInternal/AlgorithmConfig.h"
#include "algorithm/CombinedAlgorithmsInternal/CombinedAlgorithms.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "algorithm/Kernel/Gaussian.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "structure/Morton.h"
#include "types/AlgorithmTypes.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"

#include "adapter/octree/OctreeAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"

#include "factory/algorithm/algorithm_config_factory.h"
#include "factory/kernel/kernel_factory.h"
#include "factory/memory_holder/memory_holder_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"
#include "factory/synaptic_elements/synaptic_elements_factory.h"

#include <cpp-utility/Cast.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <utility>

#ifndef RELEARN_CUDA_ENABLED

TEST_F(CombinedAlgorithmsTest, testAlgorithmPointers) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_excitatory_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto number_inhibitory_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto number_neurons = number_excitatory_neurons + number_inhibitory_neurons;

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto level = RelearnTypes::level_type{ 0 };
    const auto morton = std::make_shared<Morton>(level);

    const auto signal_types = SynapticElementsFactory::get_signal_types(number_excitatory_neurons, number_inhibitory_neurons, mt);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, mt);

    auto positions = std::map<NeuronID::value_type, RelearnTypes::position_type>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    const auto num_configs = AlgorithmConfigFactory::get_random_number_algorithms(mt);
    auto configs = AlgorithmConfigFactory::create_random_algorithm_configs(num_configs, mt);

    // remember kernels because later they will be moved to the algorithms
    auto remembered_kernels = std::vector<KernelBase*>(num_configs);
    std::transform(configs.begin(), configs.end(), remembered_kernels.begin(), [](const AlgorithmConfig& config) {
        return config.get_kernel();
    });

    const auto indices_and_neurons_placeholder = RelearnTypes::AlgorithmIndexWithNeuronsType{};

    const auto theta = RandomFactory::get_random_double(std::numeric_limits<RelearnTypes::acceptance_criterion_type>::min(), utility::as<RelearnTypes::acceptance_criterion_type>(0.5), mt); // the default theta to be used if none is given in the configuration file for a config

    auto combined_algorithms = CombinedAlgorithms{ RelearnTypes::bounding_box_type{ min, max }, morton, std::move(configs), indices_and_neurons_placeholder, utility::cast<RelearnTypes::acceptance_criterion_type>(theta) };

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    const auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    combined_algorithms.set_neuron_extra_infos(extra_infos);
    combined_algorithms.set_network_graph(network_graph);
    combined_algorithms.set_synaptic_elements(synaptic_elements);
    combined_algorithms.init(number_neurons);

    // assertions
    const auto& algorithm_pointers = combined_algorithms.get_algorithm_pointers();

    ASSERT_EQ(algorithm_pointers.size(), static_cast<size_t>(num_configs));

    auto configs_of_combined_algorithms = combined_algorithms.transfer_algorithm_configs(); // move back algorithm_configs, as they are not needed in combined algorithms, but here
    for (auto i = 0UL; i < num_configs; ++i) {
        auto& config = configs_of_combined_algorithms[i];
        const auto& alg_ptr = algorithm_pointers[i];
        const auto& alg_extra_infos = alg_ptr->get_extra_infos();
        const auto& alg_network_graph = alg_ptr->get_network_graph();
        const auto& alg_synaptic_elements = alg_ptr->get_synaptic_elements();
        const auto& alg_kernel = alg_ptr->get_kernel();
        const auto& alg_type = alg_ptr->get_algorithm_type();

        ASSERT_EQ(alg_extra_infos, extra_infos);
        ASSERT_EQ(alg_network_graph, network_graph);
        ASSERT_EQ(alg_synaptic_elements, synaptic_elements);
        ASSERT_TRUE((*alg_kernel).is_approximately_equal(*remembered_kernels[i]));
        ASSERT_EQ(alg_type, config.get_algorithm_type());
        if (is_barnes_hut(alg_type)) {
            RelearnTypes::acceptance_criterion_type alg_theta{};
            switch (alg_type) {
            case AlgorithmEnum::BarnesHut:
                alg_theta = utility::cast<double>(std::static_pointer_cast<BarnesHut>(alg_ptr)->get_acceptance_criterion());
                break;
            case AlgorithmEnum::BarnesHutInverted:
                alg_theta = utility::cast<double>(std::static_pointer_cast<BarnesHutInverted>(alg_ptr)->get_acceptance_criterion());
                break;
            case AlgorithmEnum::BarnesHutLocationAware:
                alg_theta = utility::cast<double>(std::static_pointer_cast<BarnesHutLocationAware>(alg_ptr)->get_acceptance_criterion());
                break;
            default:
                RelearnException::fail("CombinedAlgorithmsTest::testAlgorithmPointers: algorithm type is said to be barnes hut, but it is not! (actual algorithm type: {})", alg_type);
            }
            const auto golden_theta = config.get_theta().value_or(theta); // the golden theta is the one in the config if one is set, otherwise it is the theta given to the CombinedAlgorithms object

            ASSERT_EQ(alg_theta, golden_theta);
        }
    }
}

TEST_F(CombinedAlgorithmsTest, testTwoNeuronsAlmostDeterministic1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }
        return;
    }

    // Test almost deterministic because of randomness in some steps. But the way the neurons connect should be determined.

    const auto number_neurons = CombinedAlgorithms::number_neurons_type{ 2 };

    std::unique_ptr<KernelBase> first_kernel = KernelFactory::get_standard_gaussian();
    auto first_algorithm_config = AlgorithmConfig(AlgorithmEnum::BarnesHut, std::move(first_kernel));
    auto first_indices_and_neurons_pair = std::make_pair<std::size_t, std::vector<NeuronID>>(0, { NeuronID(0) });

    std::unique_ptr<KernelBase> second_kernel = KernelFactory::get_standard_gaussian();
    auto second_algorithm_config = AlgorithmConfig(AlgorithmEnum::BarnesHutInverted, std::move(second_kernel));
    auto second_indices_and_neurons_pair = std::make_pair<std::size_t, std::vector<NeuronID>>(1, { NeuronID(1) });

    auto algorithm_configs = RelearnTypes::AlgorithmConfigs{};
    algorithm_configs.push_back(std::move(first_algorithm_config));
    algorithm_configs.push_back(std::move(second_algorithm_config));
    const auto indices_and_neurons = RelearnTypes::AlgorithmIndexWithNeuronsType{ first_indices_and_neurons_pair, second_indices_and_neurons_pair };

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto level = RelearnTypes::level_type{ 0 };
    const auto morton = std::make_shared<Morton>(level);

    auto combined_algorithms = std::make_shared<CombinedAlgorithms>(RelearnTypes::bounding_box_type{ minimum, maximum }, morton, std::move(algorithm_configs), indices_and_neurons);

    const auto signal_type = RandomFactory::get_random_bool(mt) ? SignalType::Excitatory : SignalType::Inhibitory;
    const auto signal_types = std::vector<SignalType>(number_neurons, signal_type);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(minimum, maximum, number_neurons, mt);

    auto positions = std::map<NeuronID::value_type, RelearnTypes::position_type>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(extra_infos, signal_types, 1.0, 1.0);

    combined_algorithms->set_synaptic_elements(synaptic_elements);
    combined_algorithms->set_network_graph(network_graph);
    combined_algorithms->set_neuron_extra_infos(extra_infos);

    combined_algorithms->init(number_neurons);

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    const auto number_connected_axons_before = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Axon, signal_type));
    const auto number_connected_dendrites_before = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Dendrite, signal_type));

    // no connections before updating connectivity
    ASSERT_EQ(number_connected_axons_before[0], 0);
    ASSERT_EQ(number_connected_dendrites_before[0], 0);

    ASSERT_EQ(number_connected_axons_before[1], 0);
    ASSERT_EQ(number_connected_dendrites_before[1], 0);

    ASSERT_NO_THROW(combined_algorithms->prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));

    ASSERT_NO_THROW(std::ignore = combined_algorithms->update_connectivity(number_neurons));

    const auto number_connected_axons = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Axon, signal_type));
    const auto number_connected_dendrites = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Dendrite, signal_type));

    // explanation of golden values: the first neuron searches from axons to dendrites. It should connect to the second neuron
    // -> axon from first neuron connected to dendrite from second neuron.
    // The second neuron then searches from dendrites to axons. Since its dendrite is already connected, nothing happens.
    const auto golden_number_connected_axons_first_neuron = 1;
    const auto golden_number_connected_dendrites_first_neuron = 0;

    const auto golden_number_connected_axons_second_neuron = 0;
    const auto golden_number_connected_dendrites_second_neuron = 1;

    ASSERT_EQ(number_connected_axons[0], golden_number_connected_axons_first_neuron);
    ASSERT_EQ(number_connected_dendrites[0], golden_number_connected_dendrites_first_neuron);

    ASSERT_EQ(number_connected_axons[1], golden_number_connected_axons_second_neuron);
    ASSERT_EQ(number_connected_dendrites[1], golden_number_connected_dendrites_second_neuron);
}

TEST_F(CombinedAlgorithmsTest, testTwoNeuronsAlmostDeterministic2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }
        return;
    }

    // Test almost deterministic because of randomness in some steps. But the way the neurons connect should be determined.

    const auto number_neurons = CombinedAlgorithms::number_neurons_type{ 2 };

    std::unique_ptr<KernelBase> first_kernel = KernelFactory::get_standard_linear();
    auto first_algorithm_config = AlgorithmConfig(AlgorithmEnum::BarnesHutInverted, std::move(first_kernel));
    auto first_indices_and_neurons_pair = std::make_pair<std::size_t, std::vector<NeuronID>>(0, { NeuronID(0) });

    std::unique_ptr<KernelBase> second_kernel = KernelFactory::get_standard_weibull();
    auto second_algorithm_config = AlgorithmConfig(AlgorithmEnum::BarnesHut, std::move(second_kernel));
    auto second_indices_and_neurons_pair = std::make_pair<std::size_t, std::vector<NeuronID>>(1, { NeuronID(1) });

    auto algorithm_configs = RelearnTypes::AlgorithmConfigs{};
    algorithm_configs.push_back(std::move(first_algorithm_config));
    algorithm_configs.push_back(std::move(second_algorithm_config));
    const auto indices_and_neurons = RelearnTypes::AlgorithmIndexWithNeuronsType{ first_indices_and_neurons_pair, second_indices_and_neurons_pair };

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto level = RelearnTypes::level_type{ 0 };
    const auto morton = std::make_shared<Morton>(level);

    auto combined_algorithms = std::make_shared<CombinedAlgorithms>(RelearnTypes::bounding_box_type{ minimum, maximum }, morton, std::move(algorithm_configs), indices_and_neurons);

    const auto signal_type = RandomFactory::get_random_bool(mt) ? SignalType::Excitatory : SignalType::Inhibitory;
    const auto signal_types = std::vector<SignalType>(number_neurons, signal_type);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(minimum, maximum, number_neurons, mt);

    auto positions = std::map<NeuronID::value_type, RelearnTypes::position_type>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(extra_infos, signal_types, 1.0, 1.0);

    combined_algorithms->set_synaptic_elements(synaptic_elements);
    combined_algorithms->set_network_graph(network_graph);
    combined_algorithms->set_neuron_extra_infos(extra_infos);

    combined_algorithms->init(number_neurons);

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    const auto number_connected_axons_before = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Axon, signal_type));
    const auto number_connected_dendrites_before = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Dendrite, signal_type));

    // no connections before updating connectivity
    ASSERT_EQ(number_connected_axons_before[0], 0);
    ASSERT_EQ(number_connected_dendrites_before[0], 0);

    ASSERT_EQ(number_connected_axons_before[1], 0);
    ASSERT_EQ(number_connected_dendrites_before[1], 0);

    ASSERT_NO_THROW(combined_algorithms->prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));

    ASSERT_NO_THROW(std::ignore = combined_algorithms->update_connectivity(number_neurons));

    const auto number_connected_axons = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Axon, signal_type));
    const auto number_connected_dendrites = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Dendrite, signal_type));

    // explanation of golden values: the first neuron searches from dendrites to axons. It should connect to the second neuron
    // -> dendrite from first neuron connected to axon from second neuron.
    // The second neuron then searches from axons to dendrites. Since its axon is already connected, nothing happens.
    const auto golden_number_connected_axons_first_neuron = 0;
    const auto golden_number_connected_dendrites_first_neuron = 1;

    const auto golden_number_connected_axons_second_neuron = 1;
    const auto golden_number_connected_dendrites_second_neuron = 0;

    ASSERT_EQ(number_connected_axons[0], golden_number_connected_axons_first_neuron);
    ASSERT_EQ(number_connected_dendrites[0], golden_number_connected_dendrites_first_neuron);

    ASSERT_EQ(number_connected_axons[1], golden_number_connected_axons_second_neuron);
    ASSERT_EQ(number_connected_dendrites[1], golden_number_connected_dendrites_second_neuron);
}

TEST_F(CombinedAlgorithmsTest, testTwoNeuronsAlmostDeterministic3) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }
        return;
    }

    // Test almost deterministic because of randomness in some steps. But the way the neurons connect should be determined.

    const auto number_neurons = CombinedAlgorithms::number_neurons_type{ 2 };

    std::unique_ptr<KernelBase> first_kernel = KernelFactory::get_standard_gamma();
    auto first_algorithm_config = AlgorithmConfig(AlgorithmEnum::BarnesHutInverted, std::move(first_kernel));
    auto first_indices_and_neurons_pair = std::make_pair<std::size_t, std::vector<NeuronID>>(0, { NeuronID(0) });

    std::unique_ptr<KernelBase> second_kernel = KernelFactory::get_standard_gaussian();
    auto second_algorithm_config = AlgorithmConfig(AlgorithmEnum::BarnesHutLocationAware, std::move(second_kernel));
    auto second_indices_and_neurons_pair = std::make_pair<std::size_t, std::vector<NeuronID>>(1, { NeuronID(1) });

    auto algorithm_configs = RelearnTypes::AlgorithmConfigs{};
    algorithm_configs.push_back(std::move(first_algorithm_config));
    algorithm_configs.push_back(std::move(second_algorithm_config));
    const auto indices_and_neurons = RelearnTypes::AlgorithmIndexWithNeuronsType{ first_indices_and_neurons_pair, second_indices_and_neurons_pair };

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto level = RelearnTypes::level_type{ 0 };
    const auto morton = std::make_shared<Morton>(level);

    auto combined_algorithms = std::make_shared<CombinedAlgorithms>(RelearnTypes::bounding_box_type{ minimum, maximum }, morton, std::move(algorithm_configs), indices_and_neurons);

    const auto signal_type = RandomFactory::get_random_bool(mt) ? SignalType::Excitatory : SignalType::Inhibitory;
    const auto signal_types = std::vector<SignalType>(number_neurons, signal_type);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(minimum, maximum, number_neurons, mt);

    auto positions = std::map<NeuronID::value_type, RelearnTypes::position_type>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(extra_infos, signal_types, 2.0, 2.0);

    combined_algorithms->set_synaptic_elements(synaptic_elements);
    combined_algorithms->set_network_graph(network_graph);
    combined_algorithms->set_neuron_extra_infos(extra_infos);

    combined_algorithms->init(number_neurons);

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    const auto number_connected_axons_before = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Axon, signal_type));
    const auto number_connected_dendrites_before = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Dendrite, signal_type));

    // no connections before updating connectivity
    ASSERT_EQ(number_connected_axons_before[0], 0);
    ASSERT_EQ(number_connected_dendrites_before[0], 0);

    ASSERT_EQ(number_connected_axons_before[1], 0);
    ASSERT_EQ(number_connected_dendrites_before[1], 0);

    ASSERT_NO_THROW(combined_algorithms->prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));

    ASSERT_NO_THROW(std::ignore = combined_algorithms->update_connectivity(number_neurons));

    const auto number_connected_axons = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Axon, signal_type));
    const auto number_connected_dendrites = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Dendrite, signal_type));

    // explanation of golden values: the first neuron searches from dendrites to axons. It should connect to the second neuron
    // -> dendrite from first neuron connected to axon from second neuron.
    // The second neuron then searches from axons to dendrites. It has a free axon so it should connect to the free dendrite
    // of the first neuron.
    const auto golden_number_connected_axons_first_neuron = 0;
    const auto golden_number_connected_dendrites_first_neuron = 2;

    const auto golden_number_connected_axons_second_neuron = 2;
    const auto golden_number_connected_dendrites_second_neuron = 0;

    ASSERT_EQ(number_connected_axons[0], golden_number_connected_axons_first_neuron);
    ASSERT_EQ(number_connected_dendrites[0], golden_number_connected_dendrites_first_neuron);

    ASSERT_EQ(number_connected_axons[1], golden_number_connected_axons_second_neuron);
    ASSERT_EQ(number_connected_dendrites[1], golden_number_connected_dendrites_second_neuron);
}

TEST_F(CombinedAlgorithmsTest, testTwoNeuronsAlmostDeterministic4) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }
        return;
    }

    // Test almost deterministic because of randomness in some steps. But the way the neurons connect should be determined.

    const auto number_neurons = CombinedAlgorithms::number_neurons_type{ 2 };

    std::unique_ptr<KernelBase> first_kernel = KernelFactory::get_standard_linear();
    auto first_algorithm_config = AlgorithmConfig(AlgorithmEnum::BarnesHut, std::move(first_kernel));
    auto first_indices_and_neurons_pair = std::make_pair<std::size_t, std::vector<NeuronID>>(0, { NeuronID(0) });

    std::unique_ptr<KernelBase> second_kernel = KernelFactory::get_standard_gaussian();
    auto second_algorithm_config = AlgorithmConfig(AlgorithmEnum::Naive, std::move(second_kernel));
    auto second_indices_and_neurons_pair = std::make_pair<std::size_t, std::vector<NeuronID>>(1, { NeuronID(1) });

    auto algorithm_configs = RelearnTypes::AlgorithmConfigs{};
    algorithm_configs.push_back(std::move(first_algorithm_config));
    algorithm_configs.push_back(std::move(second_algorithm_config));
    const auto indices_and_neurons = RelearnTypes::AlgorithmIndexWithNeuronsType{ first_indices_and_neurons_pair, second_indices_and_neurons_pair };

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto level = RelearnTypes::level_type{ 0 };
    const auto morton = std::make_shared<Morton>(level);

    auto combined_algorithms = std::make_shared<CombinedAlgorithms>(RelearnTypes::bounding_box_type{ minimum, maximum }, morton, std::move(algorithm_configs), indices_and_neurons);

    const auto signal_type = RandomFactory::get_random_bool(mt) ? SignalType::Excitatory : SignalType::Inhibitory;
    const auto signal_types = std::vector<SignalType>(number_neurons, signal_type);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(minimum, maximum, number_neurons, mt);

    auto positions = std::map<NeuronID::value_type, RelearnTypes::position_type>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(extra_infos, signal_types, 2.0, 2.0);

    combined_algorithms->set_synaptic_elements(synaptic_elements);
    combined_algorithms->set_network_graph(network_graph);
    combined_algorithms->set_neuron_extra_infos(extra_infos);

    combined_algorithms->init(number_neurons);

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    const auto number_connected_axons_before = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Axon, signal_type));
    const auto number_connected_dendrites_before = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Dendrite, signal_type));

    // no connections before updating connectivity
    ASSERT_EQ(number_connected_axons_before[0], 0);
    ASSERT_EQ(number_connected_dendrites_before[0], 0);

    ASSERT_EQ(number_connected_axons_before[1], 0);
    ASSERT_EQ(number_connected_dendrites_before[1], 0);

    ASSERT_NO_THROW(combined_algorithms->prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));

    ASSERT_NO_THROW(std::ignore = combined_algorithms->update_connectivity(number_neurons));

    const auto number_connected_axons = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Axon, signal_type));
    const auto number_connected_dendrites = synaptic_elements->get_connected_elements(get_synaptic_element_type(ElementType::Dendrite, signal_type));

    // explanation of golden values: both neurons search from axons to dendrites. They should both connect their two vacant
    // axons to the two vacant dendrites of the other neuron.
    const auto golden_number_connected_axons_first_neuron = 2;
    const auto golden_number_connected_dendrites_first_neuron = 2;

    const auto golden_number_connected_axons_second_neuron = 2;
    const auto golden_number_connected_dendrites_second_neuron = 2;

    ASSERT_EQ(number_connected_axons[0], golden_number_connected_axons_first_neuron);
    ASSERT_EQ(number_connected_dendrites[0], golden_number_connected_dendrites_first_neuron);

    ASSERT_EQ(number_connected_axons[1], golden_number_connected_axons_second_neuron);
    ASSERT_EQ(number_connected_dendrites[1], golden_number_connected_dendrites_second_neuron);
}

TEST_F(CombinedAlgorithmsTest, testAllAlgorithmsRunNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }
        return;
    }

    const auto number_configs = std::uint64_t{ 4 };                                              // because 4 different types of algorithms are supported
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + number_configs; // we want at least number_configs many neurons

    auto algorithm_configs = RelearnTypes::AlgorithmConfigs{};
    for (std::uint64_t i = 0; i < number_configs; ++i) { // when AlgorithmEnum changes, this might need update
        std::unique_ptr<KernelBase> kernel = KernelFactory::get_random_standard_kernel(mt);
        const auto algorithm_type = AlgorithmEnum{ static_cast<std::uint8_t>(i) };
        algorithm_configs.push_back(AlgorithmConfig(algorithm_type, std::move(kernel)));
    }

    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto level = RelearnTypes::level_type{ 0 };
    const auto morton = std::make_shared<Morton>(level);

    auto indices_and_neurons = RelearnTypes::AlgorithmIndexWithNeuronsType{};

    for (auto i = 0ULL; i < number_configs; ++i) {
        indices_and_neurons.emplace_back(static_cast<std::size_t>(i), std::vector<NeuronID>{});
    }

    for (auto i = 0ULL; i < number_neurons; ++i) { // assigns neurons to algorithms. 0 neurons for an algorithm are unlikely but not a problem
        const auto index = RandomFactory::get_random_integer<NeuronID::value_type>(NeuronID::value_type{ 0 }, number_configs - 1, mt);
        indices_and_neurons[index].second.push_back(NeuronID(i));
    }

    auto combined_algorithms = std::make_shared<CombinedAlgorithms>(RelearnTypes::bounding_box_type{ minimum, maximum }, morton, std::move(algorithm_configs), indices_and_neurons);

    const auto number_excitatory_neurons = RandomFactory::get_random_integer<RelearnTypes::number_neurons_type>(RelearnTypes::number_neurons_type{ 0 }, number_neurons, mt);
    const auto number_inhibitory_neurons = number_neurons - number_excitatory_neurons;

    const auto signal_types = SynapticElementsFactory::get_signal_types(number_excitatory_neurons, number_inhibitory_neurons, mt);

    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);

    auto network_graph = NetworkGraphFactory::construct_network_graph(number_neurons, mpiPP::MPIRank::root_rank(), 8, mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(minimum, maximum, number_neurons, mt);

    auto positions = std::map<NeuronID::value_type, RelearnTypes::position_type>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    synaptic_elements->set_extra_infos(extra_infos);

    combined_algorithms->set_synaptic_elements(synaptic_elements);
    combined_algorithms->set_network_graph(network_graph);
    combined_algorithms->set_neuron_extra_infos(extra_infos);

    combined_algorithms->init(number_neurons);

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    ASSERT_NO_THROW(combined_algorithms->prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));

    ASSERT_NO_THROW(std::ignore = combined_algorithms->update_connectivity(number_neurons));
}
#endif