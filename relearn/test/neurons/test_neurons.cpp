/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neurons.h"

#include "Types.h"
#include "Types3.h"

#include "neurons/Neurons.h"
#include "neurons/calcium/CalciumCalculator.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/growthrate/ConstantGrowthrateCalculator.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "neurons/synaptic_elements/Axons.h"
#include "neurons/synaptic_elements/Dendrites.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include "cpp-utility/data/vectorify.hpp"
#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/CommunicationMap.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "adapter/mpi/MpiAdapter.h"
#include "adapter/network_graph/NetworkGraphAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"

#include "factory/algorithm/algorithm_factory.h"
#include "factory/kernel/kernel_factory.h"
#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_model/neuron_model_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"
#include "factory/synapses/synapses_factory.h"
#include "factory/synaptic_elements/synaptic_elements_factory.h"

#include <gtest/gtest.h>

#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/algorithm/contains.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/map.hpp>
#include <range/v3/view/transform.hpp>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <span>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

TEST_F(NeuronsTest, testNeuronsConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto partition = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());

    auto model = NeuronModelFactory::construct_poisson_model();

    auto calcium = std::make_unique<CalciumCalculator>();
    auto network_graph = std::make_shared<NetworkGraph>(mpiPP::MPIRank::root_rank());

    auto signal_types = SynapticElementsFactory::get_signal_types(100, 50, mt);
    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);
    SynapticElementsAdapter::set_growthrate_calculators(synaptic_elements);

    auto sdf = std::make_unique<RandomSynapseDeletionFinder>();
    sdf->set_synaptic_elements(synaptic_elements);

    const auto neurons = Neurons{ partition, std::move(model), std::move(calcium), std::move(network_graph), std::move(synaptic_elements), std::move(sdf) };
}

TEST_F(NeuronsTest, testSignalTypeCheck) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_test_synapses = RandomFactory::get_random_integer(50, 500, mt);
    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;
    const auto num_synapses = RandomFactory::get_random_integer(10, 100, mt);
    const auto num_ranks = MPIRankFactory::get_random_number_ranks(mt);
    const auto network_graph = std::make_shared<NetworkGraph>(mpiPP::MPIRank::root_rank(), num_ranks);
    network_graph->init(num_neurons);

    const auto signal_types = ranges::views::generate_n([this]() { return NeuronTypesFactory::get_random_signal_type(mt); }, num_neurons)
                              | ranges::to_vector;

    for (auto synapse_nr = 0; synapse_nr < num_synapses; synapse_nr++) {
        const auto source_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, mt);
        const auto target_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, source_neuron, mt);
        const auto target_rank = MPIRankFactory::get_random_mpi_rank(num_ranks, mt);

        auto weight = RandomFactory::get_random_double(0.1, 20.0, mt);
        if (signal_types[source_neuron.get_neuron_id()] == SignalType::Inhibitory) {
            weight = -weight;
        }

        if (target_rank == mpiPP::MPIRank::root_rank()) {
            network_graph->add_synapse(StaticLocalSynapse{ target_neuron, source_neuron, weight });
        } else {
            network_graph->add_synapse(StaticDistantOutSynapse{ RankNeuronId(target_rank, target_neuron), source_neuron, weight });
        }
    }

    ASSERT_NO_THROW(Neurons::check_signal_types(network_graph, signal_types, mpiPP::MPIRank::root_rank()));

    for (auto test_synapse = 0; test_synapse < num_test_synapses; test_synapse++) {
        const auto source_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, mt);
        const auto target_rank = MPIRankFactory::get_random_mpi_rank(num_ranks, mt);
        const auto target_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, source_neuron, mt);
        auto weight = std::abs(SynapsesFactory::get_random_plastic_synapse_weight(mt));

        if (signal_types[source_neuron.get_neuron_id()] == SignalType::Excitatory) {
            weight = -weight;
        }

        if (target_rank == mpiPP::MPIRank::root_rank()) {
            network_graph->add_synapse(PlasticLocalSynapse{ target_neuron, source_neuron, weight });
        } else {
            network_graph->add_synapse(PlasticDistantOutSynapse{ RankNeuronId(target_rank, target_neuron), source_neuron, weight });
        }

        ASSERT_THROW_NO_PRINT(Neurons::check_signal_types(network_graph, signal_types, mpiPP::MPIRank::root_rank()), RelearnException);

        if (target_rank == mpiPP::MPIRank::root_rank()) {
            network_graph->add_synapse(PlasticLocalSynapse{ target_neuron, source_neuron, -weight });
        } else {
            network_graph->add_synapse(PlasticDistantOutSynapse{ RankNeuronId(target_rank, target_neuron), source_neuron, -weight });
        }

        ASSERT_NO_THROW(Neurons::check_signal_types(network_graph, signal_types, mpiPP::MPIRank::root_rank()));
    }
}

TEST_F(NeuronsTest, testStaticConnectionsChecker) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 30;
    const auto num_static_neurons = RandomFactory::get_random_integer(static_cast<NeuronID::value_type>(15), num_neurons - 10, mt);

    const auto static_neurons = NeuronIdFactory::get_random_neuron_ids(num_neurons, num_static_neurons, mt) | ranges::to_vector;

    auto calcium = std::make_unique<CalciumCalculator>();
    calcium->set_initial_calcium_calculator(
        [](mpiPP::MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
    calcium->set_target_calcium_calculator(
        [](mpiPP::MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });

    auto axons = std::make_shared<Axons>();
    auto dendrites = std::make_shared<Dendrites>();

    auto synaptic_elements = std::make_shared<SynapticElements>(std::move(axons), std::move(dendrites));
    SynapticElementsAdapter::set_growthrate_calculators(synaptic_elements);

    auto sdf = std::make_unique<RandomSynapseDeletionFinder>();
    sdf->set_synaptic_elements(synaptic_elements);

    auto network_graph = std::make_shared<NetworkGraph>(mpiPP::MPIRank::root_rank());

    auto partition = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto model = NeuronModelFactory::construct_poisson_model();
    auto neurons = Neurons{ std::move(partition), std::move(model), std::move(calcium), network_graph, std::move(synaptic_elements), std::move(sdf) };

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, num_neurons, mt);

    auto positions = std::map<NeuronID::value_type, Vec3d>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    auto algorithm = AlgorithmFactory::construct_naive_algorithm({ min, max }, 0);
    neurons.set_algorithm(algorithm);
    neurons.set_probability_kernel(KernelFactory::get_standard_gaussian());

    neurons.init(num_neurons, neuron_positions);

    const auto num_synapses_static = RandomFactory::get_random_integer(30, 100, mt);
    const auto num_synapses_plastic = RandomFactory::get_random_integer(30, 100, mt);

    for ([[maybe_unused]] const auto _ : ranges::views::indices(num_synapses_static)) {
        const auto source_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, mt);
        const auto target_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, source_neuron, mt);
        const auto weight = std::abs(SynapsesFactory::get_random_static_synapse_weight(mt));
        network_graph->add_synapse(StaticLocalSynapse{ target_neuron, source_neuron, weight });
    }

    for ([[maybe_unused]] const auto _ : ranges::views::indices(num_synapses_plastic)) {
        auto source_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, mt);
        auto target_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, mt);

        while (ranges::contains(static_neurons, source_neuron) || ranges::contains(static_neurons, target_neuron) || source_neuron == target_neuron) {
            source_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, mt);
            target_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, mt);
        }

        const auto weight = std::abs(SynapsesFactory::get_random_plastic_synapse_weight(mt));
        network_graph->add_synapse(PlasticLocalSynapse{ NeuronID{ target_neuron }, NeuronID{ source_neuron }, weight });
    }

    neurons.set_static_neurons(static_neurons);

    const auto num_tries = RandomFactory::get_random_integer(10, 100, mt);
    for ([[maybe_unused]] const auto _ : ranges::views::indices(num_tries)) {
        const bool source_is_static = RandomFactory::get_random_bool(mt);
        const bool target_is_static = !source_is_static || RandomFactory::get_random_bool(mt);

        auto source_neuron = NeuronID::uninitialized_id();
        auto target_neuron = NeuronID::uninitialized_id();

        if (source_is_static) {
            source_neuron = RandomFactory::get_random_element(static_neurons, mt);
        } else {
            source_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, static_neurons, mt);
        }
        if (target_is_static) {
            target_neuron = RandomFactory::get_random_element(static_neurons, mt);
        } else {
            target_neuron = NeuronIdFactory::get_random_neuron_id(num_neurons, static_neurons, mt);
        }

        if (target_neuron == source_neuron) {
            continue;
        }

        const auto weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);
        network_graph->add_synapse(PlasticLocalSynapse{ target_neuron, source_neuron, weight });

        ASSERT_THROW_NO_PRINT(neurons.set_static_neurons(static_neurons), RelearnException);

        network_graph->add_synapse(PlasticLocalSynapse{ target_neuron, source_neuron, -weight });
        neurons.set_static_neurons(static_neurons);
    }
}

TEST_F(NeuronsTest, testDisableNeuronsWithoutMPI) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto num_neurons = RelearnTypes::number_neurons_type{ 30 };

    auto partition = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto neurons = create_neurons_object(partition, mpiPP::MPIRank::root_rank(), 1);
    auto network_graph = neurons->get_network_graph();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, num_neurons, mt);

    auto positions = std::map<NeuronID::value_type, Vec3d>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    auto algorithm = AlgorithmFactory::construct_naive_algorithm({ min, max }, 0);
    neurons->set_algorithm(algorithm);
    neurons->set_probability_kernel(KernelFactory::get_standard_gaussian());

    neurons->init(num_neurons, neuron_positions);

    const auto signal_types = neurons->get_synaptic_elements()->get_signal_types();

    NetworkGraphFactory::construct_dense_plastic_network(network_graph, signal_types,
                                                         num_neurons, 8, 1, mpiPP::MPIRank(0), mt);
    neurons->init_synaptic_elements({}, {}, {});

    const auto disable_id = NeuronID{ 12 };
    const auto disabled_neurons = std::vector<NeuronID>{ disable_id };
    const auto enabled_neurons = std::vector<NeuronID>{ disable_id };

    const auto [out_edges_ref, _1] = network_graph->get_local_out_edges(disable_id.get_neuron_id());
    const auto out_edges = out_edges_ref;
    ASSERT_GT(out_edges.size(), 0);

    const auto [in_edges_ref, _2] = network_graph->get_local_in_edges(disable_id.get_neuron_id());
    const auto in_edges = in_edges_ref;
    ASSERT_GT(in_edges.size(), 0);

    const auto out_ids = out_edges | ranges::views::keys | ranges::to<std::set>;
    const auto in_ids = in_edges | ranges::views::keys | ranges::to<std::set>;

    ASSERT_THROW_NO_PRINT(neurons->enable_neurons(enabled_neurons), RelearnException);

    const auto& [num_deletions, synapse_deletion_Requests] = neurons->disable_neurons(1, disabled_neurons, 1);
    ASSERT_EQ(synapse_deletion_Requests.get_total_number_requests(), 0);
    ASSERT_EQ(num_deletions, out_edges.size() + in_edges.size());

    const auto& num_distant_deletions = neurons->delete_disabled_distant_synapses(synapse_deletion_Requests, mpiPP::MPIRank(0));
    ASSERT_EQ(num_distant_deletions, 0);

    ASSERT_THROW_NO_PRINT(neurons->disable_neurons(1, disabled_neurons, 1), RelearnException);

    ASSERT_EQ(out_edges_ref.size(), 0);
    ASSERT_EQ(in_edges_ref.size(), 0);

    for (const auto& id : out_ids) {
        auto [edges, _11] = network_graph->get_local_in_edges(id.get_neuron_id());

        const bool contains = ranges::all_of(edges, utility::not_equal_to(disable_id), utility::element<0>);
        ASSERT_TRUE(contains);

        ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[id.get_neuron_id()], UpdateStatus::Enabled);
    }

    const auto connected_axons = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::Axon);
    const auto connected_excitatory_dendrites = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteExcitatory);
    const auto connected_inhibitory_dendrites = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteInhibitory);

    for (const auto& id : in_ids) {
        auto [edges, _11] = network_graph->get_local_out_edges(id.get_neuron_id());

        const bool contains = ranges::all_of(edges, utility::not_equal_to(disable_id), utility::element<0>);
        ASSERT_TRUE(contains);

        ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[id.get_neuron_id()], UpdateStatus::Enabled);
        ASSERT_EQ(connected_axons[id.get_neuron_id()], 7);
    }

    ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[disable_id.get_neuron_id()], UpdateStatus::Disabled);
    ASSERT_EQ(connected_axons[disable_id.get_neuron_id()], 0);
    ASSERT_EQ(connected_excitatory_dendrites[disable_id.get_neuron_id()], 0);
    ASSERT_EQ(connected_inhibitory_dendrites[disable_id.get_neuron_id()], 0);

    const auto signal_types_vv = std::vector<std::vector<SignalType>>{ signal_types | ranges::to_vector };
    NetworkGraphAdapter::check_validity_of_network_graphs({ network_graph }, signal_types_vv, num_neurons);
}

TEST_F(NeuronsTest, testDisableMultipleNeuronsWithoutMPI) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_excitatory_neurons = 20;
    const auto number_inhibitory_neurons = 10;
    const auto number_neurons = number_excitatory_neurons + number_inhibitory_neurons;

    auto partition = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto neurons = create_neurons_object(partition, mpiPP::MPIRank::root_rank(), 1);
    auto network_graph_plastic = neurons->get_network_graph();

    const auto simulation_box = BoundingBox{ Vec3d{ 0.0, 0.0, 0.0 }, Vec3d{ 1.0, 1.0, 1.0 } };
    const auto& [min, max] = simulation_box;
    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, mt);

    auto positions = std::map<NeuronID::value_type, Vec3d>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    auto algorithm = AlgorithmFactory::construct_naive_algorithm(simulation_box, 0);
    neurons->set_algorithm(algorithm);
    neurons->set_probability_kernel(KernelFactory::get_standard_gaussian());

    auto signal_types = SynapticElementsFactory::get_signal_types(number_excitatory_neurons, number_inhibitory_neurons, mt);
    neurons->init(number_neurons, neuron_positions);
    neurons->set_signal_types(signal_types);

    NetworkGraphAdapter::connect_to_n_th_other(network_graph_plastic, signal_types, 2);
    NetworkGraphAdapter::connect_to_n_th_other(network_graph_plastic, signal_types, 3);
    NetworkGraphAdapter::connect_to_n_th_other(network_graph_plastic, signal_types, 7);
    NetworkGraphAdapter::connect_to_n_th_other(network_graph_plastic, signal_types, 10);

    const auto disabled_neurons = std::unordered_set{
        NeuronID{ 2 }, NeuronID{ 5 }, NeuronID{ 10 }
    };

    const auto disabled_neurons_vector = disabled_neurons | ranges::to_vector;

    auto to_delete_axons = std::vector<double>{};
    to_delete_axons.resize(number_neurons, 0.0);

    auto to_delete_den_ex = std::vector<double>{};
    to_delete_den_ex.resize(number_neurons, 0.0);

    auto to_delete_den_inh = std::vector<double>{};
    to_delete_den_inh.resize(number_neurons, 0.0);

    auto expected_num_deletions = 0;
    for (const auto& disabled_id : disabled_neurons) {
        const auto [out_edges, _1] = network_graph_plastic->get_local_out_edges(disabled_id.get_neuron_id());
        for (const auto& [target, weight] : out_edges) {
            if (weight > 0) {
                to_delete_den_ex[target.get_neuron_id()]++;
            } else {
                to_delete_den_inh[target.get_neuron_id()]++;
            }

            expected_num_deletions++;
        }

        const auto [in_edges, _4] = network_graph_plastic->get_local_in_edges(disabled_id.get_neuron_id());
        for (const auto& [source, weight] : in_edges) {
            to_delete_axons[source.get_neuron_id()] += std::abs(weight);
            if (!disabled_neurons.contains(source)) {
                expected_num_deletions++;
            }
        }
    }

    neurons->init_synaptic_elements({}, {}, {});

    const auto axons_old = utility::vectorify_span(neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::Axon));
    const auto den_ex_old = utility::vectorify_span(neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteExcitatory));
    const auto den_inh_old = utility::vectorify_span(neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteInhibitory));

    const auto& [num_deletions, synapse_deletion_Requests] = neurons->disable_neurons(1, disabled_neurons_vector, 1);
    ASSERT_EQ(synapse_deletion_Requests.get_total_number_requests(), 0);
    ASSERT_EQ(num_deletions, expected_num_deletions);

    const auto& num_distant_deletions = neurons->delete_disabled_distant_synapses(synapse_deletion_Requests, mpiPP::MPIRank(0));
    ASSERT_EQ(num_distant_deletions, 0);

    const auto axons_new = utility::vectorify_span(neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::Axon));
    const auto den_ex_new = utility::vectorify_span(neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteExcitatory));
    const auto den_inh_new = utility::vectorify_span(neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteInhibitory));

    for (const auto& disable_id : disabled_neurons) {
        ASSERT_EQ(std::get<0>(network_graph_plastic->get_local_in_edges(disable_id.get_neuron_id())).size(), 0);
        ASSERT_EQ(std::get<0>(network_graph_plastic->get_local_out_edges(disable_id.get_neuron_id())).size(), 0);
        ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[disable_id.get_neuron_id()], UpdateStatus::Disabled);
        ASSERT_EQ(axons_new[disable_id.get_neuron_id()], 0);
        ASSERT_EQ(den_ex_new[disable_id.get_neuron_id()], 0);
        ASSERT_EQ(den_inh_new[disable_id.get_neuron_id()], 0);
    }

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        if (disabled_neurons.contains(neuron_id)) {
            continue;
        }

        ASSERT_EQ(axons_new[neuron_id.get_neuron_id()],
                  axons_old[neuron_id.get_neuron_id()] - to_delete_axons[neuron_id.get_neuron_id()]);

        ASSERT_EQ(den_ex_new[neuron_id.get_neuron_id()],
                  den_ex_old[neuron_id.get_neuron_id()] - to_delete_den_ex[neuron_id.get_neuron_id()]);

        ASSERT_EQ(den_inh_new[neuron_id.get_neuron_id()],
                  den_inh_old[neuron_id.get_neuron_id()] - to_delete_den_inh[neuron_id.get_neuron_id()]);
    }

    const auto signal_types_vv = std::vector<std::vector<SignalType>>{ signal_types | ranges::to_vector };
    NetworkGraphAdapter::check_validity_of_network_graphs({ network_graph_plastic }, signal_types_vv, number_neurons);
}

TEST_F(NeuronsTest, testDisableNeuronsWithRanks) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = 72;
    const auto num_ranks = 4;
    const auto num_ranks_cast = static_cast<std::size_t>(num_ranks);

    auto rank_to_neurons = std::vector<std::shared_ptr<Neurons>>{};
    rank_to_neurons.reserve(num_ranks_cast);

    auto network_graphs = std::vector<std::shared_ptr<NetworkGraph>>{};
    network_graphs.reserve(num_ranks_cast);

    auto rank_to_disabled_neurons = std::vector<std::unordered_set<NeuronID>>{};
    rank_to_disabled_neurons.reserve(num_ranks_cast);

    auto expected_distant_out_deletions_received = std::vector<std::size_t>{};
    expected_distant_out_deletions_received.resize(num_ranks_cast, 0);
    auto expected_distant_in_deletions_received = std::vector<std::size_t>{};
    expected_distant_in_deletions_received.resize(num_ranks_cast, 0);

    auto expected_distant_out_deletions_initiated = std::vector<std::size_t>{};
    expected_distant_out_deletions_initiated.resize(num_ranks_cast, 0);
    auto expected_distant_in_deletions_initiated = std::vector<std::size_t>{};
    expected_distant_in_deletions_initiated.resize(num_ranks_cast, 0);

    for (const auto mpi_rank : mpiPP::MPIRank::range(num_ranks)) {
        auto partition = std::make_shared<Partition>(1, mpiPP::MPIRank(0));
        auto neurons = create_neurons_object(partition, mpi_rank, num_ranks);
        auto network_graph_plastic = neurons->get_network_graph();

        const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(mt);
        const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, num_neurons, mt);

        auto positions = std::map<NeuronID::value_type, Vec3d>{};
        for (const auto& [position, id] : neurons_to_place) {
            positions[id.get_neuron_id()] = position;
        }
        const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

        auto algorithm = AlgorithmFactory::construct_naive_algorithm({ min, max }, 0);
        neurons->set_algorithm(algorithm);
        neurons->set_probability_kernel(KernelFactory::get_standard_gaussian());

        neurons->init(num_neurons, neuron_positions);

        const auto signal_types = neurons->get_synaptic_elements()->get_signal_types();

        const auto disabled_neurons = NeuronIdFactory::get_random_neuron_ids(num_neurons, 15, mt);
        for (const auto& neuron1 : disabled_neurons) {
            for (const auto& neuron2 : disabled_neurons) {
                if (neuron1 == neuron2) {
                    continue;
                }

                const auto target_rank = MPIRankFactory::get_random_mpi_rank(num_ranks, mt);
                const auto weight = signal_types[neuron1.get_neuron_id()] == SignalType::Excitatory ? 1 : -1;

                if (target_rank == mpi_rank) {
                    network_graph_plastic->add_synapse(PlasticLocalSynapse(neuron2, neuron1, weight));
                } else {
                    network_graph_plastic->add_synapse(
                        PlasticDistantOutSynapse(RankNeuronId(target_rank, neuron2), neuron1, weight));
                }
            }
        }
        NetworkGraphFactory::construct_dense_plastic_network(network_graph_plastic, signal_types,
                                                             num_neurons, 8, num_ranks, mpi_rank, mt);

        Neurons::check_signal_types(network_graph_plastic, signal_types, mpi_rank);

        rank_to_neurons.emplace_back(std::move(neurons));
        rank_to_disabled_neurons.emplace_back(disabled_neurons);
        network_graphs.emplace_back(network_graph_plastic);
    }

    NetworkGraphAdapter::harmonize_network_graphs_from_different_ranks(network_graphs, num_neurons);

    const auto signal_types = rank_to_neurons
                              | ranges::views::transform([](const auto& neurons_of_rank) { return neurons_of_rank->get_synaptic_elements()->get_signal_types() | ranges::to_vector; })
                              | ranges::to_vector;

    auto expected_axons = std::vector<std::vector<unsigned int>>{};
    expected_axons.resize(num_ranks_cast);
    auto expected_den_ex = std::vector<std::vector<unsigned int>>{};
    expected_den_ex.resize(num_ranks_cast);
    auto expected_den_inh = std::vector<std::vector<unsigned int>>{};
    expected_den_inh.resize(num_ranks_cast);

    for (auto rank = 0; rank < num_ranks; rank++) {
        const auto rank_cast = static_cast<std::size_t>(rank);

        auto& neurons = rank_to_neurons[rank_cast];
        neurons->init_synaptic_elements({}, {}, {});

        const auto axons = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::Axon);
        const auto den_ex = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteExcitatory);
        const auto den_inh = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteInhibitory);
        expected_axons[rank_cast] = axons | ranges::to_vector;
        expected_den_ex[rank_cast] = den_ex | ranges::to_vector;
        expected_den_inh[rank_cast] = den_inh | ranges::to_vector;
    }

    auto outgoing_requests = std::vector<RelearnTypes::comm_map_deletion<SynapseDeletionRequest>>{};

    for (auto rank = 0; rank < num_ranks; rank++) {
        const auto rank_cast = static_cast<std::size_t>(rank);

        const auto& network_graph = network_graphs[rank_cast];
        const auto& disabled_neurons = rank_to_disabled_neurons[rank_cast];
        const auto disabled_neurons_vector = disabled_neurons | ranges::to_vector;
        auto& neurons = rank_to_neurons[rank_cast];

        auto expected_num_local_deletions = 0;

        for (const auto neuron_id : rank_to_disabled_neurons[rank_cast]) {
            const auto& [distant_out_edges, _3] = network_graph->get_distant_out_edges(neuron_id.get_neuron_id());
            expected_distant_out_deletions_initiated[rank_cast] += distant_out_edges.size();

            for (const auto& [target, weight] : distant_out_edges) {
                expected_distant_in_deletions_received[target.get_rank().get_rank_cast()]++;
                if (weight > 0) {
                    expected_den_ex[target.get_rank().get_rank_cast()][target.get_neuron_id().get_neuron_id()]--;
                } else {
                    expected_den_inh[target.get_rank().get_rank_cast()][target.get_neuron_id().get_neuron_id()]--;
                }
            }

            const auto& [distant_in_edges, _2] = network_graph->get_distant_in_edges(neuron_id.get_neuron_id());
            expected_distant_out_deletions_initiated[rank_cast] += distant_in_edges.size();
            for (const auto& source : distant_in_edges | ranges::views::keys) {
                expected_distant_out_deletions_received[source.get_rank().get_rank_cast()]++;
                expected_axons[source.get_rank().get_rank_cast()][source.get_neuron_id().get_neuron_id()]--;
            }

            const auto [out_edges, _1] = network_graph->get_local_out_edges(neuron_id.get_neuron_id());
            for (const auto& [target, weight] : out_edges) {
                if (weight > 0) {
                    expected_den_ex[rank_cast][target.get_neuron_id()]--;
                } else {
                    expected_den_inh[rank_cast][target.get_neuron_id()]--;
                }
                expected_num_local_deletions++;
            }

            const auto [in_edges, _4] = network_graph->get_local_in_edges(neuron_id.get_neuron_id());
            for (const auto& source : in_edges | ranges::views::keys) {
                expected_axons[rank_cast][source.get_neuron_id()]--;
                if (!disabled_neurons.contains(source)) {
                    expected_num_local_deletions++;
                }
            }
        }

        const auto& [num_deletions, synapse_deletion_Requests] = neurons->disable_neurons(1, disabled_neurons_vector,
                                                                                          num_ranks);
        ASSERT_EQ(synapse_deletion_Requests.get_total_number_requests(),
                  expected_distant_out_deletions_initiated[rank_cast] + expected_distant_in_deletions_initiated[rank_cast]);

        const auto expected_deletions = static_cast<std::size_t>(expected_num_local_deletions)
                                        + expected_distant_out_deletions_initiated[rank_cast]
                                        + expected_distant_in_deletions_initiated[rank_cast];
        ASSERT_EQ(num_deletions, expected_deletions);

        ASSERT_FALSE(synapse_deletion_Requests.contains(mpiPP::MPIRank(rank)));

        outgoing_requests.push_back(synapse_deletion_Requests);

        const auto axons = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::Axon);
        const auto den_ex = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteExcitatory);
        const auto den_inh = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteInhibitory);

        for (const auto& disable_id : disabled_neurons) {
            ASSERT_EQ(std::get<0>(network_graph->get_local_in_edges(disable_id.get_neuron_id())).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_local_out_edges(disable_id.get_neuron_id())).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_distant_in_edges(disable_id.get_neuron_id())).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_distant_out_edges(disable_id.get_neuron_id())).size(), 0);
            ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[disable_id.get_neuron_id()],
                      UpdateStatus::Disabled);
            ASSERT_EQ(axons[disable_id.get_neuron_id()], 0);
            ASSERT_EQ(den_ex[disable_id.get_neuron_id()], 0);
            ASSERT_EQ(den_inh[disable_id.get_neuron_id()], 0);
        }
    }

    const auto& ingoing_requests = MPIAdapter::exchange_requests(outgoing_requests);

    for (const auto mpi_rank : mpiPP::MPIRank::range(num_ranks)) {
        const auto rank = mpi_rank.get_rank_cast();

        const auto& synapse_deletion_Requests = ingoing_requests[rank];

        const auto& network_graph = network_graphs[rank];
        const auto& disabled_neurons = rank_to_disabled_neurons[rank];
        auto& neurons = rank_to_neurons[rank];
        const auto& num_distant_deletions = neurons->delete_disabled_distant_synapses(synapse_deletion_Requests,
                                                                                      mpi_rank);

        const auto expected_number_distant_deletions = expected_distant_in_deletions_received[rank] + expected_distant_out_deletions_received[rank];
        ASSERT_EQ(num_distant_deletions, expected_number_distant_deletions);

        const auto axons = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::Axon);
        const auto den_ex = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteExcitatory);
        const auto den_inh = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteInhibitory);

        for (const auto& disable_id : disabled_neurons) {
            ASSERT_EQ(std::get<0>(network_graph->get_local_in_edges(disable_id.get_neuron_id())).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_local_out_edges(disable_id.get_neuron_id())).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_distant_in_edges(disable_id.get_neuron_id())).size(), 0);
            ASSERT_EQ(std::get<0>(network_graph->get_distant_out_edges(disable_id.get_neuron_id())).size(), 0);
            ASSERT_EQ(neurons->get_extra_info()->get_disable_flags()[disable_id.get_neuron_id()],
                      UpdateStatus::Disabled);
            ASSERT_EQ(axons[disable_id.get_neuron_id()], 0);
            ASSERT_EQ(den_ex[disable_id.get_neuron_id()], 0);
            ASSERT_EQ(den_inh[disable_id.get_neuron_id()], 0);
        }

        for (const auto& neuron_id : NeuronID::range(num_neurons)) {
            if (disabled_neurons.contains(neuron_id)) {
                continue;
            }

            ASSERT_EQ(axons[neuron_id.get_neuron_id()],
                      expected_axons[rank][neuron_id.get_neuron_id()]);
            ASSERT_EQ(den_ex[neuron_id.get_neuron_id()],
                      expected_den_ex[rank][neuron_id.get_neuron_id()]);
            ASSERT_EQ(den_inh[neuron_id.get_neuron_id()],
                      expected_den_inh[rank][neuron_id.get_neuron_id()]);
            ASSERT_EQ(neurons->get_disable_flags()[neuron_id.get_neuron_id()], UpdateStatus::Enabled);
        }
    }

    NetworkGraphAdapter::check_validity_of_network_graphs(network_graphs, signal_types, num_neurons);
}

TEST_F(NeuronsTest, testDisableNeuronsWithRanksAndOnlyOneDisabledNeuron) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_neurons = 101;
    const auto num_ranks = 6;
    const auto num_ranks_cast = static_cast<std::size_t>(num_ranks);

    const auto& disabled_neuron_id = NeuronIdFactory::get_random_neuron_id(num_neurons, mt);
    const auto disabled_id = disabled_neuron_id.get_neuron_id();

    const auto& disabled_mpi_rank = MPIRankFactory::get_random_mpi_rank(num_ranks, mt);
    const auto disabled_rank = disabled_mpi_rank.get_rank_cast();

    auto rank_to_neurons = std::vector<std::unique_ptr<Neurons>>{};
    auto network_graphs = std::vector<std::shared_ptr<NetworkGraph>>{};

    for (const auto mpi_rank : mpiPP::MPIRank::range(num_ranks)) {
        auto partition = std::make_shared<Partition>(1, mpiPP::MPIRank(0));
        auto neurons = create_neurons_object(partition, mpi_rank, num_ranks);
        auto network_graph_plastic = neurons->get_network_graph();

        const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(mt);
        const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, num_neurons, mt);

        auto positions = std::map<NeuronID::value_type, Vec3d>{};
        for (const auto& [position, id] : neurons_to_place) {
            positions[id.get_neuron_id()] = position;
        }
        const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

        auto algorithm = AlgorithmFactory::construct_naive_algorithm({ min, max }, 0);
        neurons->set_algorithm(algorithm);
        neurons->set_probability_kernel(KernelFactory::get_standard_gaussian());

        neurons->init(num_neurons, neuron_positions);

        const auto signal_types = neurons->get_synaptic_elements()->get_signal_types();

        NetworkGraphFactory::construct_dense_plastic_network(network_graph_plastic, signal_types,
                                                             num_neurons, 2, num_ranks, mpi_rank, mt);

        Neurons::check_signal_types(network_graph_plastic, signal_types, mpi_rank);

        rank_to_neurons.emplace_back(std::move(neurons));
        network_graphs.emplace_back(network_graph_plastic);
    }

    NetworkGraphAdapter::harmonize_network_graphs_from_different_ranks(network_graphs, num_neurons);

    const auto signal_types = rank_to_neurons | ranges::views::transform([](const auto& neurons_of_rank) { return neurons_of_rank->get_synaptic_elements()->get_signal_types() | ranges::to_vector; }) | ranges::to_vector;

    auto expected_axons = std::vector<std::vector<unsigned int>>{};
    expected_axons.resize(num_ranks_cast);

    auto expected_den_ex = std::vector<std::vector<unsigned int>>{};
    expected_den_ex.resize(num_ranks_cast);

    auto expected_den_inh = std::vector<std::vector<unsigned int>>{};
    expected_den_inh.resize(num_ranks_cast);

    for (auto rank = 0; rank < num_ranks; rank++) {
        const auto rank_cast = static_cast<std::size_t>(rank);

        auto& neurons = rank_to_neurons[rank_cast];
        neurons->init_synaptic_elements({}, {}, {});

        const auto axons = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::Axon);
        const auto den_ex = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteExcitatory);
        const auto den_inh = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteInhibitory);

        expected_axons[rank_cast] = axons | ranges::to_vector;
        expected_den_ex[rank_cast] = den_ex | ranges::to_vector;
        expected_den_inh[rank_cast] = den_inh | ranges::to_vector;
    }

    auto num_distant_deletions = std::size_t{ 0 };
    auto num_local_deletions = std::size_t{ 0 };

    const auto [distant_out_edges, _1] = network_graphs[disabled_rank]->get_distant_out_edges(disabled_neuron_id.get_neuron_id());
    num_distant_deletions += distant_out_edges.size();

    const auto [distant_in_edges, _2] = network_graphs[disabled_rank]->get_distant_in_edges(disabled_neuron_id.get_neuron_id());
    num_distant_deletions += distant_in_edges.size();

    const auto [local_out_edges, _3] = network_graphs[disabled_rank]->get_local_out_edges(disabled_neuron_id.get_neuron_id());
    num_local_deletions += local_out_edges.size();

    const auto [local_in_edges, _4] = network_graphs[disabled_rank]->get_local_in_edges(disabled_neuron_id.get_neuron_id());
    num_local_deletions += local_in_edges.size();

    for (const auto& [target, weight] : distant_out_edges) {
        const auto& [rank, id] = target;
        const auto actual_rank = rank.get_rank_cast();
        const auto actual_id = id.get_neuron_id();

        if (weight > 0) {
            expected_den_ex[actual_rank][actual_id]--;
        } else {
            expected_den_inh[actual_rank][actual_id]--;
        }
    }

    for (const auto& [rank, id] : distant_in_edges | ranges::views::keys) {
        const auto actual_rank = rank.get_rank_cast();
        const auto actual_id = id.get_neuron_id();

        expected_axons[actual_rank][actual_id]--;
    }

    for (const auto& [target, weight] : local_out_edges) {
        const auto id = target.get_neuron_id();

        if (weight > 0) {
            expected_den_ex[disabled_rank][id]--;
        } else {
            expected_den_inh[disabled_rank][id]--;
        }
    }

    for (const auto& source : local_in_edges | ranges::views::keys) {
        const auto id = source.get_neuron_id();
        expected_axons[disabled_rank][id]--;
    }

    expected_axons[disabled_rank][disabled_id] = 0;
    expected_den_ex[disabled_rank][disabled_id] = 0;
    expected_den_inh[disabled_rank][disabled_id] = 0;

    auto outgoing_requests = std::vector<RelearnTypes::comm_map_deletion<SynapseDeletionRequest>>{};

    for (auto rank = 0; rank < num_ranks; rank++) {
        auto& neurons = rank_to_neurons[static_cast<std::size_t>(rank)];

        auto disable_vector = std::vector<NeuronID>{};
        if (rank == disabled_mpi_rank.get_rank()) {
            disable_vector.push_back(disabled_neuron_id);
        }

        const auto& [num_deletions, synapse_deletion_requests] = neurons->disable_neurons(1, disable_vector, num_ranks);
        ASSERT_FALSE(synapse_deletion_requests.contains(mpiPP::MPIRank(rank)));

        if (rank == disabled_mpi_rank.get_rank()) {
            ASSERT_EQ(num_deletions, num_local_deletions + num_distant_deletions);
            ASSERT_EQ(synapse_deletion_requests.get_total_number_requests(), num_distant_deletions);
        } else {
            ASSERT_EQ(num_deletions, 0);
            ASSERT_EQ(synapse_deletion_requests.get_total_number_requests(), 0);
        }

        outgoing_requests.push_back(synapse_deletion_requests);
    }

    const auto& ingoing_requests = MPIAdapter::exchange_requests(outgoing_requests);

    for (const auto mpi_rank : mpiPP::MPIRank::range(num_ranks)) {
        const auto rank = mpi_rank.get_rank_cast();

        auto& neurons = rank_to_neurons[rank];
        const auto& synapse_deletion_requests = ingoing_requests[rank];
        std::ignore = neurons->delete_disabled_distant_synapses(synapse_deletion_requests, mpi_rank);

        const auto axons = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::Axon);
        const auto den_ex = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteExcitatory);
        const auto den_inh = neurons->get_synaptic_elements()->get_connected_elements(SynapticElementType::DendriteInhibitory);

        for (const auto neuron_id : NeuronID::range(num_neurons)) {
            const auto id = neuron_id.get_neuron_id();

            const auto expected_axon_count = expected_axons[rank][id];
            const auto expected_den_ex_count = expected_den_ex[rank][id];
            const auto expected_den_inh_count = expected_den_inh[rank][id];

            const auto calculated_axon_count = axons[neuron_id.get_neuron_id()];
            const auto calculated_den_ex_count = den_ex[neuron_id.get_neuron_id()];
            const auto calculated_den_inh_count = den_inh[neuron_id.get_neuron_id()];

            ASSERT_EQ(calculated_axon_count, expected_axon_count);
            ASSERT_EQ(calculated_den_ex_count, expected_den_ex_count);
            ASSERT_EQ(calculated_den_inh_count, expected_den_inh_count);
        }
    }

    NetworkGraphAdapter::check_validity_of_network_graphs(network_graphs, signal_types, num_neurons);
}

std::unique_ptr<Neurons> NeuronsTest::create_neurons_object(std::shared_ptr<Partition>& partition, mpiPP::MPIRank rank, int number_ranks) {
    auto model = NeuronModelFactory::construct_poisson_model();
    auto calcium = std::make_unique<CalciumCalculator>();
    calcium->set_initial_calcium_calculator([](mpiPP::MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
    calcium->set_target_calcium_calculator([](mpiPP::MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
    auto network_graph = std::make_shared<NetworkGraph>(rank, number_ranks);

    auto axons = std::make_shared<Axons>();
    auto dendrites = std::make_shared<Dendrites>();

    auto synaptic_elements = std::make_shared<SynapticElements>(std::move(axons), std::move(dendrites));
    SynapticElementsAdapter::set_growthrate_calculators(synaptic_elements);

    auto sdf = std::make_unique<RandomSynapseDeletionFinder>();
    sdf->set_synaptic_elements(synaptic_elements);

    return std::make_unique<Neurons>(partition, std::move(model), std::move(calcium), std::move(network_graph), std::move(synaptic_elements), std::move(sdf));
}
