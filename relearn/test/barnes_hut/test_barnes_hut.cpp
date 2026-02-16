/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_barnes_hut.h"

#include "Config.h"

#include "algorithm/Algorithms.h"
#include "algorithm/BarnesHutInternal/BarnesHut.h"
#include "algorithm/BarnesHutInternal/BarnesHutBase.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/Cells.h"
#include "algorithm/Internal/octree/Cell.h"
#include "algorithm/Internal/octree/Octree.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "structure/Morton.h"
#include "util/NeuronID.h"
#include "util/Vec3.h"

#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "adapter/neurons/NeuronTypesAdapter.h"
#include "adapter/synaptic_elements/SynapticElementsAdapter.h"

#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"
#include "factory/synaptic_elements/synaptic_elements_factory.h"

#include <gtest/gtest.h>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>

#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <tuple>
#include <vector>

TEST_F(BarnesHutTest, testBarnesHutGetterSetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(mt);
    const auto level = std::uint8_t{ 0 };
    const auto morton = std::make_shared<Morton>(level);

    ASSERT_NO_THROW(BarnesHut algorithm(RelearnTypes::bounding_box_type{ min, max }, morton););

    auto algorithm = BarnesHut(RelearnTypes::bounding_box_type{ min, max }, morton);
    ASSERT_EQ(algorithm.get_acceptance_criterion(), Constants::bh_default_theta);

    const auto random_acceptance_criterion = RandomFactory::get_random_double<double>(0.0, Constants::bh_max_theta, mt);
    auto algorithm_2 = BarnesHut(RelearnTypes::bounding_box_type{ min, max }, morton, random_acceptance_criterion);
    ASSERT_EQ(algorithm_2.get_acceptance_criterion(), random_acceptance_criterion);
}

TEST_F(BarnesHutTest, testUpdateFunctor) {
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

    const auto level = std::uint8_t{ 0 };
    const auto morton = std::make_shared<Morton>(level);

    const auto signal_types = SynapticElementsFactory::get_signal_types(number_excitatory_neurons, number_inhibitory_neurons, mt);
    const auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements(signal_types);
    SynapticElementsAdapter::grow_and_connect(synaptic_elements, mt);

    const auto& neurons_to_place = NeuronsFactory::generate_random_neurons(min, max, number_neurons, mt);

    auto positions = std::map<NeuronID::value_type, Vec3d>{};
    for (const auto& [position, id] : neurons_to_place) {
        positions[id.get_neuron_id()] = position;
    }
    const auto neuron_positions = positions | ranges::views::values | ranges::to_vector;

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    auto barnes_hut = BarnesHut(RelearnTypes::bounding_box_type{ min, max }, morton);
    barnes_hut.set_synaptic_elements(synaptic_elements);
    barnes_hut.set_neuron_extra_infos(extra_infos);
    barnes_hut.init(number_neurons);

    const auto update_status = NeuronTypesFactory::get_update_status(number_neurons, mt);

    const auto disabled_neurons = NeuronID::range(number_neurons)
                                  | ranges::views::filter(utility::equal_to(UpdateStatus::Disabled), utility::lookup(update_status, &NeuronID::get_neuron_id))
                                  | ranges::to_vector;

    extra_infos->set_disabled_neurons(disabled_neurons);

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    ASSERT_NO_THROW(barnes_hut.prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));

    const auto& octree = barnes_hut.get_octree();

    auto stack = std::stack<OctreeNode<BarnesHutCell>*>{};
    stack.push(octree->get_root());

    while (!stack.empty()) {
        auto* node = stack.top();
        stack.pop();

        const auto& cell = node->get_cell();

        if (node->is_leaf()) {
            const auto id = cell.get_neuron_id();
            const auto local_id = id.get_neuron_id();

            ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
            ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());

            const auto& golden_position = positions[local_id];

            ASSERT_EQ(cell.get_excitatory_dendrites_position().value(), golden_position);
            ASSERT_EQ(cell.get_inhibitory_dendrites_position().value(), golden_position);

            if (update_status[local_id] == UpdateStatus::Disabled) {
                ASSERT_EQ(cell.get_number_excitatory_dendrites(), 0);
                ASSERT_EQ(cell.get_number_inhibitory_dendrites(), 0);
            } else {
                const auto golden_excitatory_dendrites = vacant_excitatory_dendrites[id.get_neuron_id()];
                const auto golden_inhibitory_dendrites = vacant_inhibitory_dendrites[id.get_neuron_id()];

                ASSERT_EQ(cell.get_number_excitatory_dendrites(), golden_excitatory_dendrites);
                ASSERT_EQ(cell.get_number_inhibitory_dendrites(), golden_inhibitory_dendrites);
            }
        } else {
            auto total_number_excitatory_dendrites = 0U;
            auto total_number_inhibitory_dendrites = 0U;

            auto excitatory_dendrites_position = Vec3d{ 0, 0, 0 };
            auto inhibitory_dendrites_position = Vec3d{ 0, 0, 0 };

            for (auto* child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                const auto& child_cell = child->get_cell();

                const auto number_excitatory_dendrites = child_cell.get_number_excitatory_dendrites();
                const auto number_inhibitory_dendrites = child_cell.get_number_inhibitory_dendrites();

                total_number_excitatory_dendrites += number_excitatory_dendrites;
                total_number_inhibitory_dendrites += number_inhibitory_dendrites;

                if (number_excitatory_dendrites != 0) {
                    const auto& opt = child_cell.get_excitatory_dendrites_position();
                    ASSERT_TRUE(opt.has_value());
                    const auto& position = opt.value();

                    excitatory_dendrites_position += (position * number_excitatory_dendrites);
                }

                if (number_inhibitory_dendrites != 0) {
                    const auto& opt = child_cell.get_inhibitory_dendrites_position();
                    ASSERT_TRUE(opt.has_value());
                    const auto& position = opt.value();

                    inhibitory_dendrites_position += (position * number_inhibitory_dendrites);
                }

                stack.push(child);
            }

            ASSERT_EQ(total_number_excitatory_dendrites, cell.get_number_excitatory_dendrites());
            ASSERT_EQ(total_number_inhibitory_dendrites, cell.get_number_inhibitory_dendrites());

            if (total_number_excitatory_dendrites == 0) {
                ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
            } else {
                const auto& opt = cell.get_excitatory_dendrites_position();
                ASSERT_TRUE(opt.has_value());
                const auto& position = opt.value();

                const auto& diff = (excitatory_dendrites_position / total_number_excitatory_dendrites) - position;
                const auto& norm = diff.calculate_2_norm();

                ASSERT_NEAR(norm, 0.0, eps);
            }

            if (total_number_inhibitory_dendrites == 0) {
                ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
            } else {
                const auto& opt = cell.get_inhibitory_dendrites_position();
                ASSERT_TRUE(opt.has_value());
                const auto& position = opt.value();

                const auto& diff = (inhibitory_dendrites_position / total_number_inhibitory_dendrites) - position;
                const auto& norm = diff.calculate_2_norm();

                ASSERT_NEAR(norm, 0.0, eps);
            }
        }
    }
}
