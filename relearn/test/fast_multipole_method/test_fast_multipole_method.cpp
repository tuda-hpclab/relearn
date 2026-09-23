/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_fast_multipole_method.h"

#include "Config.h"

#include "algorithm/Algorithm.h"
#include "algorithm/FMMInternal/FastMultipoleMethod.h"
#include "algorithm/FMMInternal/FastMultipoleMethodBase.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/FMMInternal/FastMultipoleMethodInverted.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/Kernel/Gaussian.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "structure/Morton.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/BoundingBox.h"
#include "util/MemoryHolder.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"
#include "util/shuffle/shuffle.h"

#include "adapter/octree/OctreeAdapter.h"

#include "factory/fmm/fmm_factory.h"
#include "factory/kernel/kernel_factory.h"
#include "factory/memory_holder/memory_holder_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/octree/octree_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"
#include "factory/synaptic_elements/synaptic_elements_factory.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/ranges/Functional.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/patterns/MPIAdvancedCommunicationPatterns.h>

#include <range/v3/action/sort.hpp>
#include <range/v3/algorithm/any_of.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#ifndef RELEARN_CUDA_ENABLED

TEST_F(FMMTest, testMultiIndexGetNumberOfIndices) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_EQ(MultiIndex::get_number_of_indices(), Constants::p3);
}

TEST_F(FMMTest, testMultIndexGetIndices) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto expected_indices = std::vector<Vec3u>{};
    expected_indices.reserve(MultiIndex::get_number_of_indices());

    for (auto x = 0U; x < Constants::p; x++) {
        for (auto y = 0U; y < Constants::p; y++) {
            for (auto z = 0U; z < Constants::p; z++) {
                expected_indices.emplace_back(x, y, z);
            }
        }
    }
    ranges::sort(expected_indices, std::less{});

    const auto actual_indices_vector = MultiIndex::get_indices() | ranges::to_vector | ranges::actions::sort(std::less{});

    ASSERT_EQ(expected_indices, actual_indices_vector);
}

TEST_F(FMMTest, testH) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto t = RandomFactory::get_random_double(RelearnTypes::attraction_type{ -10 }, RelearnTypes::attraction_type{ 10 }, mt);
    const auto alpha = RandomFactory::get_random_integer<unsigned int>(0, 8, mt);

    const auto squared_t = t * t;
    const auto exponented_t = std::exp(-squared_t);

    const auto hermite_t_n = std::hermite(alpha, t);

    const auto multiplied = exponented_t * hermite_t_n;

    ASSERT_NEAR(multiplied, FastMultipoleMethodBase::h(alpha, t), eps) << t << ' ' << alpha << '\n';
}

TEST_F(FMMTest, testHMultiIndex) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto multi_index = FMMFactory::get_random_multi_index(mt);
    const auto position = SimulationFactory::get_random_position(mt);

    const auto actual_value = FastMultipoleMethodBase::h_multi_index(multi_index, position);

    const auto [multi_index_x, multi_index_y, multi_index_z] = multi_index;
    const auto [position_x, position_y, position_z] = position;

    const auto val_x = FastMultipoleMethodBase::h(multi_index_x, position_x);
    const auto val_y = FastMultipoleMethodBase::h(multi_index_y, position_y);
    const auto val_z = FastMultipoleMethodBase::h(multi_index_z, position_z);

    const auto product = val_x * val_y * val_z;

    ASSERT_NEAR(actual_value, product, eps);
}

TEST_F(FMMTest, testExtractElement) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_pointers = NeuronIdFactory::get_random_number_neurons(mt);
    const auto number_nullptrs = NeuronIdFactory::get_random_number_neurons(mt);

    auto memory_holder = std::vector<OctreeNode<FastMultipoleMethodCell>>(number_pointers);

    const auto num_nullptrs = static_cast<std::ptrdiff_t>(number_nullptrs);

    const auto pointers = ranges::views::concat(
                              memory_holder | ranges::views::transform([](auto& Val) { return &Val; }),
                              ranges::views::repeat_n(
                                  static_cast<OctreeNode<FastMultipoleMethodCell>*>(nullptr),
                                  num_nullptrs))
                          | ranges::to_vector | actions::shuffle(mt);

    auto received_pointers = std::vector<OctreeNode<FastMultipoleMethodCell>*>{};
    received_pointers.reserve(number_pointers);

    for (auto pointer_index = 0U; pointer_index < number_pointers; pointer_index++) {
        auto* ptr = FastMultipoleMethodBase::extract_element(pointers, pointer_index);
        ASSERT_NE(ptr, nullptr);

        received_pointers.emplace_back(ptr);
    }

    ranges::sort(received_pointers);

    for (auto i = 0U; i < number_pointers; i++) {
        ASSERT_EQ(received_pointers[i], &memory_holder[i]);
    }

    for (auto pointer_index = number_pointers; pointer_index < number_pointers + number_nullptrs + number_neurons_out_of_scope; pointer_index++) {
        auto* ptr = FastMultipoleMethodBase::extract_element(pointers, pointer_index);
        ASSERT_EQ(ptr, nullptr);
    }
}

TEST_F(FMMTest, testCheckCalculationRequirementsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto node = OctreeNode<FastMultipoleMethodCell>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(nullptr, nullptr, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(nullptr, nullptr, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(nullptr, nullptr, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(nullptr, nullptr, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(&node, nullptr, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(&node, nullptr, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(&node, nullptr, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(&node, nullptr, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(nullptr, &node, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(nullptr, &node, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(nullptr, &node, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::check_calculation_requirements(nullptr, &node, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
}

TEST_F(FMMTest, testCheckCalculationRequirementsLeaf) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto node_1 = OctreeNode<FastMultipoleMethodCell>{};
    node_1.set_level(0);
    node_1.set_rank(my_rank);
    node_1.set_cell_size(min, max);
    node_1.set_cell_neuron_id(NeuronID(0));
    node_1.set_cell_neuron_position(SimulationFactory::get_random_position_in_box(min, max, this->mt));
    node_1.set_cell_number_axons(1, 1);
    node_1.set_cell_number_dendrites(1, 1);

    auto node_2 = OctreeNode<FastMultipoleMethodCell>{};
    node_2.set_level(0);
    node_2.set_rank(my_rank);
    node_2.set_cell_size(min, max);
    node_2.set_cell_neuron_id(NeuronID(0));
    node_2.set_cell_neuron_position(SimulationFactory::get_random_position_in_box(min, max, this->mt));
    node_2.set_cell_number_axons(0, 0);
    node_2.set_cell_number_dendrites(0, 0);

    auto root = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(20, memory_holder, min, max, this->mt);

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &node_2, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &node_2, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &node_2, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &node_2, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &node_1, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &node_1, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &node_1, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &node_1, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &root, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &root, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &root, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_1, &root, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_1, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_1, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_1, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_1, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &root, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &root, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &root, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&node_2, &root, ElementType::Axon, SignalType::Inhibitory));

    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_2, ElementType::Dendrite, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_2, ElementType::Dendrite, SignalType::Inhibitory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_2, ElementType::Axon, SignalType::Excitatory));
    ASSERT_EQ(CalculationType::Direct, FastMultipoleMethodBase::check_calculation_requirements(&root, &node_2, ElementType::Axon, SignalType::Inhibitory));
}

TEST_F(FMMTest, testCheckCalculationRequirements) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);

    const auto number_neurons_in_source = static_cast<RelearnTypes::number_neurons_type>(Constants::max_neurons_in_source * 0.75);
    const auto number_neurons_in_target = static_cast<RelearnTypes::number_neurons_type>(Constants::max_neurons_in_target * 0.75);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto source = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(number_neurons_in_source, memory_holder, min, max, this->mt);
    auto target = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(number_neurons_in_target, memory_holder, min, max, this->mt);

    const auto& source_cell = source.get_cell();
    const auto& target_cell = target.get_cell();

    const auto check_combi = [&](const ElementType e, const SignalType s) {
        const auto has_enough_in_source = source_cell.get_number_elements_for(get_other_element_type(e), s) > Constants::max_neurons_in_source;
        const auto has_enough_in_target = target_cell.get_number_elements_for(e, s) > Constants::max_neurons_in_target;

        const auto type = FastMultipoleMethodBase::check_calculation_requirements(&source, &target, e, s);

        if (has_enough_in_source && has_enough_in_target) {
            ASSERT_EQ(CalculationType::Hermite, type);
        } else if (has_enough_in_target) {
            ASSERT_EQ(CalculationType::Taylor, type);
        } else {
            ASSERT_EQ(CalculationType::Direct, type);
        }
    };

    check_combi(ElementType::Axon, SignalType::Excitatory);
    check_combi(ElementType::Dendrite, SignalType::Excitatory);
    check_combi(ElementType::Axon, SignalType::Inhibitory);
    check_combi(ElementType::Dendrite, SignalType::Inhibitory);
}

TEST_F(FMMTest, testDirectGaussException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto node = OctreeNode<FastMultipoleMethodCell>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, nullptr, nullptr, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, nullptr, nullptr, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, nullptr, nullptr, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, nullptr, nullptr, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, &node, nullptr, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, &node, nullptr, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, &node, nullptr, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, &node, nullptr, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, nullptr, &node, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, nullptr, &node, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, nullptr, &node, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, nullptr, &node, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
}

TEST_F(FMMTest, testDirectGauss) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);

    const auto number_neurons_in_source = static_cast<RelearnTypes::number_neurons_type>(Constants::max_neurons_in_source * 0.2);
    const auto number_neurons_in_target = static_cast<RelearnTypes::number_neurons_type>(Constants::max_neurons_in_target * 0.2);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto source = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(number_neurons_in_source, memory_holder, min, max, this->mt);
    auto target = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(number_neurons_in_target, memory_holder, min, max, this->mt);

    const auto& source_leaves = OctreeAdapter::extract_leaf_nodes(&source);
    const auto& target_leaves = OctreeAdapter::extract_leaf_nodes(&target);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto check_combi = [&](const ElementType e, const SignalType s) {
        auto sum = RelearnTypes::attraction_type{ 0 };

        const auto sigma = RelearnTypes::attraction_type{ 150 };

        for (const auto* source_leaf : source_leaves) {
            for (const auto* target_leaf : target_leaves) {
                const auto vacant_sources = source_leaf->get_cell().get_number_elements_for(get_other_element_type(e), s);
                const auto vacant_targets = target_leaf->get_cell().get_number_elements_for(e, s);

                const auto product = vacant_sources * vacant_targets;
                if (product == 0) {
                    continue;
                }

                const auto attraction = FastMultipoleMethodBase::kernel(source_leaf->get_cell().get_position_for(get_other_element_type(e), s).value(),
                                                                        target_leaf->get_cell().get_position_for(e, s).value(), sigma);

                sum += attraction * static_cast<RelearnTypes::attraction_type>(product);
            }
        }

        ASSERT_NEAR(sum, FastMultipoleMethodBase::calc_direct_gauss(sigma, node_cache, &source, &target, e, s), eps);
    };

    check_combi(ElementType::Axon, SignalType::Excitatory);
    check_combi(ElementType::Dendrite, SignalType::Excitatory);
    check_combi(ElementType::Axon, SignalType::Inhibitory);
    check_combi(ElementType::Dendrite, SignalType::Inhibitory);
}

TEST_F(FMMTest, testHermiteCoefficientsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto node = OctreeNode<FastMultipoleMethodCell>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, nullptr, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, nullptr, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, nullptr, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, nullptr, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &node, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &node, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &node, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &node, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
}

TEST_F(FMMTest, testHermiteCoefficientsException2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto no_axon_tree = OctreeFactory::get_tree_no_axons<FastMultipoleMethodCell>(number_neurons, memory_holder, min, max, mt);
    auto no_dendrite_tree = OctreeFactory::get_tree_no_dendrites<FastMultipoleMethodCell>(number_neurons, memory_holder, min, max, mt);
    auto no_synaptic_elements_tree = OctreeFactory::get_tree_no_synaptic_elements<FastMultipoleMethodCell>(number_neurons, memory_holder, min, max, mt);

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &no_axon_tree, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &no_axon_tree, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &no_dendrite_tree, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &no_dendrite_tree, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &no_synaptic_elements_tree, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &no_synaptic_elements_tree, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &no_synaptic_elements_tree, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &no_synaptic_elements_tree, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
}

TEST_F(FMMTest, testHermiteCoefficientsForm) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(number_neurons, memory_holder, min, max, mt);

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    const auto coefficients_a_e = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &tree, ElementType::Axon, SignalType::Excitatory);
    ASSERT_EQ(coefficients_a_e.size(), Constants::p3);
    ASSERT_TRUE(ranges::any_of(coefficients_a_e, utility::not_equal_to(0.0)));

    const auto coefficients_a_i = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &tree, ElementType::Axon, SignalType::Inhibitory);
    ASSERT_EQ(coefficients_a_i.size(), Constants::p3);
    ASSERT_TRUE(ranges::any_of(coefficients_a_i, utility::not_equal_to(0.0)));

    const auto coefficients_d_e = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &tree, ElementType::Dendrite, SignalType::Excitatory);
    ASSERT_EQ(coefficients_d_e.size(), Constants::p3);
    ASSERT_TRUE(ranges::any_of(coefficients_d_e, utility::not_equal_to(0.0)));

    const auto coefficients_d_i = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &tree, ElementType::Dendrite, SignalType::Inhibitory);
    ASSERT_EQ(coefficients_d_i.size(), Constants::p3);
    ASSERT_TRUE(ranges::any_of(coefficients_d_i, utility::not_equal_to(0.0)));
}

TEST_F(FMMTest, testHermiteCoefficientsValues) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    auto tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(number_neurons, memory_holder, min, max, mt);
    const auto coefficients_1 = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &tree, ElementType::Axon, SignalType::Excitatory);

    OctreeAdapter::invalidate_elements<FastMultipoleMethodCell>(&tree, ElementType::Dendrite, SignalType::Excitatory);
    const auto coefficients_2 = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &tree, ElementType::Axon, SignalType::Excitatory);

    ASSERT_EQ(coefficients_1, coefficients_2);

    OctreeAdapter::invalidate_elements<FastMultipoleMethodCell>(&tree, ElementType::Dendrite, SignalType::Inhibitory);
    const auto coefficients_3 = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &tree, ElementType::Axon, SignalType::Excitatory);

    ASSERT_EQ(coefficients_1, coefficients_3);

    OctreeAdapter::invalidate_elements<FastMultipoleMethodCell>(&tree, ElementType::Axon, SignalType::Inhibitory);
    const auto coefficients_4 = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &tree, ElementType::Axon, SignalType::Excitatory);

    ASSERT_EQ(coefficients_1, coefficients_4);
}

TEST_F(FMMTest, testTaylorCoefficientsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);
    const auto& other_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto node = OctreeNode<FastMultipoleMethodCell>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor_coefficients(sigma, nullptr, other_position, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor_coefficients(sigma, nullptr, other_position, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor_coefficients(sigma, nullptr, other_position, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor_coefficients(sigma, nullptr, other_position, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &node, other_position, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &node, other_position, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &node, other_position, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &node, other_position, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
}

TEST_F(FMMTest, testTaylorCoefficientsZero) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& other_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto no_axon_tree = OctreeFactory::get_tree_no_axons<FastMultipoleMethodCell>(number_neurons, memory_holder, min, max, mt);
    auto no_dendrite_tree = OctreeFactory::get_tree_no_dendrites<FastMultipoleMethodCell>(number_neurons, memory_holder, min, max, mt);
    auto no_synaptic_elements_tree = OctreeFactory::get_tree_no_synaptic_elements<FastMultipoleMethodCell>(number_neurons, memory_holder, min, max, mt);

    const auto coefficients = std::vector<RelearnTypes::attraction_type>(Constants::p3, 0.0);

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &no_axon_tree, other_position, ElementType::Axon, SignalType::Excitatory), coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &no_axon_tree, other_position, ElementType::Axon, SignalType::Inhibitory), coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &no_dendrite_tree, other_position, ElementType::Dendrite, SignalType::Excitatory), coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &no_dendrite_tree, other_position, ElementType::Dendrite, SignalType::Inhibitory), coefficients);

    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &no_synaptic_elements_tree, other_position, ElementType::Axon, SignalType::Excitatory), coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &no_synaptic_elements_tree, other_position, ElementType::Axon, SignalType::Inhibitory), coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &no_synaptic_elements_tree, other_position, ElementType::Dendrite, SignalType::Excitatory), coefficients);
    ASSERT_EQ(FastMultipoleMethodBase::calc_taylor_coefficients(sigma, &no_synaptic_elements_tree, other_position, ElementType::Dendrite, SignalType::Inhibitory), coefficients);
}
/*
TEST_F(FMMTest, testTaylorCoefficientsForm) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() ==  mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& other_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 10;

    auto tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(number_neurons, min, max, mt);

    const auto coefficients_a_e = FastMultipoleMethodBase::calc_taylor_coefficients(&tree, other_position, ElementType::Axon, SignalType::Excitatory);
    ASSERT_EQ(coefficients_a_e.size(), Constants::p3);
    ASSERT_TRUE(ranges::any_of(coefficients_a_e, utility::not_equal_to(0.0)));

    const auto coefficients_a_i = FastMultipoleMethodBase::calc_taylor_coefficients(&tree, other_position, ElementType::Axon, SignalType::Inhibitory);
    ASSERT_EQ(coefficients_a_i.size(), Constants::p3);
    ASSERT_TRUE(ranges::any_of(coefficients_a_i, utility::not_equal_to(0.0)));

    const auto coefficients_d_e = FastMultipoleMethodBase::calc_taylor_coefficients(&tree, other_position, ElementType::Dendrite, SignalType::Excitatory);
    ASSERT_EQ(coefficients_d_e.size(), Constants::p3);
    ASSERT_TRUE(ranges::any_of(coefficients_d_e, utility::not_equal_to(0.0)));

    const auto coefficients_d_i = FastMultipoleMethodBase::calc_taylor_coefficients(&tree, other_position, ElementType::Dendrite, SignalType::Inhibitory);
    ASSERT_EQ(coefficients_d_i.size(), Constants::p3);
    ASSERT_TRUE(ranges::any_of(coefficients_d_i, utility::not_equal_to(0.0)));
}*/

TEST_F(FMMTest, testCalcHermiteException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto box = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto min = box.get_minimum();
    const auto max = box.get_maximum();

    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto node = OctreeNode<FastMultipoleMethodCell>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    auto coefficients = std::vector<RelearnTypes::attraction_type>(Constants::p3, 0.0);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, nullptr, nullptr, coefficients, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, nullptr, nullptr, coefficients, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, nullptr, nullptr, coefficients, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, nullptr, nullptr, coefficients, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &node, nullptr, coefficients, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &node, nullptr, coefficients, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &node, nullptr, coefficients, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &node, nullptr, coefficients, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, nullptr, &node, coefficients, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, nullptr, &node, coefficients, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, nullptr, &node, coefficients, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, nullptr, &node, coefficients, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    auto source_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(30, memory_holder, min, max, this->mt);
    auto target_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(40, memory_holder, min, max, this->mt);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, {}, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, {}, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, {}, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, {}, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, std::vector<RelearnTypes::attraction_type>{ RelearnTypes::attraction_type{ 1 }, utility::as<RelearnTypes::attraction_type>(2.9) }, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, std::vector<RelearnTypes::attraction_type>{ RelearnTypes::attraction_type{ 1 }, utility::as<RelearnTypes::attraction_type>(2.9) }, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, std::vector<RelearnTypes::attraction_type>{ RelearnTypes::attraction_type{ 1 }, utility::as<RelearnTypes::attraction_type>(2.9) }, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, std::vector<RelearnTypes::attraction_type>{ RelearnTypes::attraction_type{ 1 }, utility::as<RelearnTypes::attraction_type>(2.9) }, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &node, coefficients, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &node, coefficients, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &node, coefficients, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &node, coefficients, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    const auto test_combi = [&min, &max, &coefficients, &node_cache, sigma, this](const ElementType element_type, const SignalType signal_type) {
        auto _memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

        auto source_invalidated = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(30, _memory_holder, min, max, this->mt);
        auto target = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(40, _memory_holder, min, max, this->mt);

        OctreeAdapter::invalidate_elements<FastMultipoleMethodCell>(&source_invalidated, element_type, signal_type);

        ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_hermite(utility::cast<RelearnTypes::attraction_type>(sigma), node_cache, &source_invalidated, &target, coefficients, element_type, signal_type), RelearnException);
    };

    test_combi(ElementType::Axon, SignalType::Excitatory);
    test_combi(ElementType::Axon, SignalType::Inhibitory);
    test_combi(ElementType::Dendrite, SignalType::Excitatory);
    test_combi(ElementType::Dendrite, SignalType::Inhibitory);
}

TEST_F(FMMTest, testCalcTaylorException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto box = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto min = box.get_minimum();
    const auto max = box.get_maximum();
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto node = OctreeNode<FastMultipoleMethodCell>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, nullptr, nullptr, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, nullptr, nullptr, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, nullptr, nullptr, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, nullptr, nullptr, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &node, nullptr, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &node, nullptr, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &node, nullptr, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &node, nullptr, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, nullptr, &node, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, nullptr, &node, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, nullptr, &node, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, nullptr, &node, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    auto target_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(40, memory_holder, min, max, this->mt);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &node, &target_tree, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &node, &target_tree, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &node, &target_tree, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &node, &target_tree, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    const auto test_combi = [&min, &max, &node_cache, sigma, this](const ElementType element_type, const SignalType signal_type) {
        auto _memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

        auto source = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(30, _memory_holder, min, max, this->mt);
        auto target_invalidated = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(40, _memory_holder, min, max, this->mt);

        OctreeAdapter::invalidate_elements<FastMultipoleMethodCell>(&target_invalidated, get_other_element_type(element_type), signal_type);

        ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_taylor(utility::cast<RelearnTypes::attraction_type>(sigma), node_cache, &source, &target_invalidated, element_type, signal_type), RelearnException);
    };

    test_combi(ElementType::Axon, SignalType::Excitatory);
    test_combi(ElementType::Axon, SignalType::Inhibitory);
    test_combi(ElementType::Dendrite, SignalType::Excitatory);
    test_combi(ElementType::Dendrite, SignalType::Inhibitory);
}

TEST_F(FMMTest, testCalcHermiteComparisons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_small_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto source_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(30, memory_holder, min, max, this->mt);
    auto target_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(40, memory_holder, min + RelearnTypes::position_type{ 2000.0 }, max + RelearnTypes::position_type{ 2000.0 }, this->mt);

    const auto searching_element_type = NeuronTypesFactory::get_random_element_type(this->mt);
    const auto signal_type = NeuronTypesFactory::get_random_signal_type(this->mt);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    const auto coefficients_1 = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &source_tree, searching_element_type, signal_type);
    const auto value_1 = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, coefficients_1, searching_element_type, signal_type);

    OctreeAdapter::increase_number_elements(&source_tree, searching_element_type, signal_type);

    const auto coefficients_2 = FastMultipoleMethodBase::calc_hermite_coefficients(sigma, &source_tree, searching_element_type, signal_type);
    const auto value_2 = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, coefficients_2, searching_element_type, signal_type);

    if (value_2 == RelearnTypes::attraction_type{ 0 }) {
        ASSERT_NEAR(value_1, 0.0, eps);
    } else if (value_2 > utility::as<RelearnTypes::attraction_type>(1e-8)) {
        ASSERT_GT(value_2, value_1);
    }

    OctreeAdapter::increase_number_elements(&target_tree, get_other_element_type(searching_element_type), signal_type);

    const auto value_3 = FastMultipoleMethodBase::calc_hermite(sigma, node_cache, &source_tree, &target_tree, coefficients_2, searching_element_type, signal_type);

    if (value_3 == RelearnTypes::attraction_type{ 0 }) {
        ASSERT_EQ(value_2, 0.0);
    } else if (value_3 > utility::as<RelearnTypes::attraction_type>(1e-8)) {
        ASSERT_GT(value_3, value_2);
    }
}

TEST_F(FMMTest, testCalcTaylorComparisons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_small_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto source_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(30, memory_holder, min, max, this->mt);
    auto target_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(40, memory_holder, min + RelearnTypes::position_type{ 2000.0 }, max + RelearnTypes::position_type{ 2000.0 }, this->mt);

    const auto searching_element_type = NeuronTypesFactory::get_random_element_type(this->mt);
    const auto signal_type = NeuronTypesFactory::get_random_signal_type(this->mt);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    const auto value_1 = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &source_tree, &target_tree, searching_element_type, signal_type);

    OctreeAdapter::increase_number_elements(&source_tree, searching_element_type, signal_type);

    const auto value_2 = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &source_tree, &target_tree, searching_element_type, signal_type);

    if (value_2 < utility::cast<RelearnTypes::attraction_type>(eps)) {
        ASSERT_LT(value_1, utility::cast<RelearnTypes::attraction_type>(eps));
    } else {
        ASSERT_GT(value_2, value_1);
    }

    OctreeAdapter::increase_number_elements(&target_tree, get_other_element_type(searching_element_type), signal_type);

    const auto value_3 = FastMultipoleMethodBase::calc_taylor(sigma, node_cache, &source_tree, &target_tree, searching_element_type, signal_type);

    if (value_3 < utility::cast<RelearnTypes::attraction_type>(eps)) {
        ASSERT_LT(value_2, utility::cast<RelearnTypes::attraction_type>(eps));
    } else {
        ASSERT_GT(value_3, value_2);
    }
}

TEST_F(FMMTest, testCalcAttractivenessToConnectException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_attractiveness_to_connect(sigma, node_cache, nullptr, {}, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_attractiveness_to_connect(sigma, node_cache, nullptr, {}, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_attractiveness_to_connect(sigma, node_cache, nullptr, {}, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_attractiveness_to_connect(sigma, node_cache, nullptr, {}, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& own_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto node = OctreeNode<FastMultipoleMethodCell>{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    const auto targets = std::vector<OctreeNode<FastMultipoleMethodCell>*>{ &node };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_attractiveness_to_connect(sigma, node_cache, nullptr, targets, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_attractiveness_to_connect(sigma, node_cache, nullptr, targets, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_attractiveness_to_connect(sigma, node_cache, nullptr, targets, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::calc_attractiveness_to_connect(sigma, node_cache, nullptr, targets, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
}
/*
TEST_F(FMMTest, testCalcAttractivenessToConnect) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() ==  mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_small_simulation_box_size(this->mt);

    auto source_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(30, min, max, this->mt);

    const auto number_targets = NeuronIdFactory::get_random_number_neurons(this->mt) / 50 + 10;

    const auto searching_element_type = NeuronTypesFactory::get_random_element_type(this->mt);
    const auto signal_type = NeuronTypesFactory::get_random_signal_type(this->mt);

    auto targets_holder = std::vector<OctreeNode<FastMultipoleMethodCell>>{};
    for (auto i = 0U; i < number_targets; i++) {
        targets_holder.emplace_back(OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(40, min + RelearnTypes::position_type{ 2000.0 }, max + RelearnTypes::position_type{ 2000.0 }, this->mt));
    }

    auto targets = std::vector<OctreeNode<FastMultipoleMethodCell>*>{};
    for (auto i = 0U; i < number_targets; i++) {
        targets.emplace_back(&targets_holder[i]);
        targets.emplace_back(nullptr);
    }

    const auto& attractivenesses = FastMultipoleMethodBase::calc_attractiveness_to_connect(&source_tree, targets, searching_element_type, signal_type);

    ASSERT_EQ(attractivenesses.size(), number_targets);

    for (auto i = 0U; i < targets.size(); i += 2) {
        const auto calculation_type = FastMultipoleMethodBase::check_calculation_requirements(&source_tree, targets[i], searching_element_type, signal_type);
        if (calculation_type == CalculationType::Direct) {
            const auto direct_attractiveness = FastMultipoleMethodBase::calc_direct_gauss(&source_tree, targets[i], searching_element_type, signal_type);
            ASSERT_EQ(attractivenesses[i / 2], direct_attractiveness);
        } else if (calculation_type == CalculationType::Hermite) {
            const auto hermite_coefficients = FastMultipoleMethodBase::calc_hermite_coefficients(&source_tree, searching_element_type, signal_type);
            const auto hermite_attractiveness = FastMultipoleMethodBase::calc_hermite(&source_tree, targets[i], hermite_coefficients, searching_element_type, signal_type);
            ASSERT_EQ(attractivenesses[i / 2], hermite_attractiveness);
        } else {
            const auto taylor_attractiveness = FastMultipoleMethodBase::calc_taylor(&source_tree, targets[i], searching_element_type, signal_type);
            ASSERT_EQ(attractivenesses[i / 2], taylor_attractiveness);
        }
    }
}*/

TEST_F(FMMTest, testUnpackLevelsException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 0, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 0, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 0, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 0, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 1, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 1, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 1, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 1, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 4, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 4, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 4, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::unpack_levels(nullptr, node_cache, 4, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
}

TEST_F(FMMTest, testUnpackLevelsNoElements) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto check_axons = [&node_cache](OctreeNode<FastMultipoleMethodCell>* ptr, RelearnTypes::level_type level) {
        const auto targets_1 = FastMultipoleMethodBase::unpack_levels(ptr, node_cache, level, ElementType::Axon, SignalType::Excitatory);
        ASSERT_TRUE(targets_1.empty());

        const auto targets_2 = FastMultipoleMethodBase::unpack_levels(ptr, node_cache, level, ElementType::Axon, SignalType::Inhibitory);
        ASSERT_TRUE(targets_2.empty());
    };

    const auto check_dendrites = [&node_cache](OctreeNode<FastMultipoleMethodCell>* ptr, RelearnTypes::level_type level) {
        const auto targets_1 = FastMultipoleMethodBase::unpack_levels(ptr, node_cache, level, ElementType::Dendrite, SignalType::Excitatory);
        ASSERT_TRUE(targets_1.empty());

        const auto targets_2 = FastMultipoleMethodBase::unpack_levels(ptr, node_cache, level, ElementType::Dendrite, SignalType::Inhibitory);
        ASSERT_TRUE(targets_2.empty());
    };

    const auto& [min, max] = SimulationFactory::get_random_small_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto empty_tree = OctreeFactory::get_tree_no_synaptic_elements<FastMultipoleMethodCell>(3000, memory_holder, min, max, this->mt);
    check_axons(&empty_tree, 0);
    check_axons(&empty_tree, 1);
    check_axons(&empty_tree, 4);
    check_dendrites(&empty_tree, 0);
    check_dendrites(&empty_tree, 1);
    check_dendrites(&empty_tree, 4);

    auto no_axons_tree = OctreeFactory::get_tree_no_axons<FastMultipoleMethodCell>(4000, memory_holder, min, max, this->mt);
    check_axons(&no_axons_tree, 0);
    check_axons(&no_axons_tree, 1);
    check_axons(&no_axons_tree, 4);

    auto no_dendrites_tree = OctreeFactory::get_tree_no_dendrites<FastMultipoleMethodCell>(4000, memory_holder, min, max, this->mt);
    check_dendrites(&no_dendrites_tree, 0);
    check_dendrites(&no_dendrites_tree, 1);
    check_dendrites(&no_dendrites_tree, 4);
}

TEST_F(FMMTest, testUnpackLevels) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto check = [&node_cache](OctreeNode<FastMultipoleMethodCell>* ptr, RelearnTypes::level_type level, ElementType element_type, SignalType signal_type) {
        auto targets = FastMultipoleMethodBase::unpack_levels(ptr, node_cache, level, element_type, signal_type);
        for (auto* target : targets) {
            ASSERT_NE(target, nullptr);

            ASSERT_LE(ptr->get_level(), target->get_level());
            ASSERT_LE(target->get_level(), ptr->get_level() + level);

            if (target->get_level() < ptr->get_level() + level) {
                ASSERT_TRUE(target->is_leaf());
            }

            ASSERT_GT(target->get_cell().get_number_elements_for(element_type, signal_type), 0);
        }
    };

    const auto& [min, max] = SimulationFactory::get_random_small_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(100, memory_holder, min, max, this->mt);

    check(&tree, 0, ElementType::Axon, SignalType::Excitatory);
    check(&tree, 0, ElementType::Axon, SignalType::Inhibitory);
    check(&tree, 0, ElementType::Dendrite, SignalType::Excitatory);
    check(&tree, 0, ElementType::Dendrite, SignalType::Inhibitory);

    check(&tree, 1, ElementType::Axon, SignalType::Excitatory);
    check(&tree, 1, ElementType::Axon, SignalType::Inhibitory);
    check(&tree, 1, ElementType::Dendrite, SignalType::Excitatory);
    check(&tree, 1, ElementType::Dendrite, SignalType::Inhibitory);

    check(&tree, 4, ElementType::Axon, SignalType::Excitatory);
    check(&tree, 4, ElementType::Axon, SignalType::Inhibitory);
    check(&tree, 4, ElementType::Dendrite, SignalType::Excitatory);
    check(&tree, 4, ElementType::Dendrite, SignalType::Inhibitory);

    check(&tree, std::numeric_limits<RelearnTypes::level_type>::max(), ElementType::Axon, SignalType::Excitatory);
    check(&tree, std::numeric_limits<RelearnTypes::level_type>::max(), ElementType::Axon, SignalType::Inhibitory);
    check(&tree, std::numeric_limits<RelearnTypes::level_type>::max(), ElementType::Dendrite, SignalType::Excitatory);
    check(&tree, std::numeric_limits<RelearnTypes::level_type>::max(), ElementType::Dendrite, SignalType::Inhibitory);
}

TEST_F(FMMTest, testFindTargetForLocalRootException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);
    const auto& root_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);
    const auto& local_root_position = SimulationFactory::get_random_position_in_box(min, max, this->mt);

    auto root = OctreeNode<FastMultipoleMethodCell>{};
    root.set_level(0);
    root.set_rank(my_rank);
    root.set_cell_size(min, max);
    root.set_cell_neuron_id(NeuronID::virtual_id());
    root.set_cell_neuron_position(root_position);
    root.set_cell_number_axons(1, 1);
    root.set_cell_number_dendrites(2, 2);

    auto local_root = OctreeNode<FastMultipoleMethodCell>{};
    local_root.set_level(0);
    local_root.set_rank(my_rank);
    local_root.set_cell_size(min, max);
    local_root.set_cell_neuron_id(NeuronID::virtual_id());
    local_root.set_cell_neuron_position(local_root_position);
    local_root.set_cell_number_axons(3, 3);
    local_root.set_cell_number_dendrites(4, 4);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, nullptr, nullptr, 0, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, nullptr, nullptr, 0, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, nullptr, nullptr, 0, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, nullptr, nullptr, 0, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, nullptr, 0, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, nullptr, 0, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, nullptr, 0, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, nullptr, 0, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, nullptr, &local_root, 0, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, nullptr, &local_root, 0, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, nullptr, &local_root, 0, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, nullptr, &local_root, 0, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 0, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 0, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 0, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 0, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 1, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 1, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 1, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 1, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 2, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 2, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 2, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 2, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    local_root.set_level(2);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 1, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 1, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 1, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 1, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 3, ElementType::Axon, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 3, ElementType::Axon, SignalType::Inhibitory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 3, ElementType::Dendrite, SignalType::Excitatory), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &root, &local_root, 3, ElementType::Dendrite, SignalType::Inhibitory), RelearnException);
}

TEST_F(FMMTest, testFindTargetForLocalRootNoElements) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto no_elements_tree = OctreeFactory::get_tree_no_axons<FastMultipoleMethodCell>(300, memory_holder, min, max, this->mt);
    const auto no_elements_branch_nodes = OctreeAdapter::extract_branch_nodes(&no_elements_tree, 2);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    for (auto* branch_node : no_elements_branch_nodes) {
        auto target_1 = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &no_elements_tree, branch_node, 2, ElementType::Axon, SignalType::Excitatory);
        ASSERT_FALSE(target_1.has_value());

        auto target_2 = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &no_elements_tree, branch_node, 2, ElementType::Axon, SignalType::Inhibitory);
        ASSERT_FALSE(target_2.has_value());

        auto target_3 = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &no_elements_tree, branch_node, 2, ElementType::Dendrite, SignalType::Excitatory);
        ASSERT_FALSE(target_3.has_value());

        auto target_4 = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &no_elements_tree, branch_node, 2, ElementType::Dendrite, SignalType::Inhibitory);
        ASSERT_FALSE(target_4.has_value());
    }

    auto no_axons_tree = OctreeFactory::get_tree_no_axons<FastMultipoleMethodCell>(350, memory_holder, min, max, this->mt);
    const auto no_axons_branch_nodes = OctreeAdapter::extract_branch_nodes(&no_axons_tree, 2);

    for (auto* branch_node : no_axons_branch_nodes) {
        auto target_1 = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &no_axons_tree, branch_node, 2, ElementType::Axon, SignalType::Excitatory);
        ASSERT_FALSE(target_1.has_value());

        auto target_2 = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &no_axons_tree, branch_node, 2, ElementType::Axon, SignalType::Inhibitory);
        ASSERT_FALSE(target_2.has_value());
    }

    auto no_dendrites_tree = OctreeFactory::get_tree_no_dendrites<FastMultipoleMethodCell>(400, memory_holder, min, max, this->mt);
    const auto no_dendrites_branch_nodes = OctreeAdapter::extract_branch_nodes(&no_dendrites_tree, 2);

    for (auto* branch_node : no_dendrites_branch_nodes) {
        auto target_1 = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &no_dendrites_tree, branch_node, 2, ElementType::Dendrite, SignalType::Excitatory);
        ASSERT_FALSE(target_1.has_value());

        auto target_2 = FastMultipoleMethodBase::find_target_for_local_root(sigma, node_cache, &no_dendrites_tree, branch_node, 2, ElementType::Dendrite, SignalType::Inhibitory);
        ASSERT_FALSE(target_2.has_value());
    }
}

TEST_F(FMMTest, testFindTargetForLocalRoot) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_small_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(1000, memory_holder, min, max, this->mt);

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    for (auto branch_level = RelearnTypes::level_type{ 0 }; branch_level < 3; branch_level++) {
        const auto branch_nodes = OctreeAdapter::extract_branch_nodes(&tree, branch_level);

        auto node_cache = NodeCache<FastMultipoleMethodCell>{};
        node_cache.set_is_already_downloaded();

        auto check = [&tree, branch_nodes, branch_level, &node_cache, sigma](auto element_type, auto signal_type) {
            for (auto* branch_node : branch_nodes) {
                auto opt_target = FastMultipoleMethodBase::find_target_for_local_root(utility::cast<RelearnTypes::attraction_type>(sigma), node_cache, &tree, branch_node, branch_level, element_type, signal_type);
                ASSERT_TRUE(opt_target.has_value());

                const auto& [source, target] = opt_target.value();

                ASSERT_EQ(source, branch_node);
                ASSERT_NE(target, nullptr);

                ASSERT_EQ(target->get_level(), branch_level);
                ASSERT_GE(target->get_cell().get_number_elements_for(element_type, signal_type), 0);
            }
        };

        check(ElementType::Axon, SignalType::Excitatory);
        check(ElementType::Axon, SignalType::Inhibitory);
        check(ElementType::Dendrite, SignalType::Excitatory);
        check(ElementType::Dendrite, SignalType::Inhibitory);
    }
}

TEST_F(FMMTest, testFindPartnersException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_small_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(1000, memory_holder, min, max, this->mt);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { nullptr, nullptr }, ElementType::Axon, SignalType::Excitatory, 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { nullptr, nullptr }, ElementType::Axon, SignalType::Inhibitory, 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { nullptr, nullptr }, ElementType::Dendrite, SignalType::Excitatory, 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { nullptr, nullptr }, ElementType::Dendrite, SignalType::Inhibitory, 1), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { &tree, nullptr }, ElementType::Axon, SignalType::Excitatory, 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { &tree, nullptr }, ElementType::Axon, SignalType::Inhibitory, 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { &tree, nullptr }, ElementType::Dendrite, SignalType::Excitatory, 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { &tree, nullptr }, ElementType::Dendrite, SignalType::Inhibitory, 1), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { nullptr, &tree }, ElementType::Axon, SignalType::Excitatory, 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { nullptr, &tree }, ElementType::Axon, SignalType::Inhibitory, 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { nullptr, &tree }, ElementType::Dendrite, SignalType::Excitatory, 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { nullptr, &tree }, ElementType::Dendrite, SignalType::Inhibitory, 1), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { &tree, &tree }, ElementType::Axon, SignalType::Excitatory, 0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { &tree, &tree }, ElementType::Axon, SignalType::Inhibitory, 0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { &tree, &tree }, ElementType::Dendrite, SignalType::Excitatory, 0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FastMultipoleMethodBase::find_partners(sigma, node_cache, { &tree, &tree }, ElementType::Dendrite, SignalType::Inhibitory, 0), RelearnException);
}

TEST_F(FMMTest, testFindPartners) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    const auto check_valid = [&node_cache, sigma](OctreeNode<FastMultipoleMethodCell>* source, OctreeNode<FastMultipoleMethodCell>* target,
                                                  ElementType element_type, SignalType signal_type, RelearnTypes::level_type branch_level) {
        auto new_sources = OctreeAdapter::extract_branch_nodes(source, branch_level);

        const auto pairs = FastMultipoleMethodBase::find_partners(utility::cast<RelearnTypes::attraction_type>(sigma), node_cache, { .current_source = source, .current_target = target }, element_type, signal_type, branch_level);
        auto found_sources = std::vector<OctreeNode<FastMultipoleMethodCell>*>{};

        for (const auto& [s, t] : pairs) {
            ASSERT_NE(s, nullptr);
            ASSERT_NE(t, nullptr);

            found_sources.emplace_back(s);

            ASSERT_GE(s->get_cell().get_number_elements_for(element_type, signal_type), 0);
            ASSERT_GE(t->get_cell().get_number_elements_for(get_other_element_type(element_type), signal_type), 0);

            if (t->is_parent()) {
                ASSERT_EQ(t->get_level(), branch_level + target->get_level());
            }
        }

        std::ranges::sort(new_sources);
        std::ranges::sort(found_sources);

        ASSERT_EQ(new_sources, found_sources);
    };

    const auto check_none = [&node_cache, sigma](OctreeNode<FastMultipoleMethodCell>* source, OctreeNode<FastMultipoleMethodCell>* target,
                                                 ElementType element_type, SignalType signal_type, RelearnTypes::level_type branch_level) {
        const auto pairs = FastMultipoleMethodBase::find_partners(sigma, node_cache, { .current_source = source, .current_target = target }, element_type, signal_type, branch_level);

        ASSERT_TRUE(pairs.empty());
    };

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto source_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(1000, memory_holder, min, max, this->mt);
    auto target_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(1000, memory_holder, min, max, this->mt);

    check_valid(&source_tree, &target_tree, ElementType::Axon, SignalType::Excitatory, 1);
    check_valid(&source_tree, &target_tree, ElementType::Axon, SignalType::Inhibitory, 1);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Excitatory, 1);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Inhibitory, 1);

    check_valid(&source_tree, &target_tree, ElementType::Axon, SignalType::Excitatory, 2);
    check_valid(&source_tree, &target_tree, ElementType::Axon, SignalType::Inhibitory, 2);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Excitatory, 2);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Inhibitory, 2);

    OctreeAdapter::invalidate_elements(&source_tree, ElementType::Axon, SignalType::Excitatory);

    check_none(&source_tree, &target_tree, ElementType::Axon, SignalType::Excitatory, 1);
    check_valid(&source_tree, &target_tree, ElementType::Axon, SignalType::Inhibitory, 1);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Excitatory, 1);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Inhibitory, 1);

    check_none(&source_tree, &target_tree, ElementType::Axon, SignalType::Excitatory, 2);
    check_valid(&source_tree, &target_tree, ElementType::Axon, SignalType::Inhibitory, 2);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Excitatory, 2);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Inhibitory, 2);

    OctreeAdapter::invalidate_elements(&target_tree, ElementType::Axon, SignalType::Inhibitory);

    check_none(&source_tree, &target_tree, ElementType::Axon, SignalType::Excitatory, 1);
    check_valid(&source_tree, &target_tree, ElementType::Axon, SignalType::Inhibitory, 1);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Excitatory, 1);
    check_none(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Inhibitory, 1);

    check_none(&source_tree, &target_tree, ElementType::Axon, SignalType::Excitatory, 2);
    check_valid(&source_tree, &target_tree, ElementType::Axon, SignalType::Inhibitory, 2);
    check_valid(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Excitatory, 2);
    check_none(&source_tree, &target_tree, ElementType::Dendrite, SignalType::Inhibitory, 2);
}

TEST_F(FMMTest, testFindPartnersUneven) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& [min, max] = SimulationFactory::get_random_simulation_box_size(this->mt);

    auto memory_holder = MemoryHolderFactory::get_default_memory_holder<FastMultipoleMethodCell>();

    auto small_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(2, memory_holder, min, max, this->mt);
    auto large_tree = OctreeFactory::get_standard_tree<FastMultipoleMethodCell>(1000, memory_holder, min, max, this->mt);

    const auto small_branch_nodes = OctreeAdapter::extract_branch_nodes(&small_tree, 2);

    auto node_cache = NodeCache<FastMultipoleMethodCell>{};
    node_cache.set_is_already_downloaded();

    const auto sigma = RelearnTypes::attraction_type{ 150 };

    const auto found_targets_1 = FastMultipoleMethodBase::find_partners(sigma, node_cache, { .current_source = &small_tree, .current_target = &large_tree }, ElementType::Axon, SignalType::Excitatory, 2);
    if (!small_branch_nodes.empty()) {
        ASSERT_EQ(found_targets_1.size(), small_branch_nodes.size());
    } else {
        ASSERT_EQ(found_targets_1.size(), 2);
    }

    const auto large_branch_nodes = OctreeAdapter::extract_branch_nodes(&large_tree, 2);

    const auto found_targets_2 = FastMultipoleMethodBase::find_partners(sigma, node_cache, { .current_source = &large_tree, .current_target = &small_tree }, ElementType::Dendrite, SignalType::Inhibitory, 2);
    ASSERT_EQ(found_targets_2.size(), large_branch_nodes.size());
}
#endif

// End-to-end regression test of the FastMultipoleMethod class itself (as opposed to just the
// free FastMultipoleMethodBase functions above): exercises find_target_neurons() /
// process_requests() / process_responses() (FastMultipoleMethod.cpp) via the public
// update_connectivity_host_mpi() pipeline, with two neurons positioned so that each one's single
// vacant axon can only ever target the other's single vacant dendrite (autapses are forbidden).
//
// Unlike BarnesHutCUDA's acceptance criterion (a leaf with a vacant dendrite is *always*
// accepted), FastMultipoleMethodBase::find_target_for_local_root/find_partners pick probabilistically
// among the octree nodes unpacked at each level (via ProbabilityPicker::pick_target on Gaussian
// attractivenesses), and an intermediate pick can lead to a dead end with zero elements -- so a
// single update_connectivity_host_mpi() call is not guaranteed to connect every vacant axon, just
// like a single simulation step isn't in production. Retry (each time refreshing the octree with
// the current vacant-element counts, exactly as repeated simulation steps would) until both
// neurons have connected or a generous bound is exhausted.
TEST_F(FMMTest, testUpdateConnectivityHostMpiConnectsTwoNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 2 };
    const auto box = RelearnTypes::bounding_box_type{ RelearnTypes::position_type{ 0.0, 0.0, 0.0 }, RelearnTypes::position_type{ 1.0, 1.0, 1.0 } };

    const auto neuron_positions = std::vector<RelearnTypes::position_type>{
        RelearnTypes::position_type{ 0.1F, 0.1F, 0.1F },
        RelearnTypes::position_type{ 0.9F, 0.9F, 0.9F },
    };

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    const auto signal_types = std::vector<SignalType>(number_neurons, SignalType::Excitatory);
    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(extra_infos, signal_types, 1.0, 1.0);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto fmm = FastMultipoleMethod(box, std::make_shared<Morton>(0));
    fmm.set_probability_kernel(KernelFactory::get_standard_gaussian());
    fmm.set_synaptic_elements(synaptic_elements);
    fmm.set_network_graph(network_graph);
    fmm.set_neuron_extra_infos(extra_infos);
    fmm.init(number_neurons);

    auto total_connected_axons = RelearnTypes::counter_type{ 0 };
    for (auto attempt = 0; attempt < 100 && total_connected_axons < 2; ++attempt) {
        const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
        const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
        const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

        ASSERT_NO_THROW(fmm.prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));
        ASSERT_NO_THROW(std::ignore = fmm.update_connectivity(number_neurons));

        const auto& number_connected_axons = synaptic_elements->get_connected_elements(SynapticElementType::Axon);
        total_connected_axons = std::accumulate(number_connected_axons.begin(), number_connected_axons.end(), RelearnTypes::counter_type{ 0 });
    }

    ASSERT_EQ(total_connected_axons, 2) << "Each of the two neurons has exactly one vacant axon and must eventually connect it to the other neuron's dendrite";
}

// Same idea as testUpdateConnectivityHostMpiConnectsTwoNeurons above, but for
// FastMultipoleMethodInverted (FastMultipoleMethodInverted.cpp), which connects in the opposite
// direction (dendrites search for axon partners, via BackwardConnector). Unlike
// FastMultipoleMethod, BackwardAlgorithm::update_connectivity() unconditionally throws
// CPU_NOT_SUPPORTED when RELEARN_CUDA_ENABLED is set and has no public "host_mpi" escape hatch, so
// this test instead drives find_target_neurons()/process_requests()/process_responses() directly
// -- accessible here only because FastMultipoleMethodInverted declares `friend class FMMTest;` --
// replicating BackwardAlgorithm::update_connectivity()'s body without going through the
// CUDA-build-only guard.
TEST_F(FMMTest, testDrivingConnectivityStepsDirectlyConnectsTwoNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 2 };
    const auto box = RelearnTypes::bounding_box_type{ RelearnTypes::position_type{ 0.0, 0.0, 0.0 }, RelearnTypes::position_type{ 1.0, 1.0, 1.0 } };

    const auto neuron_positions = std::vector<RelearnTypes::position_type>{
        RelearnTypes::position_type{ 0.1F, 0.1F, 0.1F },
        RelearnTypes::position_type{ 0.9F, 0.9F, 0.9F },
    };

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    const auto signal_types = std::vector<SignalType>(number_neurons, SignalType::Excitatory);
    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(extra_infos, signal_types, 1.0, 1.0);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto fmm_inverted = FastMultipoleMethodInverted(box, std::make_shared<Morton>(0));
    fmm_inverted.set_probability_kernel(KernelFactory::get_standard_gaussian());
    fmm_inverted.set_synaptic_elements(synaptic_elements);
    fmm_inverted.set_network_graph(network_graph);
    fmm_inverted.set_neuron_extra_infos(extra_infos);
    fmm_inverted.init(number_neurons);

    auto total_connected_dendrites = RelearnTypes::counter_type{ 0 };
    for (auto attempt = 0; attempt < 100 && total_connected_dendrites < 2; ++attempt) {
        const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
        const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
        const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

        ASSERT_NO_THROW(fmm_inverted.prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));

        const auto& outgoing_requests = call_find_target_neurons(fmm_inverted, number_neurons);
        const auto& incoming_requests = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(outgoing_requests);

        auto [responses_outgoing, number_created_synapses, synapses] = call_process_requests(fmm_inverted, incoming_requests);
        std::ignore = number_created_synapses;
        std::ignore = synapses;

        const auto& responses_incoming = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(responses_outgoing);
        std::ignore = call_process_responses(fmm_inverted, outgoing_requests, responses_incoming);

        const auto& number_connected_dendrites = synaptic_elements->get_connected_elements(SynapticElementType::DendriteExcitatory);
        total_connected_dendrites = std::accumulate(number_connected_dendrites.begin(), number_connected_dendrites.end(), RelearnTypes::counter_type{ 0 });
    }

    ASSERT_EQ(total_connected_dendrites, 2) << "Each of the two neurons has exactly one vacant dendrite and must eventually connect it to the other neuron's axon";
}