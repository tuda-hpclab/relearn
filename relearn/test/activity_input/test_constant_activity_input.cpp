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

#include "neurons/input/ConstantActivityInput.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <tuple>

#include "test_activity_input.h"

TEST_F(ConstantActivityInputTest, testConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = RandomFactory::get_random_double<double>(ConstantActivityInput::min_constant_activity, ConstantActivityInput::max_constant_activity, this->mt);

    ASSERT_NO_THROW(std::ignore = ConstantActivityInput(0.0));
    ASSERT_NO_THROW(std::ignore = ConstantActivityInput(1.0));
    ASSERT_NO_THROW(std::ignore = ConstantActivityInput(constant_input));
}

TEST_F(ConstantActivityInputTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = RandomFactory::get_random_double<double>(ConstantActivityInput::min_constant_activity, ConstantActivityInput::max_constant_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_1 = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_2 = NeuronIdFactory::get_random_number_neurons(this->mt);

    auto constant_activity_input = ConstantActivityInput(constant_input);

    ASSERT_THROW_NO_PRINT(constant_activity_input.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.init(0), RelearnException);

    ASSERT_EQ(constant_activity_input.get_number_neurons(), 0);
    ASSERT_EQ(constant_activity_input.get_input().size(), 0);

    constant_activity_input.init(number_neurons_init);

    ASSERT_EQ(constant_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(constant_activity_input.get_input().size(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(constant_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.init(0), RelearnException);

    ASSERT_EQ(constant_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(constant_activity_input.get_input().size(), number_neurons_init);

    constant_activity_input.create_neurons(number_neurons_create_1);

    ASSERT_EQ(constant_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(constant_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(constant_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.init(0), RelearnException);

    ASSERT_EQ(constant_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(constant_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    constant_activity_input.create_neurons(number_neurons_create_2);

    ASSERT_EQ(constant_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(constant_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(constant_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(constant_activity_input.init(0), RelearnException);

    ASSERT_EQ(constant_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(constant_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(ConstantActivityInputTest, testUpdateInput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = RandomFactory::get_random_double<double>(ConstantActivityInput::min_constant_activity, ConstantActivityInput::max_constant_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto constant_activity_input = ConstantActivityInput(constant_input);
    constant_activity_input.init(number_neurons_init);
    constant_activity_input.set_extra_infos(neurons_extra_info);

    const auto actual_input = constant_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        ASSERT_EQ(constant_activity_input.get_input(neuron_id), 0.0);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
    }

    constant_activity_input.update_input(102);
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        ASSERT_EQ(constant_activity_input.get_input(neuron_id), constant_input);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], constant_input);
    }
}

TEST_F(ConstantActivityInputTest, testUpdateInputRange) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = RandomFactory::get_random_double<double>(ConstantActivityInput::min_constant_activity, ConstantActivityInput::max_constant_activity, this->mt);

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto constant_activity_input = ConstantActivityInput(constant_input);
    constant_activity_input.init(number_neurons_init);
    constant_activity_input.set_extra_infos(neurons_extra_info);

    const auto actual_input = constant_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        ASSERT_EQ(constant_activity_input.get_input(neuron_id), 0.0);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
    }

    constant_activity_input.update_input_range(105, first_neuron_id, last_neuron_id);
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        if (first_neuron_id <= neuron_id && neuron_id < last_neuron_id) {
            ASSERT_EQ(constant_activity_input.get_input(neuron_id), constant_input);
            ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], constant_input);
        } else {
            ASSERT_EQ(constant_activity_input.get_input(neuron_id), 0.0);
            ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
        }
    }
}

TEST_F(ConstantActivityInputTest, testUpdateInputMultipleRanges) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = RandomFactory::get_random_double<double>(ConstantActivityInput::min_constant_activity, ConstantActivityInput::max_constant_activity, this->mt);

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto constant_activity_input = ConstantActivityInput(constant_input);
    constant_activity_input.init(number_neurons_init);
    constant_activity_input.set_extra_infos(neurons_extra_info);

    const auto actual_input = constant_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        ASSERT_EQ(constant_activity_input.get_input(neuron_id), 0.0);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
    }

    constant_activity_input.update_input_range(105, first_neuron_id, last_neuron_id);
    constant_activity_input.update_input_range(105, NeuronID{ 0 }, first_neuron_id);
    constant_activity_input.update_input_range(105, last_neuron_id, NeuronID{ number_neurons_init });

    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        ASSERT_EQ(constant_activity_input.get_input(neuron_id), constant_input);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], constant_input);
    }
}

TEST_F(ConstantActivityInputTest, testUpdateInputAfterCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = RandomFactory::get_random_double<double>(ConstantActivityInput::min_constant_activity, ConstantActivityInput::max_constant_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto positions = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(this->mt, number_neurons_init);
    neurons_extra_info->set_positions(positions);

    auto constant_activity_input = ConstantActivityInput(constant_input);
    constant_activity_input.init(number_neurons_init);
    constant_activity_input.set_extra_infos(neurons_extra_info);

    constant_activity_input.update_input(102);
    constant_activity_input.create_neurons(number_neurons_create);
    neurons_extra_info->create_neurons(number_neurons_create);
    constant_activity_input.update_input(103);

    const auto actual_input = constant_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init + number_neurons_create)) {
        ASSERT_EQ(constant_activity_input.get_input(neuron_id), constant_input);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], constant_input);
    }
}

TEST_F(ConstantActivityInputTest, testFootprintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = RandomFactory::get_random_double<double>(ConstantActivityInput::min_constant_activity, ConstantActivityInput::max_constant_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto constant_activity_input = ConstantActivityInput(constant_input);
    constant_activity_input.init(number_neurons_init);
    constant_activity_input.set_extra_infos(neurons_extra_info);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(constant_activity_input.record_memory_footprint(footprint));
}

TEST_F(ConstantActivityInputTest, testSetExtraInfoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = RandomFactory::get_random_double<double>(ConstantActivityInput::min_constant_activity, ConstantActivityInput::max_constant_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info_too_large = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_large->init(number_neurons_init + 2);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init + 1);

    const auto neurons_extra_info_too_small = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_small->init(number_neurons_init + 0);

    auto constant_activity_input = ConstantActivityInput(constant_input);
    constant_activity_input.init(number_neurons_init + 1);

    ASSERT_THROW_NO_PRINT(constant_activity_input.set_extra_infos({}), RelearnException);

    constant_activity_input.set_extra_infos(neurons_extra_info_too_large);
    ASSERT_THROW_NO_PRINT(constant_activity_input.update_input(102), RelearnException);

    constant_activity_input.set_extra_infos(neurons_extra_info_too_small);
    ASSERT_THROW_NO_PRINT(constant_activity_input.update_input(102), RelearnException);

    constant_activity_input.set_extra_infos(neurons_extra_info);
    ASSERT_NO_THROW(constant_activity_input.update_input(102));
}
