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

#include "neurons/input/ScaleActivityInput.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/activity_input/activity_input_factory.h"
#include "factory/extra_info/extra_info_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <tuple>

#include "test_activity_input.h"

TEST_F(ScaleActivityInputTest, testConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scaling_function = [](const double d) -> double { return (d * 2.3) + 1.412; };

    const auto other = ActivityInputFactory::construct_normal_activity(2.3, 1.7);

    ASSERT_NO_THROW(std::ignore = ScaleActivityInput(other, scaling_function));
}

TEST_F(ScaleActivityInputTest, testConstructorThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scaling_function = [](const double d) -> double { return (d * 2.3) + 1.412; };

    const auto other = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    auto empty_ptr = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    empty_ptr.reset();

    ASSERT_THROW_NO_PRINT(std::ignore = ScaleActivityInput(empty_ptr, scaling_function), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = ScaleActivityInput(other, {}), RelearnException);
}

TEST_F(ScaleActivityInputTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scaling_function = [](const double d) -> double { return (d * 2.3) + 1.412; };

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_1 = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_2 = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto other = ActivityInputFactory::construct_normal_activity(2.3, 1.7);

    auto scale_activity_input = ScaleActivityInput(other, scaling_function);

    ASSERT_THROW_NO_PRINT(scale_activity_input.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.init(0), RelearnException);

    ASSERT_EQ(scale_activity_input.get_number_neurons(), 0);
    ASSERT_EQ(scale_activity_input.get_input().size(), 0);

    scale_activity_input.init(number_neurons_init);

    ASSERT_EQ(scale_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(scale_activity_input.get_input().size(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(scale_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.init(0), RelearnException);

    ASSERT_EQ(scale_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(scale_activity_input.get_input().size(), number_neurons_init);

    scale_activity_input.create_neurons(number_neurons_create_1);

    ASSERT_EQ(scale_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(scale_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(scale_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.init(0), RelearnException);

    ASSERT_EQ(scale_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(scale_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    scale_activity_input.create_neurons(number_neurons_create_2);

    ASSERT_EQ(scale_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(scale_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(scale_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(scale_activity_input.init(0), RelearnException);

    ASSERT_EQ(scale_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(scale_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(ScaleActivityInputTest, testUpdateInput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scaling_function = [](const double d) -> double { return (d * 2.3) + 1.412; };

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto other = ActivityInputFactory::construct_normal_activity(2.3, 1.7);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto scale_activity_input = ScaleActivityInput(other, scaling_function);
    scale_activity_input.init(number_neurons_init);
    scale_activity_input.set_extra_infos(neurons_extra_info);

    scale_activity_input.update_input(102);

    const auto actual_input = scale_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_input = scaling_function(other->get_input(neuron_id));

        ASSERT_NEAR(scale_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(ScaleActivityInputTest, testUpdateInputRange) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scaling_function = [](const double d) -> double { return (d * 2.3) + 1.412; };

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto other = ActivityInputFactory::construct_normal_activity(2.3, 1.7);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto scale_activity_input = ScaleActivityInput(other, scaling_function);
    scale_activity_input.init(number_neurons_init);
    scale_activity_input.set_extra_infos(neurons_extra_info);

    scale_activity_input.update_input_range(102, first_neuron_id, last_neuron_id);

    const auto actual_input = scale_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_input = scaling_function(other->get_input(neuron_id));

        if (first_neuron_id <= neuron_id && neuron_id < last_neuron_id) {
            ASSERT_NEAR(scale_activity_input.get_input(neuron_id), expected_input, eps);
            ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
        } else {
            ASSERT_EQ(scale_activity_input.get_input(neuron_id), 0.0);
            ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
        }
    }
}

TEST_F(ScaleActivityInputTest, testUpdateInputMultipleRanges) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scaling_function = [](const double d) -> double { return (d * 2.3) + 1.412; };

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto other = ActivityInputFactory::construct_normal_activity(2.3, 1.7);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto scale_activity_input = ScaleActivityInput(other, scaling_function);
    scale_activity_input.init(number_neurons_init);
    scale_activity_input.set_extra_infos(neurons_extra_info);

    scale_activity_input.update_input_range(105, first_neuron_id, last_neuron_id);
    scale_activity_input.update_input_range(105, NeuronID{ 0 }, first_neuron_id);
    scale_activity_input.update_input_range(105, last_neuron_id, NeuronID{ number_neurons_init });
    const auto actual_input = scale_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_input = scaling_function(other->get_input(neuron_id));

        ASSERT_NEAR(scale_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(ScaleActivityInputTest, testUpdateInputAfterCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scaling_function = [](const double d) -> double { return (d * 2.3) + 1.412; };

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto other = ActivityInputFactory::construct_normal_activity(2.3, 1.7);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto positions = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(this->mt, number_neurons_init);
    neurons_extra_info->set_positions(positions);

    auto scale_activity_input = ScaleActivityInput(other, scaling_function);
    scale_activity_input.init(number_neurons_init);
    scale_activity_input.set_extra_infos(neurons_extra_info);

    scale_activity_input.update_input(102);
    scale_activity_input.create_neurons(number_neurons_create);
    neurons_extra_info->create_neurons(number_neurons_create);
    scale_activity_input.update_input(103);

    const auto actual_input = scale_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_input = scaling_function(other->get_input(neuron_id));

        ASSERT_NEAR(scale_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(ScaleActivityInputTest, testFootprintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scaling_function = [](const double d) -> double { return (d * 2.3) + 1.412; };
    const auto other = ActivityInputFactory::construct_normal_activity(2.3, 1.7);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto scale_activity_input = ScaleActivityInput(other, scaling_function);
    scale_activity_input.init(number_neurons_init);
    scale_activity_input.set_extra_infos(neurons_extra_info);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(scale_activity_input.record_memory_footprint(footprint));
}

TEST_F(ScaleActivityInputTest, testSetExtraInfoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scaling_function = [](const double d) -> double { return (d * 2.3) + 1.412; };
    const auto other = ActivityInputFactory::construct_normal_activity(2.3, 1.7);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info_too_large = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_large->init(number_neurons_init + 2);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init + 1);

    const auto neurons_extra_info_too_small = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_small->init(number_neurons_init + 0);

    auto scale_activity_input = ScaleActivityInput(other, scaling_function);
    scale_activity_input.init(number_neurons_init + 1);

    ASSERT_THROW_NO_PRINT(scale_activity_input.set_extra_infos({}), RelearnException);

    scale_activity_input.set_extra_infos(neurons_extra_info_too_large);
    ASSERT_THROW_NO_PRINT(scale_activity_input.update_input(102), RelearnException);

    scale_activity_input.set_extra_infos(neurons_extra_info_too_small);
    ASSERT_THROW_NO_PRINT(scale_activity_input.update_input(102), RelearnException);

    scale_activity_input.set_extra_infos(neurons_extra_info);
    ASSERT_NO_THROW(scale_activity_input.update_input(102));
}
