/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"
#include "test_activity_input.h"

#include "neurons/input/NormalActivityInput.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <iostream>
#include <memory>
#include <tuple>

TEST_F(NormalActivityInputTest, testConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = RandomFactory::get_random_double(NormalActivityInput::min_mean_activity, NormalActivityInput::max_mean_activity, this->mt);
    const auto stddev_input = RandomFactory::get_random_double(NormalActivityInput::min_stddev_activity, NormalActivityInput::max_stddev_activity, this->mt);

    ASSERT_NO_THROW(std::ignore = NormalActivityInput(1, 0.0, 1.0));
    ASSERT_NO_THROW(std::ignore = NormalActivityInput(1, 1.0, 1.0));
    ASSERT_NO_THROW(std::ignore = NormalActivityInput(1, -1.0, 1.0));
    ASSERT_NO_THROW(std::ignore = NormalActivityInput(1, 0.0, static_cast<ActivityInput::activity_type>(stddev_input)));
    ASSERT_NO_THROW(std::ignore = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), 1.0));
    ASSERT_NO_THROW(std::ignore = NormalActivityInput(1, -mean_input, 1.0));
    ASSERT_NO_THROW(std::ignore = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), static_cast<ActivityInput::activity_type>(stddev_input)));
}

TEST_F(NormalActivityInputTest, testConstructorThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = RandomFactory::get_random_double(NormalActivityInput::min_mean_activity, NormalActivityInput::max_mean_activity, this->mt);
    const auto stddev_input = RandomFactory::get_random_double(NormalActivityInput::min_stddev_activity, NormalActivityInput::max_stddev_activity, this->mt);

    ASSERT_THROW_NO_PRINT(std::ignore = NormalActivityInput(1, 0.0, -1.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NormalActivityInput(1, 1.0, -1.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NormalActivityInput(1, -1.0, -1.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NormalActivityInput(1, 0.0, -stddev_input), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NormalActivityInput(1, -mean_input, 0.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), -1.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NormalActivityInput(1, -mean_input, -1.0), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), -stddev_input), RelearnException);
}

TEST_F(NormalActivityInputTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = RandomFactory::get_random_double(NormalActivityInput::min_mean_activity, NormalActivityInput::max_mean_activity, this->mt);
    const auto stddev_input = RandomFactory::get_random_double(NormalActivityInput::min_stddev_activity, NormalActivityInput::max_stddev_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_1 = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_2 = NeuronIdFactory::get_random_number_neurons(this->mt);

    auto normal_activity_input = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), static_cast<ActivityInput::activity_type>(stddev_input));

    ASSERT_THROW_NO_PRINT(normal_activity_input.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.init(0), RelearnException);

    ASSERT_EQ(normal_activity_input.get_number_neurons(), 0);
    ASSERT_EQ(normal_activity_input.get_input().size(), 0);

    normal_activity_input.init(number_neurons_init);

    ASSERT_EQ(normal_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(normal_activity_input.get_input().size(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(normal_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.init(0), RelearnException);

    ASSERT_EQ(normal_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(normal_activity_input.get_input().size(), number_neurons_init);

    normal_activity_input.create_neurons(number_neurons_create_1);

    ASSERT_EQ(normal_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(normal_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(normal_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.init(0), RelearnException);

    ASSERT_EQ(normal_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(normal_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    normal_activity_input.create_neurons(number_neurons_create_2);

    ASSERT_EQ(normal_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(normal_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(normal_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(normal_activity_input.init(0), RelearnException);

    ASSERT_EQ(normal_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(normal_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(NormalActivityInputTest, testUpdateInput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = RandomFactory::get_random_double(NormalActivityInput::min_mean_activity, NormalActivityInput::max_mean_activity, this->mt);
    const auto stddev_input = RandomFactory::get_random_double(NormalActivityInput::min_stddev_activity, NormalActivityInput::max_stddev_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto normal_activity_input = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), static_cast<ActivityInput::activity_type>(stddev_input));
    normal_activity_input.init(number_neurons_init);
    normal_activity_input.set_extra_infos(neurons_extra_info);

    const auto actual_input = normal_activity_input.get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        ASSERT_EQ(normal_activity_input.get_input(neuron_id), 0.0);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
    }

    normal_activity_input.update_input(102);
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        ASSERT_NO_THROW(std::ignore = normal_activity_input.get_input(neuron_id));
        ASSERT_NO_THROW(std::ignore = actual_input[neuron_id.get_neuron_id()]);
    }
}

TEST_F(NormalActivityInputTest, testUpdateInputRange) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = RandomFactory::get_random_double(NormalActivityInput::min_mean_activity, NormalActivityInput::max_mean_activity, this->mt);
    const auto stddev_input = RandomFactory::get_random_double(NormalActivityInput::min_stddev_activity, NormalActivityInput::max_stddev_activity, this->mt);

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    std::unique_ptr<ActivityInput> normal_activity_input = std::make_unique<NormalActivityInput>(1, mean_input, stddev_input);
    normal_activity_input->init(number_neurons_init);
    normal_activity_input->set_extra_infos(neurons_extra_info);

    const auto actual_input = normal_activity_input->get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        ASSERT_EQ(normal_activity_input->get_input(neuron_id), 0.0);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
    }

    normal_activity_input->update_input_range(102, first_neuron_id, last_neuron_id);
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        const auto input = normal_activity_input->get_input(neuron_id);
        const auto input2 = actual_input[neuron_id.get_neuron_id()];
        ASSERT_NEAR(input, input2, eps);
        if (neuron_id < first_neuron_id || neuron_id >= last_neuron_id) {
            ASSERT_EQ(input, 0.0);
            ASSERT_EQ(input2, 0.0);
        }
    }
}

TEST_F(NormalActivityInputTest, testUpdateInputMultipleRanges) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = RandomFactory::get_random_double(NormalActivityInput::min_mean_activity, NormalActivityInput::max_mean_activity, this->mt);
    const auto stddev_input = RandomFactory::get_random_double(NormalActivityInput::min_stddev_activity, NormalActivityInput::max_stddev_activity, this->mt);

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    std::unique_ptr<ActivityInput> normal_activity_input = std::make_unique<NormalActivityInput>(1, mean_input, stddev_input);
    normal_activity_input->init(number_neurons_init);
    normal_activity_input->set_extra_infos(neurons_extra_info);

    const auto actual_input = normal_activity_input->get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        ASSERT_EQ(normal_activity_input->get_input(neuron_id), 0.0);
        ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
    }

    normal_activity_input->update_input_range(105, first_neuron_id, last_neuron_id);
    normal_activity_input->update_input_range(105, NeuronID{ 0 }, first_neuron_id);
    normal_activity_input->update_input_range(105, last_neuron_id, NeuronID{ number_neurons_init });
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        const auto input = normal_activity_input->get_input(neuron_id);
        const auto input2 = actual_input[neuron_id.get_neuron_id()];
        ASSERT_NEAR(input, input2, eps);
    }
}

TEST_F(NormalActivityInputTest, testUpdateInputAfterCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = RandomFactory::get_random_double(NormalActivityInput::min_mean_activity, NormalActivityInput::max_mean_activity, this->mt);
    const auto stddev_input = RandomFactory::get_random_double(NormalActivityInput::min_stddev_activity, NormalActivityInput::max_stddev_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto positions = SimulationFactory::get_random_positions<std::allocator<RelearnTypes::position_type>>(this->mt, number_neurons_init);
    neurons_extra_info->set_positions(positions);

    auto normal_activity_input = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), static_cast<ActivityInput::activity_type>(stddev_input));
    normal_activity_input.init(number_neurons_init);
    normal_activity_input.set_extra_infos(neurons_extra_info);

    normal_activity_input.update_input(102);
    normal_activity_input.create_neurons(number_neurons_create);
    neurons_extra_info->create_neurons(number_neurons_create);
    normal_activity_input.update_input(103);
    const auto actual_input = normal_activity_input.get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init + number_neurons_create)) {
        ASSERT_NO_THROW(std::ignore = normal_activity_input.get_input(neuron_id));
        ASSERT_NO_THROW(std::ignore = actual_input[neuron_id.get_neuron_id()]);
    }
}

TEST_F(NormalActivityInputTest, testFootprintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = RandomFactory::get_random_double(NormalActivityInput::min_mean_activity, NormalActivityInput::max_mean_activity, this->mt);
    const auto stddev_input = RandomFactory::get_random_double(NormalActivityInput::min_stddev_activity, NormalActivityInput::max_stddev_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto normal_activity_input = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), static_cast<ActivityInput::activity_type>(stddev_input));
    normal_activity_input.init(number_neurons_init);
    normal_activity_input.set_extra_infos(neurons_extra_info);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(normal_activity_input.record_memory_footprint(footprint));
}

TEST_F(NormalActivityInputTest, testSetExtraInfoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = RandomFactory::get_random_double(NormalActivityInput::min_mean_activity, NormalActivityInput::max_mean_activity, this->mt);
    const auto stddev_input = RandomFactory::get_random_double(NormalActivityInput::min_stddev_activity, NormalActivityInput::max_stddev_activity, this->mt);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info_too_large = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_large->init(number_neurons_init + 2);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init + 1);

    const auto neurons_extra_info_too_small = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_small->init(number_neurons_init + 0);

    auto normal_activity_input = NormalActivityInput(1, static_cast<ActivityInput::activity_type>(mean_input), static_cast<ActivityInput::activity_type>(stddev_input));
    normal_activity_input.init(number_neurons_init + 1);

    ASSERT_THROW_NO_PRINT(normal_activity_input.set_extra_infos({}), RelearnException);

    normal_activity_input.set_extra_infos(neurons_extra_info_too_large);
    ASSERT_THROW_NO_PRINT(normal_activity_input.update_input(102), RelearnException);

    normal_activity_input.set_extra_infos(neurons_extra_info_too_small);
    ASSERT_THROW_NO_PRINT(normal_activity_input.update_input(102), RelearnException);

    normal_activity_input.set_extra_infos(neurons_extra_info);
    ASSERT_NO_THROW(normal_activity_input.update_input(102));
}
