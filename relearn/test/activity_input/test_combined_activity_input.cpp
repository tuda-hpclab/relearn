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

#include "neurons/input/ActivityInput.h"
#include "neurons/input/CombinedActivityInput.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "factory/activity_input/activity_input_factory.h"
#include "factory/extra_info/extra_info_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

TEST_F(CombinedActivityInputTest, testConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    ASSERT_NO_THROW(std::ignore = CombinedActivityInput(1, {}));
    ASSERT_NO_THROW(std::ignore = CombinedActivityInput(1, { constant_input_1 }));
    ASSERT_NO_THROW(std::ignore = CombinedActivityInput(1, { normal_input_1 }));
    ASSERT_NO_THROW(std::ignore = CombinedActivityInput(1, { constant_input_1, normal_input_1 }));
    ASSERT_NO_THROW(std::ignore = CombinedActivityInput(1, { constant_input_1, normal_input_1, constant_input_2, normal_input_2 }));
}

TEST_F(CombinedActivityInputTest, testConstructorThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    auto empty_ptr = ActivityInputFactory::construct_constant_activity();
    empty_ptr.reset();

    ASSERT_THROW_NO_PRINT(std::ignore = CombinedActivityInput(1, { empty_ptr }), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = CombinedActivityInput(1, { constant_input_1, empty_ptr }), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = CombinedActivityInput(1, { normal_input_1, empty_ptr }), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = CombinedActivityInput(1, { constant_input_1, empty_ptr, normal_input_1 }), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = CombinedActivityInput(1, { constant_input_1, normal_input_1, empty_ptr, constant_input_2, normal_input_2 }), RelearnException);

    const auto combined = std::make_shared<CombinedActivityInput>(1, std::vector<std::shared_ptr<ActivityInput>>{});
    const auto vec = std::vector<std::shared_ptr<ActivityInput>>{ empty_ptr, combined };
    ASSERT_THROW_NO_PRINT(std::ignore = CombinedActivityInput(1, vec), RelearnException);
}

#ifndef RELEARN_CUDA_ENABLED
TEST_F(CombinedActivityInputTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_1 = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_2 = NeuronIdFactory::get_random_number_neurons(this->mt);

    auto combined_activity_input = CombinedActivityInput(1, { normal_input_2, constant_input_1, normal_input_1, constant_input_2 });

    ASSERT_THROW_NO_PRINT(combined_activity_input.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.init(0), RelearnException);

    ASSERT_EQ(combined_activity_input.get_number_neurons(), 0);
    ASSERT_EQ(combined_activity_input.get_input().size(), 0);

    combined_activity_input.init(number_neurons_init);

    ASSERT_EQ(combined_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(combined_activity_input.get_input().size(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(combined_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.init(0), RelearnException);

    ASSERT_EQ(combined_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(combined_activity_input.get_input().size(), number_neurons_init);

    combined_activity_input.create_neurons(number_neurons_create_1);

    ASSERT_EQ(combined_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(combined_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(combined_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.init(0), RelearnException);

    ASSERT_EQ(combined_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(combined_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    combined_activity_input.create_neurons(number_neurons_create_2);

    ASSERT_EQ(combined_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(combined_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(combined_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.init(0), RelearnException);

    ASSERT_EQ(combined_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(combined_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}
#endif

TEST_F(CombinedActivityInputTest, testAddInputSourceThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init + 1);

    auto combined_activity_input = CombinedActivityInput(1, { normal_input_2, constant_input_1, normal_input_1, constant_input_2 });

    auto empty_ptr = ActivityInputFactory::construct_constant_activity();
    empty_ptr.reset();

    ASSERT_THROW_NO_PRINT(combined_activity_input.add_input_source(empty_ptr), RelearnException);

    combined_activity_input.init(number_neurons_init + 1);
    combined_activity_input.set_extra_infos(neurons_extra_info);

    const auto too_small_input = ActivityInputFactory::construct_constant_activity(1.1);
    too_small_input->init(number_neurons_init + 0);

    const auto too_large_input = ActivityInputFactory::construct_constant_activity(1.1);
    too_large_input->init(number_neurons_init + 2);

    ASSERT_THROW_NO_PRINT(combined_activity_input.add_input_source(too_small_input), RelearnException);
    ASSERT_THROW_NO_PRINT(combined_activity_input.add_input_source(too_large_input), RelearnException);

    combined_activity_input.update_input(102);

    const auto actual_input = combined_activity_input.get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        const auto expected_input = constant_input_1->get_input(neuron_id) + constant_input_2->get_input(neuron_id) + normal_input_1->get_input(neuron_id) + normal_input_2->get_input(neuron_id);

        ASSERT_NEAR(combined_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(CombinedActivityInputTest, testUpdateInputAfterAdd) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto other_constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto other_constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto other_normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto other_normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto combined_activity_input = CombinedActivityInput(1, { normal_input_2, constant_input_1, normal_input_1, constant_input_2 });
    combined_activity_input.init(number_neurons_init);
    combined_activity_input.set_extra_infos(neurons_extra_info);

    const auto vec = std::vector<std::shared_ptr<ActivityInput>>{ other_normal_input_2, other_constant_input_1, other_normal_input_1, other_constant_input_2 };
    auto other_combined_activity_input = std::make_shared<CombinedActivityInput>(1, vec);
    other_combined_activity_input->init(number_neurons_init);
    other_combined_activity_input->set_extra_infos(neurons_extra_info);

    combined_activity_input.add_input_source(other_combined_activity_input);

    combined_activity_input.update_input(102);
    const auto actual_input = combined_activity_input.get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        const auto expected_input = constant_input_1->get_input(neuron_id) + constant_input_2->get_input(neuron_id) + normal_input_1->get_input(neuron_id) + normal_input_2->get_input(neuron_id) + other_combined_activity_input->get_input(neuron_id);

        ASSERT_NEAR(combined_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(CombinedActivityInputTest, testUpdateInput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto combined_activity_input = CombinedActivityInput(1, { normal_input_2, constant_input_1, normal_input_1, constant_input_2 });
    combined_activity_input.init(number_neurons_init);
    combined_activity_input.set_extra_infos(neurons_extra_info);

    combined_activity_input.update_input(102);

    const auto actual_input = combined_activity_input.get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        const auto expected_input = constant_input_1->get_input(neuron_id) + constant_input_2->get_input(neuron_id) + normal_input_1->get_input(neuron_id) + normal_input_2->get_input(neuron_id);

        ASSERT_NEAR(combined_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(CombinedActivityInputTest, testUpdateInputRange) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    std::unique_ptr<ActivityInput> combined_activity_input = std::make_unique<CombinedActivityInput>(1, std::vector{ normal_input_2, constant_input_1, normal_input_1, constant_input_2 });
    combined_activity_input->init(number_neurons_init);
    combined_activity_input->set_extra_infos(neurons_extra_info);

    combined_activity_input->update_input_range(102, first_neuron_id, last_neuron_id);

    const auto actual_input = combined_activity_input->get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        const auto expected_input = constant_input_1->get_input(neuron_id) + constant_input_2->get_input(neuron_id) + normal_input_1->get_input(neuron_id) + normal_input_2->get_input(neuron_id);
        if (first_neuron_id <= neuron_id && neuron_id < last_neuron_id) {
            ASSERT_NEAR(combined_activity_input->get_input(neuron_id), expected_input, eps);
            ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
        } else {
            ASSERT_EQ(expected_input, 0.0);
            ASSERT_EQ(combined_activity_input->get_input(neuron_id), 0.0);
            ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
        }
    }
}

TEST_F(CombinedActivityInputTest, testUpdateInputMultipleRanges) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto number_neurons_init = 50;
    const auto first_neuron_id = NeuronID{ 10 };
    const auto last_neuron_id = NeuronID{ 42 };

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    std::unique_ptr<ActivityInput> combined_activity_input = std::make_unique<CombinedActivityInput>(1, std::vector{ normal_input_2, constant_input_1, normal_input_1, constant_input_2 });
    combined_activity_input->init(number_neurons_init);
    combined_activity_input->set_extra_infos(neurons_extra_info);

    combined_activity_input->update_input_range(105, first_neuron_id, last_neuron_id);
    combined_activity_input->update_input_range(105, NeuronID{ 0 }, first_neuron_id);
    combined_activity_input->update_input_range(105, last_neuron_id, NeuronID{ number_neurons_init });

    const auto actual_input = combined_activity_input->get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        const auto expected_input = constant_input_1->get_input(neuron_id) + constant_input_2->get_input(neuron_id) + normal_input_1->get_input(neuron_id) + normal_input_2->get_input(neuron_id);
        ASSERT_NEAR(combined_activity_input->get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

#ifndef RELEARN_CUDA_ENABLED
TEST_F(CombinedActivityInputTest, testUpdateInputAfterCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto positions = SimulationFactory::get_random_positions<std::allocator<RelearnTypes::position_type>>(this->mt, number_neurons_init);
    neurons_extra_info->set_positions(positions);

    auto combined_activity_input = CombinedActivityInput(1, { normal_input_2, constant_input_1, normal_input_1, constant_input_2 });
    combined_activity_input.init(number_neurons_init);
    combined_activity_input.set_extra_infos(neurons_extra_info);

    combined_activity_input.update_input(102);
    combined_activity_input.create_neurons(number_neurons_create);
    neurons_extra_info->create_neurons(number_neurons_create);
    combined_activity_input.update_input(103);

    const auto actual_input = combined_activity_input.get_input();
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        const auto expected_input = constant_input_1->get_input(neuron_id) + constant_input_2->get_input(neuron_id) + normal_input_1->get_input(neuron_id) + normal_input_2->get_input(neuron_id);

        ASSERT_NEAR(combined_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}
#endif

TEST_F(CombinedActivityInputTest, testFootprintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    auto combined_activity_input = CombinedActivityInput(1, { normal_input_2, constant_input_1, normal_input_1, constant_input_2 });
    combined_activity_input.init(number_neurons_init);
    combined_activity_input.set_extra_infos(neurons_extra_info);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(combined_activity_input.record_memory_footprint(footprint));
}

TEST_F(CombinedActivityInputTest, testSetExtraInfoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
    const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
    const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
    const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);

    const auto neurons_extra_info_too_large = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_large->init(number_neurons_init + 2);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init + 1);

    const auto neurons_extra_info_too_small = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_small->init(number_neurons_init + 0);

    auto combined_activity_input = CombinedActivityInput(1, { normal_input_2, constant_input_1, normal_input_1, constant_input_2 });
    combined_activity_input.init(number_neurons_init + 1);

    ASSERT_THROW_NO_PRINT(combined_activity_input.set_extra_infos({}), RelearnException);

    combined_activity_input.set_extra_infos(neurons_extra_info_too_large);
    ASSERT_THROW_NO_PRINT(combined_activity_input.update_input(102), RelearnException);

    combined_activity_input.set_extra_infos(neurons_extra_info_too_small);
    ASSERT_THROW_NO_PRINT(combined_activity_input.update_input(102), RelearnException);

    combined_activity_input.set_extra_infos(neurons_extra_info);
    ASSERT_NO_THROW(combined_activity_input.update_input(102));
}
