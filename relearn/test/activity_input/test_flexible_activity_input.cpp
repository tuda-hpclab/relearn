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
#include "Types.h"

#include "neurons/input/ActivityInput.h"
#include "neurons/input/FlexibleActivityInput.h"
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

#include <cstddef>
#include <iostream>

#include "test_activity_input.h"

class ComplexChoiceFunction : public ChoiceFunction {
public:
    ComplexChoiceFunction(std::function<std::unordered_set<size_t>(RelearnTypes::step_type step, NeuronID neuron_id)>&& _function, size_t _number_neurons)
        : number_neurons(_number_neurons)
        , function(std::move(_function)) {
        range = { { NeuronID{ 0 }, NeuronID{ number_neurons - 1 } } };
    }
    const std::unordered_set<size_t>& get_inputs_for_neuron_id(RelearnTypes::step_type step, NeuronID neuron_id) override {
        const auto inputs = function(step, neuron_id);
        cache.emplace_back(inputs);
        return cache[cache.size() - 1];
    }
    const std::unordered_set<std::pair<NeuronID, NeuronID>, boost::hash<std::pair<NeuronID, NeuronID>>>& get_neuron_id_ranges_for_input(RelearnTypes::step_type /*step*/, size_t /*input_index*/) override {
        return range;
    }
    void create_neurons(RelearnTypes::number_neurons_type /*creation_count*/) override { }

private:
    size_t number_neurons;
    std::function<std::unordered_set<size_t>(RelearnTypes::step_type step, NeuronID neuron_id)> function;
    std::unordered_set<std::pair<NeuronID, NeuronID>, boost::hash<std::pair<NeuronID, NeuronID>>> range;
    std::vector<std::unordered_set<size_t>> cache{};
};

static std::unique_ptr<ChoiceFunction> create_choice_function(RelearnTypes::number_neurons_type number_neurons) noexcept {
    return std::make_unique<ComplexChoiceFunction>([](RelearnTypes::step_type, NeuronID) { return std::unordered_set<size_t>{ 0 }; }, number_neurons);
}

TEST_F(FlexibleActivityInputTest, testConstructorNoThrow) {
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

    ASSERT_NO_THROW(std::ignore = FlexibleActivityInput({}, create_choice_function(10)));
    ASSERT_NO_THROW(std::ignore = FlexibleActivityInput({ constant_input_1 }, create_choice_function(10)));
    ASSERT_NO_THROW(std::ignore = FlexibleActivityInput({ normal_input_1 }, create_choice_function(10)));
    ASSERT_NO_THROW(std::ignore = FlexibleActivityInput({ constant_input_1, normal_input_1 }, create_choice_function(10)));
    ASSERT_NO_THROW(std::ignore = FlexibleActivityInput({ constant_input_1, normal_input_1, constant_input_2, normal_input_2 }, create_choice_function(10)));
}

TEST_F(FlexibleActivityInputTest, testConstructorThrow) {
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

    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput({ empty_ptr }, create_choice_function(10)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput({ constant_input_1, empty_ptr }, create_choice_function(10)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput({ normal_input_1, empty_ptr }, create_choice_function(10)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput({ constant_input_1, empty_ptr, normal_input_1 }, create_choice_function(10)), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput({ constant_input_1, normal_input_1, empty_ptr, constant_input_2, normal_input_2 }, create_choice_function(10)), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput(std::vector<std::shared_ptr<ActivityInput>>{}, {}), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput({ constant_input_1 }, {}), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput({ normal_input_1 }, {}), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput({ constant_input_1, normal_input_1 }, {}), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput({ constant_input_1, normal_input_1, constant_input_2, normal_input_2 }, {}), RelearnException);

    const auto combined = std::make_shared<FlexibleActivityInput>(std::vector<std::shared_ptr<ActivityInput>>{}, create_choice_function(10));
    const auto vec = std::vector<std::shared_ptr<ActivityInput>>{ empty_ptr, combined };
    ASSERT_THROW_NO_PRINT(std::ignore = FlexibleActivityInput(vec, create_choice_function(10)), RelearnException);
}

TEST_F(FlexibleActivityInputTest, testInitAndCreate) {
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

    auto flexible_activity_input = FlexibleActivityInput({ normal_input_2, constant_input_1, normal_input_1, constant_input_2 }, create_choice_function(10));

    ASSERT_THROW_NO_PRINT(flexible_activity_input.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.init(0), RelearnException);

    ASSERT_EQ(flexible_activity_input.get_number_neurons(), 0);
    ASSERT_EQ(flexible_activity_input.get_input().size(), 0);

    flexible_activity_input.init(number_neurons_init);

    ASSERT_EQ(flexible_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(flexible_activity_input.get_input().size(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(flexible_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.init(0), RelearnException);

    ASSERT_EQ(flexible_activity_input.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(flexible_activity_input.get_input().size(), number_neurons_init);

    flexible_activity_input.create_neurons(number_neurons_create_1);

    ASSERT_EQ(flexible_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(flexible_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(flexible_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.init(0), RelearnException);

    ASSERT_EQ(flexible_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(flexible_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1);

    flexible_activity_input.create_neurons(number_neurons_create_2);

    ASSERT_EQ(flexible_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(flexible_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(flexible_activity_input.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.init(0), RelearnException);

    ASSERT_EQ(flexible_activity_input.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(flexible_activity_input.get_input().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(FlexibleActivityInputTest, testAddInputSourceThrow) {
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

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt) + 1;

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    const auto vec = std::vector{ normal_input_2, constant_input_1, normal_input_1, constant_input_2 };

    auto flexible_activity_input = FlexibleActivityInput(vec, create_choice_function(number_neurons_init));

    auto empty_ptr = ActivityInputFactory::construct_constant_activity();
    empty_ptr.reset();

    ASSERT_THROW_NO_PRINT(flexible_activity_input.add_input_source(empty_ptr), RelearnException);

    flexible_activity_input.init(number_neurons_init);
    flexible_activity_input.set_extra_infos(neurons_extra_info);

    const auto too_small_input = ActivityInputFactory::construct_constant_activity(1.1);
    too_small_input->init(number_neurons_init - 1);

    const auto too_large_input = ActivityInputFactory::construct_constant_activity(1.1);
    too_large_input->init(number_neurons_init + 1);

    ASSERT_THROW_NO_PRINT(flexible_activity_input.add_input_source(too_small_input), RelearnException);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.add_input_source(too_large_input), RelearnException);

    flexible_activity_input.update_input(102);

    const auto actual_input = flexible_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto expected_input = normal_input_2->get_input(neuron_id);

        ASSERT_NEAR(flexible_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(FlexibleActivityInputTest, testUpdateInputAfterAdd) {
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

    auto flexible_activity_input = FlexibleActivityInput({ normal_input_2, constant_input_1, normal_input_1, constant_input_2 }, create_choice_function(number_neurons_init));
    flexible_activity_input.init(number_neurons_init);
    flexible_activity_input.set_extra_infos(neurons_extra_info);

    const auto vec = std::vector<std::shared_ptr<ActivityInput>>{ other_normal_input_2, other_constant_input_1, other_normal_input_1, other_constant_input_2 };
    auto other_flexible_activity_input = std::make_shared<FlexibleActivityInput>(vec, create_choice_function(number_neurons_init));
    other_flexible_activity_input->init(number_neurons_init);
    other_flexible_activity_input->set_extra_infos(neurons_extra_info);

    flexible_activity_input.add_input_source(other_flexible_activity_input);

    flexible_activity_input.update_input(102);

    const auto actual_input = flexible_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto input_vector = std::vector{ normal_input_2, constant_input_1, normal_input_1, constant_input_2 };
        const auto expected_input = normal_input_2->get_input(neuron_id);

        ASSERT_NEAR(flexible_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(FlexibleActivityInputTest, testUpdateInput) {
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

    auto flexible_activity_input = FlexibleActivityInput({ normal_input_2, constant_input_1, normal_input_1, constant_input_2 }, create_choice_function(number_neurons_init));
    flexible_activity_input.init(number_neurons_init);
    flexible_activity_input.set_extra_infos(neurons_extra_info);

    flexible_activity_input.update_input(102);

    const auto actual_input = flexible_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto vec = std::vector{ normal_input_2, constant_input_1, normal_input_1, constant_input_2 };
        const auto expected_input = normal_input_2->get_input(neuron_id);

        ASSERT_NEAR(flexible_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(FlexibleActivityInputTest, testUpdateInputRAnge) {
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

    auto flexible_activity_input = FlexibleActivityInput({ normal_input_2, constant_input_1, normal_input_1, constant_input_2 }, create_choice_function(10));
    flexible_activity_input.init(number_neurons_init);
    flexible_activity_input.set_extra_infos(neurons_extra_info);

    flexible_activity_input.update_input_range(102, first_neuron_id, last_neuron_id);

    const auto actual_input = flexible_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto vec = std::vector{ normal_input_2, constant_input_1, normal_input_1, constant_input_2 };
        const auto expected_input = normal_input_2->get_input(neuron_id);

        if (first_neuron_id <= neuron_id && neuron_id < last_neuron_id) {
            ASSERT_NEAR(flexible_activity_input.get_input(neuron_id), expected_input, eps);
            ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
        } else {
            ASSERT_EQ(flexible_activity_input.get_input(neuron_id), 0.0);
            ASSERT_EQ(actual_input[neuron_id.get_neuron_id()], 0.0);
        }
    }
}

TEST_F(FlexibleActivityInputTest, testUpdateInputMultipleRanges) {
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

    auto flexible_activity_input = FlexibleActivityInput({ normal_input_2, constant_input_1, normal_input_1, constant_input_2 }, create_choice_function(10));
    flexible_activity_input.init(number_neurons_init);
    flexible_activity_input.set_extra_infos(neurons_extra_info);

    flexible_activity_input.update_input_range(105, first_neuron_id, last_neuron_id);
    flexible_activity_input.update_input_range(105, NeuronID{ 0 }, first_neuron_id);
    flexible_activity_input.update_input_range(105, last_neuron_id, NeuronID{ number_neurons_init });

    const auto actual_input = flexible_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto vec = std::vector{ normal_input_2, constant_input_1, normal_input_1, constant_input_2 };
        const auto expected_input = normal_input_2->get_input(neuron_id);
        ASSERT_NEAR(flexible_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(FlexibleActivityInputTest, testUpdateInputDifferentChoices) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(this->mt);

    auto test = [number_neurons](std::unique_ptr<ChoiceFunction>&& function) {
        auto* function2 = function.get();
        const auto constant_input_1 = ActivityInputFactory::construct_constant_activity(2.0);
        const auto constant_input_2 = ActivityInputFactory::construct_constant_activity(4.3);
        const auto normal_input_1 = ActivityInputFactory::construct_normal_activity(2.3, 1.7);
        const auto normal_input_2 = ActivityInputFactory::construct_normal_activity(0.0, 2.2);

        const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
        neurons_extra_info->init(number_neurons);

        auto flexible_activity_input = FlexibleActivityInput({ normal_input_2, constant_input_1, normal_input_1, constant_input_2 }, std::move(function));
        flexible_activity_input.init(number_neurons);
        flexible_activity_input.set_extra_infos(neurons_extra_info);

        flexible_activity_input.update_input(102);

        const auto actual_input = flexible_activity_input.get_input();
        for (const auto neuron_id : NeuronID::range(number_neurons)) {
            const auto vec = std::vector{ normal_input_2, constant_input_1, normal_input_1, constant_input_2 };
            const auto expected_input = vec[*function2->get_inputs_for_neuron_id(102, neuron_id).begin()]->get_input(neuron_id);

            ASSERT_NEAR(flexible_activity_input.get_input(neuron_id), expected_input, eps);
            ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
        }
    };

    auto choice_function_1 = std::make_unique<ComplexChoiceFunction>([](RelearnTypes::step_type, NeuronID) { return std::unordered_set<size_t>{ 0 }; }, number_neurons);

    auto choice_function_2 = std::make_unique<ComplexChoiceFunction>([](RelearnTypes::step_type, NeuronID) { return std::unordered_set<size_t>{ 1 }; }, number_neurons);

    auto choice_function_3 = std::make_unique<ComplexChoiceFunction>([](RelearnTypes::step_type, NeuronID) { return std::unordered_set<size_t>{ 2 }; }, number_neurons);

    auto choice_function_4 = std::make_unique<ComplexChoiceFunction>([](RelearnTypes::step_type, NeuronID) { return std::unordered_set<size_t>{ 3 }; }, number_neurons);
    auto choice_function_5 = std::make_unique<ComplexChoiceFunction>([](RelearnTypes::step_type step, NeuronID) { return std::unordered_set<size_t>{ step % 4 }; }, number_neurons);

    test(std::move(choice_function_1));
    test(std::move(choice_function_2));
    test(std::move(choice_function_3));
    test(std::move(choice_function_4));
    test(std::move(choice_function_5));
}

TEST_F(FlexibleActivityInputTest, testUpdateInputAfterCreate) {
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

    auto positions = SimulationFactory::get_random_positions<std::allocator<Vec3d>>(this->mt, number_neurons_init);
    neurons_extra_info->set_positions(positions);

    auto flexible_activity_input = FlexibleActivityInput({ normal_input_2, constant_input_1, normal_input_1, constant_input_2 }, create_choice_function(number_neurons_init));
    flexible_activity_input.init(number_neurons_init);
    flexible_activity_input.set_extra_infos(neurons_extra_info);

    flexible_activity_input.update_input(102);
    flexible_activity_input.create_neurons(number_neurons_create);
    neurons_extra_info->create_neurons(number_neurons_create);
    flexible_activity_input.update_input(103);

    const auto actual_input = flexible_activity_input.get_input();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        const auto vec = std::vector{ normal_input_2, constant_input_1, normal_input_1, constant_input_2 };
        const auto expected_input = normal_input_2->get_input(neuron_id);

        ASSERT_NEAR(flexible_activity_input.get_input(neuron_id), expected_input, eps);
        ASSERT_NEAR(actual_input[neuron_id.get_neuron_id()], expected_input, eps);
    }
}

TEST_F(FlexibleActivityInputTest, testFootprintNoThrow) {
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

    auto flexible_activity_input = FlexibleActivityInput({ normal_input_2, constant_input_1, normal_input_1, constant_input_2 }, create_choice_function(number_neurons_init));
    flexible_activity_input.init(number_neurons_init);
    flexible_activity_input.set_extra_infos(neurons_extra_info);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(flexible_activity_input.record_memory_footprint(footprint));
}

TEST_F(FlexibleActivityInputTest, testSetExtraInfoThrow) {
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

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt) + 1;

    const auto neurons_extra_info_too_large = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_large->init(number_neurons_init + 1);

    const auto neurons_extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info->init(number_neurons_init);

    const auto neurons_extra_info_too_small = NeuronsExtraInfoFactory::construct_extra_info();
    neurons_extra_info_too_small->init(number_neurons_init - 1);

    auto flexible_activity_input = FlexibleActivityInput({ normal_input_2, constant_input_1, normal_input_1, constant_input_2 }, create_choice_function(number_neurons_init));
    flexible_activity_input.init(number_neurons_init);

    ASSERT_THROW_NO_PRINT(flexible_activity_input.set_extra_infos({}), RelearnException);

    flexible_activity_input.set_extra_infos(neurons_extra_info_too_large);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.update_input(102), RelearnException);

    flexible_activity_input.set_extra_infos(neurons_extra_info_too_small);
    ASSERT_THROW_NO_PRINT(flexible_activity_input.update_input(102), RelearnException);

    flexible_activity_input.set_extra_infos(neurons_extra_info);
    ASSERT_NO_THROW(flexible_activity_input.update_input(102));
}
