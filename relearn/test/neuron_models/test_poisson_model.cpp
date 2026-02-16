/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/enums/FiredStatus.h"
#include "neurons/models/poisson/Parameters.h"
#include "neurons/models/poisson/PoissonModel.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/activity_input/activity_input_factory.h"
#include "factory/extra_info/extra_info_factory.h"
#include "factory/fired_status_communicator/fired_status_communicator_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "test_neuron_models.h"

TEST_F(PoissonModelTest, testDefaultParameters) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using param_type = models::poisson::Parameters<double, unsigned int>;

    const auto default_parameters = param_type{};

    ASSERT_EQ(default_parameters.get_x_0(), param_type::default_x_0);
    ASSERT_EQ(default_parameters.get_tau_x(), param_type::default_tau_x);
    ASSERT_EQ(default_parameters.get_refractory_period(), param_type::default_refractory_period);
}

TEST_F(PoissonModelTest, testParameters) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto parameters = models::poisson::Parameters<double, unsigned int>{ 1.0, 2.0, 3U };

    ASSERT_EQ(parameters.get_x_0(), 1.0);
    ASSERT_EQ(parameters.get_tau_x(), 2.0);
    ASSERT_EQ(parameters.get_refractory_period(), 3U);
}

TEST_F(PoissonModelTest, testDefaultConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = models::PoissonModel::default_h;
    const auto parameters = models::poisson::Parameters<double, unsigned int>{};

    ASSERT_NO_THROW(std::ignore = models::PoissonModel(h, activity_input, fired_status_comm, parameters));
}

TEST_F(PoissonModelTest, testGetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = RandomFactory::get_random_integer<unsigned int>(models::PoissonModel::min_h, models::PoissonModel::max_h, this->mt);
    const auto x0 = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_x_0, models::poisson::Parameters<double, unsigned int>::max_x_0, this->mt);
    const auto tau_x = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_tau_x, models::poisson::Parameters<double, unsigned int>::max_tau_x, this->mt);
    const auto refrac = RandomFactory::get_random_integer<unsigned int>(models::poisson::Parameters<double, unsigned int>::min_refractory_time, models::poisson::Parameters<double, unsigned int>::max_refractory_time, this->mt);

    const auto parameters = models::poisson::Parameters<double, unsigned int>{ x0, tau_x, refrac };
    const auto model = models::PoissonModel(h, activity_input, fired_status_comm, parameters);

    ASSERT_EQ(model.get_h(), h);
    ASSERT_EQ(model.get_number_neurons(), 0);
    ASSERT_EQ(model.get_model_parameters().get_tau_x(), tau_x);
    ASSERT_EQ(model.get_model_parameters().get_x_0(), x0);
    ASSERT_EQ(model.get_model_parameters().get_refractory_period(), refrac);
}

TEST_F(PoissonModelTest, testConstructorThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = models::PoissonModel::default_h;
    const auto parameters = models::poisson::Parameters<double, unsigned int>{};

    auto fired_status_comm_empty = fired_status_comm;
    fired_status_comm_empty.reset();

    auto activity_input_empty = activity_input;
    activity_input_empty.reset();

    ASSERT_THROW_NO_PRINT(std::ignore = models::PoissonModel(0, activity_input, fired_status_comm, parameters), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = models::PoissonModel(h, activity_input, fired_status_comm_empty, parameters), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = models::PoissonModel(h, activity_input_empty, fired_status_comm, parameters), RelearnException);
}

TEST_F(PoissonModelTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = RandomFactory::get_random_integer<unsigned int>(models::PoissonModel::min_h, models::PoissonModel::max_h, this->mt);
    const auto x0 = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_x_0, models::poisson::Parameters<double, unsigned int>::max_x_0, this->mt);
    const auto tau_x = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_tau_x, models::poisson::Parameters<double, unsigned int>::max_tau_x, this->mt);
    const auto refrac = RandomFactory::get_random_integer<unsigned int>(models::poisson::Parameters<double, unsigned int>::min_refractory_time, models::poisson::Parameters<double, unsigned int>::max_refractory_time, this->mt);

    const auto parameters = models::poisson::Parameters<double, unsigned int>{ x0, tau_x, refrac };
    auto model = models::PoissonModel(h, activity_input, fired_status_comm, parameters);

    const auto number_neurons_init = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_1 = NeuronIdFactory::get_random_number_neurons(this->mt);
    const auto number_neurons_create_2 = NeuronIdFactory::get_random_number_neurons(this->mt);

    ASSERT_THROW_NO_PRINT(model.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(model.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(model.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(model.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(model.init(0), RelearnException);

    ASSERT_EQ(model.get_number_neurons(), 0);
    ASSERT_EQ(model.get_x().size(), 0);
    ASSERT_EQ(model.get_fired().size(), 0);

    model.init(number_neurons_init);

    ASSERT_EQ(model.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(model.get_input().size(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(model.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(model.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(model.init(0), RelearnException);

    ASSERT_EQ(model.get_number_neurons(), number_neurons_init);
    ASSERT_EQ(model.get_x().size(), number_neurons_init);
    ASSERT_EQ(model.get_fired().size(), number_neurons_init);

    model.create_neurons(number_neurons_create_1);

    ASSERT_EQ(model.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(model.get_input().size(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(model.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(model.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(model.init(0), RelearnException);

    ASSERT_EQ(model.get_number_neurons(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(model.get_x().size(), number_neurons_init + number_neurons_create_1);
    ASSERT_EQ(model.get_fired().size(), number_neurons_init + number_neurons_create_1);

    model.create_neurons(number_neurons_create_2);

    ASSERT_EQ(model.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(model.get_x().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(model.get_fired().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(model.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(model.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(model.init(0), RelearnException);

    ASSERT_EQ(model.get_number_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(model.get_x().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
    ASSERT_EQ(model.get_fired().size(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(PoissonModelTest, testFootprintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = RandomFactory::get_random_integer<unsigned int>(models::PoissonModel::min_h, models::PoissonModel::max_h, this->mt);
    const auto x0 = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_x_0, models::poisson::Parameters<double, unsigned int>::max_x_0, this->mt);
    const auto tau_x = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_tau_x, models::poisson::Parameters<double, unsigned int>::max_tau_x, this->mt);
    const auto refrac = RandomFactory::get_random_integer<unsigned int>(models::poisson::Parameters<double, unsigned int>::min_refractory_time, models::poisson::Parameters<double, unsigned int>::max_refractory_time, this->mt);

    const auto parameters = models::poisson::Parameters<double, unsigned int>{ x0, tau_x, refrac };
    auto model = models::PoissonModel(h, activity_input, fired_status_comm, parameters);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(model.record_memory_footprint(footprint));
}

TEST_F(PoissonModelTest, testUpdateConstantInput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = 0.5;

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(constant_input);

    const auto h = RandomFactory::get_random_integer<unsigned int>(models::PoissonModel::min_h, models::PoissonModel::max_h, this->mt);
    const auto x0 = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_x_0, models::poisson::Parameters<double, unsigned int>::max_x_0, this->mt);
    const auto tau_x = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_tau_x, models::poisson::Parameters<double, unsigned int>::max_tau_x, this->mt);
    const auto refrac = RandomFactory::get_random_integer<unsigned int>(models::poisson::Parameters<double, unsigned int>::min_refractory_time, models::poisson::Parameters<double, unsigned int>::max_refractory_time, this->mt);

    const auto parameters = models::poisson::Parameters<double, unsigned int>{ x0, tau_x, refrac };
    auto model = models::PoissonModel(h, activity_input, fired_status_comm, parameters);

    const auto number_neurons_init = 41;

    auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());
    fired_status_comm->set_network_graph(network_graph);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons_init);
    model.set_extra_infos(extra_info);

    model.init(number_neurons_init);
    model.update_electrical_activity(102);

    const auto input = model.get_input();
    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        ASSERT_EQ(input[neuron_id], constant_input);
    }

    const auto fired_status = model.get_fired();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        ASSERT_EQ(model.has_fired(neuron_id), fired_status[neuron_id.get_neuron_id()] == FiredStatus::Fired);
    }

    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto status = fired_status[neuron_id];
        const auto refrac_neuron = model.get_refractory_time(NeuronID(neuron_id));
        if (status == FiredStatus::Inactive) {
            ASSERT_EQ(refrac_neuron, 0.0);
        } else {
            ASSERT_EQ(refrac_neuron, static_cast<double>(refrac));
        }
    }

    assert_getter_equality(model);
    assert_getter_throws(model);
}

TEST_F(PoissonModelTest, testUpdateNormalInput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = 0.5;
    const auto stddev_input = 0.2;

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_normal_activity(mean_input, stddev_input);

    const auto h = RandomFactory::get_random_integer<unsigned int>(models::PoissonModel::min_h, models::PoissonModel::max_h, this->mt);
    const auto x0 = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_x_0, models::poisson::Parameters<double, unsigned int>::max_x_0, this->mt);
    const auto tau_x = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_tau_x, models::poisson::Parameters<double, unsigned int>::max_tau_x, this->mt);
    const auto refrac = RandomFactory::get_random_integer<unsigned int>(models::poisson::Parameters<double, unsigned int>::min_refractory_time, models::poisson::Parameters<double, unsigned int>::max_refractory_time, this->mt);

    const auto parameters = models::poisson::Parameters<double, unsigned int>{ x0, tau_x, refrac };
    auto model = models::PoissonModel(h, activity_input, fired_status_comm, parameters);

    const auto number_neurons_init = 48;

    auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());
    fired_status_comm->set_network_graph(network_graph);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons_init);
    model.set_extra_infos(extra_info);

    model.init(number_neurons_init);
    model.update_electrical_activity(102);

    const auto input = model.get_input();
    const auto golden_input = activity_input->get_input();
    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        ASSERT_EQ(input[neuron_id], golden_input[neuron_id]);
    }

    const auto fired_status = model.get_fired();
    for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
        ASSERT_EQ(model.has_fired(neuron_id), fired_status[neuron_id.get_neuron_id()] == FiredStatus::Fired);
    }

    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        const auto status = fired_status[neuron_id];
        const auto refrac_neuron = model.get_refractory_time(NeuronID(neuron_id));
        if (status == FiredStatus::Inactive) {
            ASSERT_EQ(refrac_neuron, 0.0);
        } else {
            ASSERT_EQ(refrac_neuron, static_cast<double>(refrac));
        }
    }

    assert_getter_equality(model);
    assert_getter_throws(model);
}

TEST_F(PoissonModelTest, testMultipleUpdates) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto mean_input = 0.5;
    const auto stddev_input = 0.2;

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_normal_activity(mean_input, stddev_input);

    const auto h = RandomFactory::get_random_integer<unsigned int>(models::PoissonModel::min_h, models::PoissonModel::max_h, this->mt);
    const auto x0 = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_x_0, models::poisson::Parameters<double, unsigned int>::max_x_0, this->mt);
    const auto tau_x = RandomFactory::get_random_double<double>(models::poisson::Parameters<double, unsigned int>::min_tau_x, models::poisson::Parameters<double, unsigned int>::max_tau_x, this->mt);
    const auto refrac = RandomFactory::get_random_integer<unsigned int>(models::poisson::Parameters<double, unsigned int>::min_refractory_time, models::poisson::Parameters<double, unsigned int>::max_refractory_time, this->mt);

    const auto parameters = models::poisson::Parameters<double, unsigned int>{ x0, tau_x, refrac };
    auto model = models::PoissonModel(h, activity_input, fired_status_comm, parameters);

    const auto number_neurons_init = 32;

    auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());
    fired_status_comm->set_network_graph(network_graph);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons_init);
    model.set_extra_infos(extra_info);

    model.init(number_neurons_init);

    auto steps_fired_in = std::vector<std::size_t>(number_neurons_init, 0);

    for (auto cur_step = 102U; cur_step < 123U; cur_step++) {
        model.update_electrical_activity(cur_step);

        for (const auto neuron_id : NeuronID::range(number_neurons_init)) {
            const auto id = neuron_id.get_neuron_id();

            const auto fired = model.has_fired(neuron_id);
            if (fired) {
                const auto last_recorded_spike = steps_fired_in[id];
                steps_fired_in[id] = cur_step;

                if (last_recorded_spike == 0) {
                    // First spike, do not calculate time since last spike
                    continue;
                }

                const auto step_diff = cur_step - last_recorded_spike;
                ASSERT_GE(step_diff, refrac);
            } else {
                if (steps_fired_in[id] == 0) {
                    // No spike yet, do not calculate time since last spike
                    continue;
                }

                const auto steps_since_last_fired = cur_step - steps_fired_in[id];
                if (steps_since_last_fired == 0) {
                    continue;
                }
                ASSERT_EQ(model.get_refractory_time(neuron_id), refrac - steps_since_last_fired);
            }
        }
    }

    assert_getter_equality(model);
    assert_getter_throws(model);
}

TEST_F(PoissonModelTest, testBenchmarkFunctionality) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto activity_input = ActivityInputFactory::construct_constant_activity(0.2);
    auto activity_input_benchmark = ActivityInputFactory::construct_constant_activity(0.2);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto fired_status_comm_benchmark = FiredStatusCommunicatorFactory::construct_map_communicator(1);

    const auto parameters = models::poisson::Parameters<double, unsigned int>{};

    auto model = models::PoissonModel(10, activity_input, fired_status_comm, parameters);
    auto model_benchmark = models::PoissonModel(10, activity_input_benchmark, fired_status_comm_benchmark, parameters);

    const auto number_neurons_init = 400;

    auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());
    auto network_graph_benchmark = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());

    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm_benchmark->set_network_graph(network_graph_benchmark);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    auto extra_info_benchmark = NeuronsExtraInfoFactory::construct_extra_info();

    extra_info->init(number_neurons_init);
    extra_info_benchmark->init(number_neurons_init);

    model.set_extra_infos(extra_info);
    model_benchmark.set_extra_infos(extra_info_benchmark);

    model.init(number_neurons_init);
    model_benchmark.init(number_neurons_init);

    model.update_electrical_activity(102);
    model_benchmark.update_electrical_activity(102);

    const auto input = model.get_input();
    const auto input_benchmark = model.get_input();

    for (const auto neuron_id : NeuronID::range_id(number_neurons_init)) {
        ASSERT_EQ(input[neuron_id], input_benchmark[neuron_id]);
    }
}
