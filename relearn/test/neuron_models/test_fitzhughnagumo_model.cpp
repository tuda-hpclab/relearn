/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (phi) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_models.h"

#include "neurons/enums/FiredStatus.h"
#include "neurons/models/fitzhughnagumo/FitzHughNagumoModel.h"
#include "neurons/models/fitzhughnagumo/Parameters.h"
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

#include <iostream>
#include <memory>
#include <tuple>

#include "test_neuron_models.h"

TEST_F(FitzHughNagumoModelTest, testDefaultParameters) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using param_type = models::fitzhughnagumo::Parameters<double>;

    const auto default_parameters = param_type{};

    ASSERT_EQ(default_parameters.get_a(), param_type::default_a);
    ASSERT_EQ(default_parameters.get_b(), param_type::default_b);
    ASSERT_EQ(default_parameters.get_phi(), param_type::default_phi);
    ASSERT_EQ(default_parameters.get_init_x(), param_type::default_init_x);
    ASSERT_EQ(default_parameters.get_init_w(), param_type::default_init_w);
}

TEST_F(FitzHughNagumoModelTest, testParameters) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto parameters = models::fitzhughnagumo::Parameters<double>{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    ASSERT_EQ(parameters.get_a(), 1.0);
    ASSERT_EQ(parameters.get_b(), 2.0);
    ASSERT_EQ(parameters.get_phi(), 3.0);
    ASSERT_EQ(parameters.get_init_x(), 4.0);
    ASSERT_EQ(parameters.get_init_w(), 5.0);
}

TEST_F(FitzHughNagumoModelTest, testDefaultConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = models::FitzHughNagumoModel::default_h;
    const auto parameters = models::fitzhughnagumo::Parameters<double>{};

    ASSERT_NO_THROW(std::ignore = models::FitzHughNagumoModel(h, activity_input, fired_status_comm, parameters));
}

TEST_F(FitzHughNagumoModelTest, testGetter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = RandomFactory::get_random_integer<unsigned int>(models::FitzHughNagumoModel::min_h, models::FitzHughNagumoModel::max_h, this->mt);
    const auto a = RandomFactory::get_random_double<double>(models::fitzhughnagumo::Parameters<double>::min_a, models::fitzhughnagumo::Parameters<double>::max_a, this->mt);
    const auto b = RandomFactory::get_random_double<double>(models::fitzhughnagumo::Parameters<double>::min_b, models::fitzhughnagumo::Parameters<double>::max_b, this->mt);
    const auto phi = RandomFactory::get_random_double<double>(models::fitzhughnagumo::Parameters<double>::min_phi, models::fitzhughnagumo::Parameters<double>::max_phi, this->mt);
    const auto init_x = RandomFactory::get_random_double<double>(models::fitzhughnagumo::Parameters<double>::min_init_x, models::fitzhughnagumo::Parameters<double>::max_init_x, this->mt);
    const auto init_w = RandomFactory::get_random_double<double>(models::fitzhughnagumo::Parameters<double>::min_init_w, models::fitzhughnagumo::Parameters<double>::max_init_w, this->mt);

    const auto parameters = models::fitzhughnagumo::Parameters<double>{ a, b, phi, init_x, init_w };
    auto model = models::FitzHughNagumoModel(h, activity_input, fired_status_comm, parameters);

    ASSERT_EQ(model.get_h(), h);
    ASSERT_EQ(model.get_number_neurons(), 0);
    ASSERT_EQ(model.get_model_parameters().get_a(), a);
    ASSERT_EQ(model.get_model_parameters().get_b(), b);
    ASSERT_EQ(model.get_model_parameters().get_phi(), phi);
    ASSERT_EQ(model.get_model_parameters().get_init_x(), init_x);
    ASSERT_EQ(model.get_model_parameters().get_init_w(), init_w);
}

TEST_F(FitzHughNagumoModelTest, testConstructorThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = models::FitzHughNagumoModel::default_h;
    const auto parameters = models::fitzhughnagumo::Parameters<double>{};

    auto fired_status_comm_empty = fired_status_comm;
    fired_status_comm_empty.reset();

    auto activity_input_empty = activity_input;
    activity_input_empty.reset();

    ASSERT_THROW_NO_PRINT(std::ignore = models::FitzHughNagumoModel(0, activity_input, fired_status_comm, parameters), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = models::FitzHughNagumoModel(h, activity_input, fired_status_comm_empty, parameters), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = models::FitzHughNagumoModel(h, activity_input_empty, fired_status_comm, parameters), RelearnException);
}

TEST_F(FitzHughNagumoModelTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = models::FitzHughNagumoModel::default_h;
    const auto parameters = models::fitzhughnagumo::Parameters<double>{};

    auto model = models::FitzHughNagumoModel(h, activity_input, fired_status_comm, parameters);

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

TEST_F(FitzHughNagumoModelTest, testFootprintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(0.0);

    const auto h = models::FitzHughNagumoModel::default_h;
    const auto parameters = models::fitzhughnagumo::Parameters<double>{};

    auto model = models::FitzHughNagumoModel(h, activity_input, fired_status_comm, parameters);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(model.record_memory_footprint(footprint));
}

TEST_F(FitzHughNagumoModelTest, testUpdateConstantInput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto constant_input = 0.5;

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(constant_input);

    const auto h = models::FitzHughNagumoModel::default_h;
    const auto parameters = models::fitzhughnagumo::Parameters<double>{};

    auto model = models::FitzHughNagumoModel(h, activity_input, fired_status_comm, parameters);

    const auto number_neurons_init = 41;

    const auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());
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
        ASSERT_EQ(FiredStatus::Inactive, fired_status[neuron_id.get_neuron_id()]);
    }

    assert_getter_equality(model);
    assert_getter_throws(model);
}

TEST_F(FitzHughNagumoModelTest, testUpdateNormalInput) {
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

    const auto h = models::FitzHughNagumoModel::default_h;
    const auto parameters = models::fitzhughnagumo::Parameters<double>{};

    auto model = models::FitzHughNagumoModel(h, activity_input, fired_status_comm, parameters);

    const auto number_neurons_init = 48;

    const auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());
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

    assert_getter_equality(model);
    assert_getter_throws(model);
}

TEST_F(FitzHughNagumoModelTest, testMultipleUpdates) {
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

    const auto h = models::FitzHughNagumoModel::default_h;
    const auto parameters = models::fitzhughnagumo::Parameters<double>{};

    auto model = models::FitzHughNagumoModel(h, activity_input, fired_status_comm, parameters);

    const auto number_neurons_init = 32;

    const auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());
    fired_status_comm->set_network_graph(network_graph);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons_init);
    model.set_extra_infos(extra_info);

    model.init(number_neurons_init);

    assert_getter_equality(model);
    assert_getter_throws(model);
}

TEST_F(FitzHughNagumoModelTest, testBenchmarkFunctionality) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto activity_input = ActivityInputFactory::construct_constant_activity(5.0);
    auto activity_input_benchmark = ActivityInputFactory::construct_constant_activity(5.0);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_map_communicator(1);
    auto fired_status_comm_benchmark = FiredStatusCommunicatorFactory::construct_map_communicator(1);

    const auto parameters = models::fitzhughnagumo::Parameters<double>{};

    auto model = models::FitzHughNagumoModel(10, activity_input, fired_status_comm, parameters);
    auto model_benchmark = models::FitzHughNagumoModel(10, activity_input_benchmark, fired_status_comm_benchmark, parameters);

    const auto number_neurons_init = 400;

    const auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());
    const auto network_graph_benchmark = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons_init, mpiPP::MPIRank::root_rank());

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
        ASSERT_EQ(model.get_w(NeuronID(neuron_id)), model_benchmark.get_w(NeuronID(neuron_id)));
    }
}
