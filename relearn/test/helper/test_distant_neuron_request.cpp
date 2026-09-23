/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_distant_neuron_request.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "util/RelearnException.h"

#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <iostream>

TEST_F(DistantNeuronRequestTest, testDefaultConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto dnr = DistantNeuronRequest{};

    const auto& source_neuron_id = dnr.get_source_id();
    const auto& source_position = dnr.get_source_position();
    const auto target_neuron_type = dnr.get_target_neuron_type();
    const auto signal_type = dnr.get_signal_type();

    ASSERT_FALSE(source_neuron_id.is_initialized());

    ASSERT_EQ(source_position, RelearnTypes::position_type{});
    ASSERT_EQ(target_neuron_type, DistantNeuronRequest::TargetNeuronType{});
    ASSERT_EQ(signal_type, SignalType{});
}

TEST_F(DistantNeuronRequestTest, testConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& golden_source_neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& golden_source_position = SimulationFactory::get_random_position(mt);

    const auto golden_target_id = NeuronIdFactory::get_random_number_neurons(mt);
    const auto golden_target_neuron_type = NeuronTypesFactory::get_random_target_neuron_type(mt);

    const auto golden_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    const auto dnr = DistantNeuronRequest{ golden_source_neuron_id, golden_source_position, golden_target_id, golden_target_neuron_type, golden_signal_type };

    const auto& source_neuron_id = dnr.get_source_id();
    const auto& source_position = dnr.get_source_position();

    const auto target_neuron_type = dnr.get_target_neuron_type();
    const auto signal_type = dnr.get_signal_type();

    ASSERT_EQ(source_neuron_id, golden_source_neuron_id);
    ASSERT_EQ(source_position, golden_source_position);
    ASSERT_EQ(target_neuron_type, golden_target_neuron_type);
    ASSERT_EQ(signal_type, golden_signal_type);

    if (target_neuron_type == DistantNeuronRequest::TargetNeuronType::Leaf) {
        const auto leaf_neuron_id = dnr.get_leaf_node_id();

        ASSERT_EQ(leaf_neuron_id, golden_target_id);
        ASSERT_THROW_NO_PRINT(std::ignore = dnr.get_rma_offset(), RelearnException);
    }

    if (target_neuron_type == DistantNeuronRequest::TargetNeuronType::VirtualNode) {
        const auto rma_offset = dnr.get_rma_offset();

        ASSERT_EQ(rma_offset, golden_target_id);
        ASSERT_THROW_NO_PRINT(std::ignore = dnr.get_leaf_node_id(), RelearnException);
    }
}

TEST_F(DistantNeuronRequestTest, testConstructorException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& golden_source_position = SimulationFactory::get_random_position(mt);

    const auto golden_target_id = NeuronIdFactory::get_random_number_neurons(mt);
    const auto golden_target_neuron_type = NeuronTypesFactory::get_random_target_neuron_type(mt);

    const auto golden_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    ASSERT_THROW_NO_PRINT(DistantNeuronRequest dnr(NeuronID::virtual_id(), golden_source_position, golden_target_id, golden_target_neuron_type, golden_signal_type), RelearnException);
    ASSERT_THROW_NO_PRINT(DistantNeuronRequest dnr(NeuronID::uninitialized_id(), golden_source_position, golden_target_id, golden_target_neuron_type, golden_signal_type), RelearnException);
}
