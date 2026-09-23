/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_synapse_creation_request.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "util/RelearnException.h"

#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <iostream>

TEST_F(SynapseCreationTest, testDefaultConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto scr = SynapseCreationRequest{};

    const auto& target_neuron_id = scr.get_target();
    const auto& source_neuron_id = scr.get_source();
    const auto& signal_type = scr.get_signal_type();

    ASSERT_FALSE(target_neuron_id.is_initialized());
    ASSERT_FALSE(source_neuron_id.is_initialized());

    ASSERT_EQ(signal_type, SignalType{});
}

TEST_F(SynapseCreationTest, testConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& golden_target_neuron_id = NeuronIdFactory::get_random_neuron_id(mt);
    const auto& golden_source_neuron_id = NeuronIdFactory::get_random_neuron_id(mt);
    const auto& golden_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    const auto scr = SynapseCreationRequest{ golden_target_neuron_id, golden_source_neuron_id, golden_signal_type };

    const auto& target_neuron_id = scr.get_target();
    const auto& source_neuron_id = scr.get_source();
    const auto& signal_type = scr.get_signal_type();

    ASSERT_EQ(target_neuron_id, golden_target_neuron_id);
    ASSERT_EQ(source_neuron_id, golden_source_neuron_id);

    ASSERT_EQ(signal_type, golden_signal_type);
}

TEST_F(SynapseCreationTest, testConstructorException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& golden_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    const auto& dummy_neuron_id = NeuronIdFactory::get_random_neuron_id(mt);

    ASSERT_THROW_NO_PRINT(SynapseCreationRequest scr(NeuronID::virtual_id(), dummy_neuron_id, golden_signal_type), RelearnException);
    ASSERT_THROW_NO_PRINT(SynapseCreationRequest scr(NeuronID::uninitialized_id(), dummy_neuron_id, golden_signal_type), RelearnException);

    ASSERT_THROW_NO_PRINT(SynapseCreationRequest scr(dummy_neuron_id, NeuronID::virtual_id(), golden_signal_type), RelearnException);
    ASSERT_THROW_NO_PRINT(SynapseCreationRequest scr(dummy_neuron_id, NeuronID::uninitialized_id(), golden_signal_type), RelearnException);

    ASSERT_THROW_NO_PRINT(SynapseCreationRequest scr(NeuronID::virtual_id(), NeuronID::virtual_id(), golden_signal_type), RelearnException);
    ASSERT_THROW_NO_PRINT(SynapseCreationRequest scr(NeuronID::virtual_id(), NeuronID::uninitialized_id(), golden_signal_type), RelearnException);

    ASSERT_THROW_NO_PRINT(SynapseCreationRequest scr(NeuronID::uninitialized_id(), NeuronID::virtual_id(), golden_signal_type), RelearnException);
    ASSERT_THROW_NO_PRINT(SynapseCreationRequest scr(NeuronID::uninitialized_id(), NeuronID::uninitialized_id(), golden_signal_type), RelearnException);
}

TEST_F(SynapseCreationTest, testStructuredBinding) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& golden_target_neuron_id = NeuronIdFactory::get_random_neuron_id(mt);
    const auto& golden_source_neuron_id = NeuronIdFactory::get_random_neuron_id(mt);
    const auto& golden_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    const auto scr = SynapseCreationRequest{ golden_target_neuron_id, golden_source_neuron_id, golden_signal_type };

    const auto& [target_neuron_id, source_neuron_id, signal_type] = scr;

    ASSERT_EQ(target_neuron_id, golden_target_neuron_id);
    ASSERT_EQ(source_neuron_id, golden_source_neuron_id);

    ASSERT_EQ(signal_type, golden_signal_type);
}
