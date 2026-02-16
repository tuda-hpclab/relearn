/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_synapse_deletion_request.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"

#include <gtest/gtest.h>

#include <iostream>

TEST_F(SynapseDeletionTest, testDefaultConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto sdr = SynapseDeletionRequest{};

    const auto& affected_neuron_id = sdr.get_affected_neuron_id();
    const auto& initiator_neuron_id = sdr.get_initiator_neuron_id();
    const auto& initiator_element_type = sdr.get_initiator_element_type();
    const auto& signal_type = sdr.get_signal_type();

    ASSERT_FALSE(affected_neuron_id.is_initialized());
    ASSERT_FALSE(initiator_neuron_id.is_initialized());

    ASSERT_EQ(signal_type, SignalType{});
    ASSERT_EQ(initiator_element_type, ElementType{});
}

TEST_F(SynapseDeletionTest, testConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& golden_affected_neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& golden_initiator_neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& golden_initiator_element_type = NeuronTypesFactory::get_random_element_type(mt);
    const auto& golden_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    const auto sdr = SynapseDeletionRequest{ golden_initiator_neuron_id, golden_affected_neuron_id, golden_initiator_element_type, golden_signal_type };

    const auto& affected_neuron_id = sdr.get_affected_neuron_id();
    const auto& initiator_neuron_id = sdr.get_initiator_neuron_id();
    const auto& initiator_element_type = sdr.get_initiator_element_type();
    const auto& signal_type = sdr.get_signal_type();

    ASSERT_EQ(affected_neuron_id, golden_affected_neuron_id);
    ASSERT_EQ(initiator_neuron_id, golden_initiator_neuron_id);

    ASSERT_EQ(signal_type, golden_signal_type);
    ASSERT_EQ(initiator_element_type, golden_initiator_element_type);
}

TEST_F(SynapseDeletionTest, testConstructorException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& golden_initiator_element_type = NeuronTypesFactory::get_random_element_type(mt);
    const auto& golden_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    const auto& dummy_neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);

    ASSERT_THROW_NO_PRINT(std::ignore = SynapseDeletionRequest(NeuronID::virtual_id(), dummy_neuron_id, golden_initiator_element_type, golden_signal_type), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = SynapseDeletionRequest(NeuronID::uninitialized_id(), dummy_neuron_id, golden_initiator_element_type, golden_signal_type), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = SynapseDeletionRequest(dummy_neuron_id, NeuronID::virtual_id(), golden_initiator_element_type, golden_signal_type), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = SynapseDeletionRequest(dummy_neuron_id, NeuronID::uninitialized_id(), golden_initiator_element_type, golden_signal_type), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = SynapseDeletionRequest(NeuronID::virtual_id(), NeuronID::virtual_id(), golden_initiator_element_type, golden_signal_type), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = SynapseDeletionRequest(NeuronID::virtual_id(), NeuronID::uninitialized_id(), golden_initiator_element_type, golden_signal_type), RelearnException);

    ASSERT_THROW_NO_PRINT(std::ignore = SynapseDeletionRequest(NeuronID::uninitialized_id(), NeuronID::virtual_id(), golden_initiator_element_type, golden_signal_type), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = SynapseDeletionRequest(NeuronID::uninitialized_id(), NeuronID::uninitialized_id(), golden_initiator_element_type, golden_signal_type), RelearnException);
}

TEST_F(SynapseDeletionTest, testStructuredBinding) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto& golden_affected_neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& golden_initiator_neuron_id = NeuronIdFactory::get_random_neuron_id(10000, mt);
    const auto& golden_initiator_element_type = NeuronTypesFactory::get_random_element_type(mt);
    const auto& golden_signal_type = NeuronTypesFactory::get_random_signal_type(mt);

    const auto sdr = SynapseDeletionRequest{ golden_initiator_neuron_id, golden_affected_neuron_id, golden_initiator_element_type, golden_signal_type };

    const auto& [initiator_neuron_id, affected_neuron_id, initiator_element_type, signal_type] = sdr;

    ASSERT_EQ(affected_neuron_id, golden_affected_neuron_id);
    ASSERT_EQ(initiator_neuron_id, golden_initiator_neuron_id);

    ASSERT_EQ(signal_type, golden_signal_type);
    ASSERT_EQ(initiator_element_type, golden_initiator_element_type);
}
