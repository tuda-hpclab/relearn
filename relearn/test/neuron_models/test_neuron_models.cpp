/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_models.h"

#include "RelearnTest.hpp"

#include "neurons/enums/FiredStatus.h"
#include "neurons/models/NeuronModel.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

void NeuronModelsTest::assert_getter_equality(const NeuronModel& model) {
    const auto number_neurons = model.get_number_neurons();

    const auto& all_fired = model.get_fired();
    const auto& all_i_syn = model.get_input();
    const auto& all_x = model.get_x();

    ASSERT_EQ(all_fired.size(), number_neurons);
    ASSERT_EQ(all_i_syn.size(), number_neurons);
    ASSERT_EQ(all_x.size(), number_neurons);

    const auto& valid_ids = NeuronIDRange::range(number_neurons);
    for (const auto& neuron_id : valid_ids) {
        ASSERT_NO_THROW(std::ignore = model.has_fired(neuron_id););
        ASSERT_NO_THROW(std::ignore = model.get_input(neuron_id););
        ASSERT_NO_THROW(std::ignore = model.get_x(neuron_id););
    }

    for (const auto neuron_id : valid_ids) {
        ASSERT_EQ(model.has_fired(neuron_id), all_fired[neuron_id.get_neuron_id()] == FiredStatus::Fired);
        ASSERT_EQ(model.get_input(neuron_id), all_i_syn[neuron_id.get_neuron_id()]);
        ASSERT_EQ(model.get_x(neuron_id), all_x[neuron_id.get_neuron_id()]);
    }
}

void NeuronModelsTest::assert_getter_throws(const NeuronModel& model) {
    const auto number_neurons = model.get_number_neurons();

    const auto& invalid_local_ids = NeuronIDRange::range(number_neurons, number_neurons + number_neurons);
    for (const auto neuron_id : invalid_local_ids) {
        ASSERT_THROW_NO_PRINT(std::ignore = model.has_fired(neuron_id), RelearnException);
        ASSERT_THROW_NO_PRINT(std::ignore = model.get_input(neuron_id), RelearnException);
        ASSERT_THROW_NO_PRINT(std::ignore = model.get_x(neuron_id), RelearnException);
    }
}
