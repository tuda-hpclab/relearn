#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"

#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"

#include <cpp-utility/Cast.hpp>

#include <cmath>

class NeuronAssignmentTest : public RelearnTest {
protected:
    static RelearnTypes::space_type calculate_box_length(const RelearnTypes::number_neurons_type number_neurons, const RelearnTypes::space_type um_per_neuron) noexcept {
        return utility::cast<RelearnTypes::space_type>(std::ceil(std::pow(static_cast<double>(number_neurons), 1 / 3.))) * um_per_neuron;
    }
};
