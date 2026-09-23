#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BasicTypes.h"

#include <functional>

// The synaptic elements of the neurons, as far as they are not a plain grown_type, which BasicTypes.h has.

namespace RelearnTypes {

/** @brief Maps the id of a neuron to one of its continuous synaptic element values, e.g., to its growth rate or its retract ratio */
using neuron_id_to_grown_calculator = std::function<grown_type(const number_neurons_type)>;

} // namespace RelearnTypes
