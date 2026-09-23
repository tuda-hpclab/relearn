#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BasicTypes.h"

#include <functional>

// The calcium of the neurons, as far as it is not a plain calcium_type, which BasicTypes.h has.

namespace RelearnTypes {

/** @brief Maps the id of a neuron to one of its calcium values, e.g., to its initial or its target concentration */
using neuron_id_to_calcium_calculator = std::function<calcium_type(const number_neurons_type)>;

} // namespace RelearnTypes
