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

#include "util/NeuronID.h"

#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

// The external stimulation of the neurons, i.e., which neurons are stimulated with which intensity, and how
// that is looked up for a step of the simulation.

namespace RelearnTypes {

/** @brief The stimuli of one step, each of them a set of neurons and the intensity they are stimulated with */
using stimuli_list_type = std::vector<std::pair<std::unordered_set<NeuronID>, activity_type>>;

/** @brief Returns the stimuli that apply in the passed step */
using stimuli_function_type = std::function<stimuli_list_type(step_type)>;

} // namespace RelearnTypes
