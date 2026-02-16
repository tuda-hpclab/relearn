#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"
#include "algorithm/CombinedAlgorithmsInternal/AlgorithmConfig.h"
#include "util/NeuronID.h"

#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace RelearnTypes {
// These types are 'hard'

using stimuli_list_type = std::vector<std::pair<std::unordered_set<NeuronID>, double>>;
using stimuli_function_type = std::function<stimuli_list_type(step_type)>;

using neuron_id_to_calcium_calculator = std::function<calcium_type(const number_neurons_type)>;

using AlgorithmConfigs = std::vector<AlgorithmConfig>;
using AlgorithmIndexWithNeuronsType = std::vector<std::pair<std::size_t, std::vector<NeuronID>>>;

} // namespace RelearnTypes