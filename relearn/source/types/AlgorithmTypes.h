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

#include "algorithm/CombinedAlgorithmsInternal/AlgorithmConfig.h"
#include "util/NeuronID.h"

#include <cstddef>
#include <utility>
#include <vector>

// The types with which the combined algorithms are configured, i.e., the algorithms that are used and which
// of the neurons each of them is responsible for.

namespace RelearnTypes {

/** @brief The configurations of the algorithms that the combined algorithms pick from */
using AlgorithmConfigs = std::vector<AlgorithmConfig>;

/** @brief Assigns neurons to algorithms, each entry an index into the AlgorithmConfigs and the neurons it handles */
using AlgorithmIndexWithNeuronsType = std::vector<std::pair<std::size_t, std::vector<NeuronID>>>;

} // namespace RelearnTypes
