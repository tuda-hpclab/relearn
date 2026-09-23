#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"

#include <vector>

/**
 * This struct summarizes neurons loaded from a file. It is made up of:
 * The minimum and maximum of (x, y, z)-positions of the neurons,
 * the number of loaded excitatory neurons, and the number
 * of loaded inhibitory neurons
 */
struct AdditionalPositionInformation {
    RelearnTypes::bounding_box_type sim_size;
    std::vector<RelearnTypes::bounding_box_type> subdomain_sizes{};
    RelearnTypes::number_neurons_type total_neurons{};
    RelearnTypes::number_neurons_type local_neurons{};
};
