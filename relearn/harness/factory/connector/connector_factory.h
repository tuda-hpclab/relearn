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

#include "Types3.h"

#include "neurons/helper/SynapseCreationRequests.h"

#include <random>
#include <tuple>
#include <vector>

class ConnectorFactory {
public:
    static std::tuple<RelearnTypes::comm_map_creation<SynapseCreationRequest>, std::vector<size_t>, std::vector<size_t>> create_incoming_requests(size_t number_ranks, int current_rank,
                                                                                                                                          size_t number_neurons, size_t number_requests_lower_bound, size_t number_requests_upper_bound, std::mt19937& mt);
};
