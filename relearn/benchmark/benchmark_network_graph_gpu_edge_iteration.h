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

#include "cuda/network_graph/NetworkHandle.h"

#include <cstdint>

void launch_kernel_iterate_edges(const NetworkHandle& network, uint32_t test_neuron_id,
                                 uint64_t* d_edge_count);
