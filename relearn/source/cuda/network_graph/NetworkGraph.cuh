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

#include "GPUEdges.h"

#include "cuda/network_graph/Views.cuh"

#include <cuda.h>

using mpi_rank_type = std::uint16_t;
using neuron_id_type = std::uint32_t;
using weight_type = std::int16_t;
