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

#include "cuda/CudaConfig.h"
#include "cuda/CudaTypes.h"
#include "cuda/util/LinearizedTreeDeviceHandle.h"

#include <cuda.h>

namespace BarnesHutCUDA_CU {
__device__ CudaConfig::bh_index_type
find_single_target_neuron(const std::uint32_t thread_id, const std::uint32_t number_threads, const std::uint64_t seed, const std::uint64_t step,
                          const CudaConfig::bh_index_type start_index,
                          const SimpleVec3d& source_position,
                          const NeuronPopulationDeviceHandle population,
                          const LinearizedTreeDeviceHandle tree,
                          const CudaConfig::gaussian_type acceptance_criterion,
                          const CudaConfig::mpi_rank_type* const neuron_ranks,
                          const CudaConfig::mpi_rank_type my_rank,
                          const CudaConfig::gaussian_type squared_sigma_inv);
}