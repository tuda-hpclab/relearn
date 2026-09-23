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
#include "neurons/enums/UpdateStatus.h"

/**
 * Read-only view of per-neuron metadata passed into CUDA kernels.
 */
struct NeuronsExtraInfoGPUHandleConst {
    CudaConfig::number_neurons_type number_neurons{}; ///< Total number of neurons on this rank.
    CudaConfig::mpi_rank_type my_rank{};              ///< MPI rank of this process.
    CudaConfig::mpi_rank_type number_ranks{};         ///< Total number of MPI ranks.
    const UpdateStatus* disable_flags{};              ///< Per-neuron flag indicating disabled (non-updating) neurons.
};
