/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NaiveCUDA_CU.h"

#ifndef RELEARN_CUDA_ENABLED

#include "cuda/CudaTypes.h"
#include "cuda/util/Util.h"

#include <cstdint>
#include <vector>

void NaiveCUDA_CU::find_target_neurons(const DeviceArray<SimpleVec3d>&, const std::uint64_t,
                                       NaiveTargetSelectionTask,
                                       std::vector<std::uint64_t>&,
                                       double) { CUDA_NOT_SUPPORTED }

#endif