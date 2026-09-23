#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

/**
 * @brief SynapticActivityInput resolves, at compile time, to the synaptic-activity-input
 *      implementation for this build: SynapticActivityInputGPU when RELEARN_CUDA_ENABLED,
 *      SynapticActivityInputCPU otherwise. A build only ever compiles one of the two, so this is
 *      a plain type alias rather than a runtime choice. SynapticEquallyWeightedActivityInput,
 *      SynapticIndividuallyWeightedActivityInput, and SynapticScalingActivityInput are the
 *      genuinely runtime-polymorphic axis (chosen by user config) and derive from this alias.
 */
#ifdef RELEARN_CUDA_ENABLED
#include "cuda/input/SynapticActivityInputGPU.h"
using SynapticActivityInput = SynapticActivityInputGPU;
#else
#include "SynapticActivityInputCPU.h"
using SynapticActivityInput = SynapticActivityInputCPU;
#endif
