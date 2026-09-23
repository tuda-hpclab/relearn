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
 * @brief ActivityInput resolves, at compile time, to the activity-input implementation for this
 *      build: ActivityInputGPU when RELEARN_CUDA_ENABLED, ActivityInputCPU otherwise. A build
 *      only ever compiles one of the two, so this is a plain type alias rather than a runtime
 *      choice. All the concrete activity inputs (ConstantActivityInput, SynapticActivityInput,
 *      ...) are the genuinely runtime-polymorphic axis (chosen by user config) and derive from
 *      this alias.
 */
#ifdef RELEARN_CUDA_ENABLED
#include "cuda/input/ActivityInputGPU.h"
using ActivityInput = ActivityInputGPU;
#else
#include "ActivityInputCPU.h"
using ActivityInput = ActivityInputCPU;
#endif
