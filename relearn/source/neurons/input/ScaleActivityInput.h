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
 * @brief ScaleActivityInput resolves, at compile time, to the implementation for this build:
 *      ScaleActivityInputGPU when RELEARN_CUDA_ENABLED, ScaleActivityInputCPU otherwise. A build
 *      only ever compiles one of the two, so this is a plain type alias rather than a runtime
 *      choice.
 */
#ifdef RELEARN_CUDA_ENABLED
#include "cuda/input/ScaleActivityInputGPU.h"
using ScaleActivityInput = ScaleActivityInputGPU;
#else
#include "neurons/input/ScaleActivityInputCPU.h"
using ScaleActivityInput = ScaleActivityInputCPU;
#endif
