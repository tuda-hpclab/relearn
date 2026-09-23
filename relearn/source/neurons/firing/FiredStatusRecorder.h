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
 * @brief FiredStatusRecorder resolves, at compile time, to the fired-status recorder
 *      implementation for this build: FiredStatusRecorderGPU when RELEARN_CUDA_ENABLED,
 *      FiredStatusRecorderCPU otherwise. A build only ever compiles one of the two, so this is a
 *      plain type alias rather than a runtime choice.
 */
#ifdef RELEARN_CUDA_ENABLED
#include "cuda/firing/FiredStatusRecorderGPU.h"
using FiredStatusRecorder = FiredStatusRecorderGPU;
#else
#include "FiredStatusRecorderCPU.h"
using FiredStatusRecorder = FiredStatusRecorderCPU;
#endif
