#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

/**
 * @brief CalciumCalculator resolves, at compile time, to the calcium calculator implementation for
 *      this build: CalciumCalculatorGPU when RELEARN_CUDA_ENABLED, CalciumCalculatorCPU otherwise.
 *      A build only ever compiles one of the two, so this is a plain type alias rather than a
 *      runtime choice: everything that only needs the CPU/GPU-agnostic surface (the decay-law
 *      subclasses, Simulation, Neurons) names CalciumCalculator and gets the right implementation
 *      for the build automatically.
 */
#ifdef RELEARN_CUDA_ENABLED
#include "cuda/calcium/CalciumCalculatorGPU.h"
using CalciumCalculator = CalciumCalculatorGPU;
#else
#include "CalciumCalculatorCPU.h"
using CalciumCalculator = CalciumCalculatorCPU;
#endif
