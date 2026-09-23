#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

/**
 * @brief NeuronModel resolves, at compile time, to the neuron-model implementation for this
 *      build: NeuronModelGPU when RELEARN_CUDA_ENABLED, NeuronModelCPU otherwise. A build only
 *      ever compiles one of the two, so this is a plain type alias rather than a runtime choice.
 *      AEIFModel, IzhikevichModel, PoissonModel, and FitzHughNagumoModel are the genuinely
 *      runtime-polymorphic axis (chosen by user config) and derive from this alias.
 */
#ifdef RELEARN_CUDA_ENABLED
#include "cuda/neuron_model/NeuronModelGPU.h"
using NeuronModel = NeuronModelGPU;
#else
#include "NeuronModelCPU.h"
using NeuronModel = NeuronModelCPU;
#endif
