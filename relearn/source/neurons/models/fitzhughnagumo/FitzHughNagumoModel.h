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

#include "neurons/models/NeuronModel.h"
#include "neurons/models/fitzhughnagumo/Parameters.h"

#ifdef RELEARN_CUDA_ENABLED
#include "cuda/neuron_model/fitzhughnagumo/FitzHughNagumoModelGPU.h"
#else
#include "neurons/models/fitzhughnagumo/FitzHughNagumoModelCPU.h"
#endif

namespace models {

/**
 * @brief FitzHughNagumoModel resolves, at compile time, to the FitzHughNagumo model implementation
 *      for this build: FitzHughNagumoModelGPU when RELEARN_CUDA_ENABLED, FitzHughNagumoModelCPU
 *      otherwise. A build only ever compiles one of the two, so this is a plain type alias rather
 *      than a runtime choice.
 *      This class inherits from NeuronModel and implements the spiking model from Fitz, Hugh, Nagumo.
 *      The differential equations are:
 *          d/dt v(t) = v(t) - (v(t)^3)/3 - w(t) + input
 *          d/dt w(t) = phi * (v(t) + a - b * w(t))
 */
#ifdef RELEARN_CUDA_ENABLED
using FitzHughNagumoModel = FitzHughNagumoModelGPU;
#else
using FitzHughNagumoModel = FitzHughNagumoModelCPU;
#endif

} // namespace models
