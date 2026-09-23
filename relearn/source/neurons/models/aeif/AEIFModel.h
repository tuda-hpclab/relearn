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
#include "neurons/models/aeif/Parameters.h"

#ifdef RELEARN_CUDA_ENABLED
#include "cuda/neuron_model/aeif/AEIFModelGPU.h"
#else
#include "neurons/models/aeif/AEIFModelCPU.h"
#endif

namespace models {

/**
 * @brief AEIFModel resolves, at compile time, to the AEIF model implementation for this build:
 *      AEIFModelGPU when RELEARN_CUDA_ENABLED, AEIFModelCPU otherwise. A build only ever compiles
 *      one of the two, so this is a plain type alias rather than a runtime choice.
 *      This class inherits from NeuronModel and implements an exponential spiking model from Brette and Gerstner.
 *      The differential equations are:
 *          d/dt v(t) = (-g_L * (v(t) - E_L) + g_L * d_T * exp((v(t) - V_T) / d_T) - w(t) + input) / C
 *          d/dt w(t) = (a * (v(t) - E_L) - w(t)) / tau_W
 *      If v(t) >= V_spike:
 *          v(t) = E_L
 *          w(t) += b
 */
#ifdef RELEARN_CUDA_ENABLED
using AEIFModel = AEIFModelGPU;
#else
using AEIFModel = AEIFModelCPU;
#endif

} // namespace models
