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
#include "neurons/models/izhikevich/Parameters.h"

#ifdef RELEARN_CUDA_ENABLED
#include "cuda/neuron_model/izhikevich/IzhikevichModelGPU.h"
#else
#include "neurons/models/izhikevich/IzhikevichModelCPU.h"
#endif

namespace models {

/**
 * @brief IzhikevichModel resolves, at compile time, to the Izhikevich model implementation for
 *      this build: IzhikevichModelGPU when RELEARN_CUDA_ENABLED, IzhikevichModelCPU otherwise. A
 *      build only ever compiles one of the two, so this is a plain type alias rather than a
 *      runtime choice.
 *      This class inherits from NeuronModel and implements the spiking model from Izhikevich.
 *      The differential equations are:
 *          d/dt v(t) = k1 * v(t)^2 + k2 * v(t) + k3 - u(t) + input
 *          d/dt u(t) = a * (b * v(t) - u(t))
 *      If v(t) >= V_spike:
 *          v(t) = c
 *          u(t) += d
 */
#ifdef RELEARN_CUDA_ENABLED
using IzhikevichModel = IzhikevichModelGPU;
#else
using IzhikevichModel = IzhikevichModelCPU;
#endif

} // namespace models
