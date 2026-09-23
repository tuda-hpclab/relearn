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
#include "neurons/models/poisson/Parameters.h"

#ifdef RELEARN_CUDA_ENABLED
#include "cuda/neuron_model/poisson/PoissonModelGPU.h"
#else
#include "neurons/models/poisson/PoissonModelCPU.h"
#endif

namespace models {

/**
 * @brief PoissonModel resolves, at compile time, to the Poisson model implementation for this
 *      build: PoissonModelGPU when RELEARN_CUDA_ENABLED, PoissonModelCPU otherwise. A build only
 *      ever compiles one of the two, so this is a plain type alias rather than a runtime choice.
 *      This class inherits from NeuronModel and implements a poisson spiking model.
 */
#ifdef RELEARN_CUDA_ENABLED
using PoissonModel = PoissonModelGPU;
#else
using PoissonModel = PoissonModelCPU;
#endif

} // namespace models
