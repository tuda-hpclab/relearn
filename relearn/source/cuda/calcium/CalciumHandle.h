#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cuda/CudaConfig.h"

/**
 * Device pointers to a neuron's calcium state. Shared between the current-calcium update and the
 * target-calcium decay kernels (Calcium.h) and the synaptic-elements growth kernel
 * (update_number_elements_kernel_entry), each of which only reads the field(s) it needs.
 */
struct CalciumHandle {
    CudaConfig::calcium_type* calcium{};        ///< Current inter-cellular calcium concentration per neuron.
    CudaConfig::calcium_type* target_calcium{}; ///< Target calcium concentration per neuron.
};

struct CalciumHandleConst {
    const CudaConfig::calcium_type* calcium{};        ///< Current inter-cellular calcium concentration per neuron.
    const CudaConfig::calcium_type* target_calcium{}; ///< Target calcium concentration per neuron.
};
