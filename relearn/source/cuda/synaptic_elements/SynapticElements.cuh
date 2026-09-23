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

#include <cuda.h>

__device__ void connect_elements(const CudaConfig::synaptic_count_type newly_connected, const CudaConfig::number_neurons_type neuron_id, CudaConfig::synaptic_count_type* d_connected_elements, CudaConfig::synaptic_count_type* d_vacant_elements, const CudaConfig::number_neurons_type size);

__device__ void disconnect_elements(const CudaConfig::synaptic_count_type newly_disconnected, const CudaConfig::number_neurons_type neuron_id, CudaConfig::synaptic_count_type* d_connected_elements, CudaConfig::synaptic_count_type* d_vacant_elements, const CudaConfig::number_neurons_type size);