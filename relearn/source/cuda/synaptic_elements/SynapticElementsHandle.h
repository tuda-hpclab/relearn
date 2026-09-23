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
 * Read-only GPU view of synaptic element arrays passed into CUDA kernels.
 * Contains raw device pointers for all synaptic element fields of one element type.
 */
struct SynapticElementsBaseCudaHandleConst {
    const CudaConfig::synaptic_grown_type* grown_elements;          ///< Accumulated grown elements per neuron.
    const CudaConfig::synaptic_grown_type* delta_since_last_update; ///< Change in grown elements since last commit.
    const CudaConfig::synaptic_count_type* vacant_elements;         ///< Number of unconnected element slots.
    const CudaConfig::synaptic_count_type* connected_elements;      ///< Number of connected element slots.
    const CudaConfig::synaptic_grown_type* vacant_retract_ratio;    ///< Fraction of vacant elements that can retract.
    const CudaConfig::calcium_type* minimum_calcium;                ///< Minimum calcium level required to grow.
    CudaConfig::number_neurons_type number_neurons;                 ///< Number of neurons this handle covers.
};

/**
 * Read-write GPU view of synaptic element arrays passed into CUDA kernels.
 * Contains raw device pointers for all synaptic element fields of one element type.
 */
struct SynapticElementsBaseCudaHandle {
    CudaConfig::synaptic_grown_type* grown_elements;             ///< Accumulated grown elements per neuron.
    CudaConfig::synaptic_grown_type* delta_since_last_update;    ///< Change in grown elements since last commit.
    CudaConfig::synaptic_count_type* vacant_elements;            ///< Number of unconnected element slots.
    CudaConfig::synaptic_count_type* connected_elements;         ///< Number of connected element slots.
    const CudaConfig::synaptic_grown_type* vacant_retract_ratio; ///< Fraction of vacant elements that can retract.
    const CudaConfig::calcium_type* minimum_calcium;             ///< Minimum calcium level required to grow.
    CudaConfig::number_neurons_type number_neurons;              ///< Number of neurons this handle covers.
};
