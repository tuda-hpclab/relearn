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
#include "cuda/calcium/CalciumHandle.h"
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "cuda/util/NeuronsExtraInfoHandle.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <memory>

class StreamWrapper;

/**
 * @brief Commits pending element deletions, reducing vacant and connected counts accordingly.
 * @param handle             Read-write GPU handle for the synaptic element arrays to update.
 * @param d_to_delete_elements Device array specifying how many elements to delete per neuron.
 * @param info_handle        Read-only neuron metadata (count, rank, disable flags).
 * @param stream             CUDA stream to use for the kernel launch.
 */
void commit_synaptic_elements_entry(SynapticElementsBaseCudaHandle handle,
                                    CudaConfig::synaptic_count_type* d_to_delete_elements,
                                    NeuronsExtraInfoGPUHandleConst info_handle,
                                    const std::shared_ptr<StreamWrapper>& stream);

/**
 * @brief Recomputes the number of grown elements based on the difference between calcium and target calcium.
 * @param calcium_handle    Device pointers to the calcium state (both .calcium and .target_calcium read).
 * @param handle            Read-write GPU handle for the synaptic element arrays to update.
 * @param info_handle       Read-only neuron metadata (count, rank, disable flags).
 * @param growth_rate       Scaling factor applied to the calcium-driven growth signal.
 * @param stream            CUDA stream to use for the kernel launch.
 */
void update_number_elements_kernel_entry(CalciumHandleConst calcium_handle,
                                         SynapticElementsBaseCudaHandle handle,
                                         NeuronsExtraInfoGPUHandleConst info_handle,
                                         CudaConfig::synaptic_grown_type growth_rate,
                                         const std::shared_ptr<StreamWrapper>& stream);
