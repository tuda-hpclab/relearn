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
#include "cuda/util/NeuronsExtraInfoHandle.h"
#include "neurons/enums/FiredStatus.h"

/**
 * @brief Updates the current calcium concentration for all neurons based on firing activity.
 * @param info_handle     Per-neuron metadata (neuron count, disable flags); only those two fields are read.
 * @param d_fired         Per-neuron fired status from the previous simulation step.
 * @param calcium_handle  Device pointers to the calcium state; only .calcium is read/written.
 * @param h               Euler integration step size.
 * @param tau_C           Calcium decay time constant.
 * @param beta            Calcium increment per spike.
 */
void update_current_calcium_entry(NeuronsExtraInfoGPUHandleConst info_handle,
                                  const FiredStatus* d_fired, CalciumHandle calcium_handle, unsigned int h,
                                  CudaConfig::calcium_type tau_C, CudaConfig::calcium_type beta);

/**
 * @brief Decays each neuron's target calcium by a fixed absolute amount.
 * @param info_handle      Per-neuron metadata (neuron count, disable flags); only those two fields are read.
 * @param calcium_handle   Device pointers to the calcium state; only .target_calcium is read/written.
 * @param decay_amount     Amount subtracted from each target calcium value per call.
 */
void update_target_calcium_absolute_decay_entry(NeuronsExtraInfoGPUHandleConst info_handle,
                                                CalciumHandle calcium_handle,
                                                CudaConfig::calcium_type decay_amount);

/**
 * @brief Decays each neuron's target calcium by a relative (multiplicative) amount.
 * @param info_handle      Per-neuron metadata (neuron count, disable flags); only those two fields are read.
 * @param calcium_handle   Device pointers to the calcium state; only .target_calcium is read/written.
 * @param decay_amount     Fraction by which target calcium is reduced per call.
 */
void update_target_calcium_relative_decay_entry(NeuronsExtraInfoGPUHandleConst info_handle,
                                                CalciumHandle calcium_handle,
                                                CudaConfig::calcium_type decay_amount);
