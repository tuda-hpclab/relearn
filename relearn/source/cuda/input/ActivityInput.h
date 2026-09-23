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
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/enums/UpdateStatus.h"

#include <cstdint>
#include <memory>

/**
 * @brief Sums multiple sub-input arrays element-wise into a single combined input array.
 * @param first            Index of the first neuron to process (inclusive).
 * @param last             Index past the last neuron to process (exclusive).
 * @param sub_input_size   Number of sub-input arrays to combine.
 * @param d_input          Device output array for the combined input.
 * @param d_sub_input_ptrs Array of device pointers to each sub-input array.
 * @param stream           CUDA stream to use.
 */
void update_combined_activity_input_range_entry(CudaConfig::number_neurons_type first, CudaConfig::number_neurons_type last,
                                                std::size_t sub_input_size, CudaConfig::input_type* d_input,
                                                CudaConfig::input_type** d_sub_input_ptrs, const std::shared_ptr<StreamWrapper>& stream);

/**
 * @brief Sets a constant baseline input current for all non-disabled neurons in [first, last).
 * @param first           Index of the first neuron to process (inclusive).
 * @param last            Index past the last neuron to process (exclusive).
 * @param d_disable_flags Per-neuron disable flags; disabled neurons receive no input.
 * @param d_input         Device output array; base_input is written for each active neuron.
 * @param base_input      Constant input value to assign.
 * @param stream          CUDA stream to use.
 */
void update_constant_activity_input_range_entry(CudaConfig::number_neurons_type first, CudaConfig::number_neurons_type last,
                                                const UpdateStatus* d_disable_flags, CudaConfig::input_type* d_input,
                                                CudaConfig::input_type base_input, const std::shared_ptr<StreamWrapper>& stream);

/**
 * @brief Draws normally-distributed random input currents and writes them into d_input.
 * @param first       Index of the first neuron to process (inclusive).
 * @param last        Index past the last neuron to process (exclusive).
 * @param d_disable_flags Per-neuron disable flags; disabled neurons receive no input.
 * @param d_input     Device output array; sampled values are written per neuron.
 * @param mean        Mean of the normal distribution.
 * @param stddev      Standard deviation of the normal distribution.
 * @param random_key  Key selecting the pre-registered cuRAND stream to draw from.
 * @param stream      CUDA stream to use.
 */
void update_normal_activity_input_range_entry(CudaConfig::number_neurons_type first, CudaConfig::number_neurons_type last,
                                              const UpdateStatus* d_disable_flags, CudaConfig::input_type* d_input, CudaConfig::input_type mean,
                                              CudaConfig::input_type stddev, std::uint32_t random_key, const std::shared_ptr<StreamWrapper>& stream);
