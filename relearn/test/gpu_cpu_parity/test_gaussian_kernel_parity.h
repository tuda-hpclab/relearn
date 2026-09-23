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

#ifdef RELEARN_CUDA_ENABLED

// Runs the GPU's calculate_attractiveness_to_connect(source, target, number_free_elements,
// weighted_dendrites=false, squared_sigma_inv) on a single thread and returns its result. Used to
// verify it stays numerically identical to the CPU reference GaussianDistributionKernel, since
// both are meant to implement the exact same connection-probability formula (see
// GaussianDistributionKernel::get_probability in algorithm/Kernel/Gaussian.h).
double device_calculate_attractiveness_to_connect(
    double source_x, double source_y, double source_z,
    double target_x, double target_y, double target_z,
    double number_free_elements, double squared_sigma_inv);

#endif
