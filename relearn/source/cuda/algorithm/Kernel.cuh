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

#include "cuda/CudaTypes.h"

// GAUSSIAN KERNEL
__device__ __constant__ constexpr static auto MU_FLOAT = 0.0f; // default in Sebastians work
__device__ __constant__ constexpr static auto MU = 0.0;        // default in Sebastians work

/**
 * @brief Calculates the squared 2-norm (sum of squared differences) between two points
 * @param vec1 The first point
 * @param vec2 The second point
 * @return The sum of squared differences between vec1 and vec2
 */
inline __device__ double calculate_squared_2_norm(const SimpleVec3d& vec1, const SimpleVec3d& vec2) {
    const auto diff_x = vec1.x - vec2.x;
    const auto diff_y = vec1.y - vec2.y;
    const auto diff_z = vec1.z - vec2.z;

    const auto sum = (diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z);

    return sum;
}

/**
 * @brief Calculates the 2-norm (Euclidean distance) between two points
 * @param vec1 First point
 * @param vec2 Second point
 * @return The Euclidean distance between vec1 and vec2
 */
inline __device__ double calculate_2_norm(const SimpleVec3d& vec1, const SimpleVec3d& vec2) {
    const auto sum = calculate_squared_2_norm(vec1, vec2);
    const auto norm = sqrt(sum);

    return norm;
}

/**
 * @brief Calculates the attractiveness of a source neuron to connect to a target neuron, weighted by the amount of free dendritic elements on the target
 * @param source_position The position of the source point
 * @param target_position The position of the target point
 * @param number_free_elements The number of free dendritic elements available for connection
 * @return The attractiveness to connect, or 0 if no free dendritic elements are available or the source position equals the target position
 */
inline __device__ double
calculate_attractiveness_to_connect(const SimpleVec3d& source_position, const SimpleVec3d& target_position,
                                    const double number_free_elements, const bool weighted_dendrites,
                                    double squared_sigma_inv) {
    if (number_free_elements == 0) {
        return 0;
    }

    const auto x = calculate_squared_2_norm(target_position, source_position);

    // prevent autapse
    if (x == 0) {
        return 0;
    }

    // x is already the squared distance (see calculate_squared_2_norm above), matching the CPU
    // reference GaussianDistributionKernel::get_probability, which computes
    // exp(-(distance - mu)^2 / sigma^2) from the *unsquared* distance. Squaring x again here
    // would compute exp(-distance^4 / sigma^2) instead -- a different kernel shape entirely.
    const auto numerator = x - MU;

    const auto exponent = -numerator * squared_sigma_inv;

    // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
    const auto exp_val = exp(exponent);
    const auto ret_val = number_free_elements * exp_val;

    return ret_val;
}