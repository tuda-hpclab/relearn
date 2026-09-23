#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CudaTypes.h"

#include "types/BasicTypes.h"

#include <cmath>
#include <cstdint>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

#ifndef __global__
// Falls der NVCC die Datei inkludiert, hat dieser __global__ bereits definiert und ignoriert das hier.
// Falls der Host-Compiler die Datei inkludiert, ist dieser von __global__ verwirrt, also sagen wir ihm, das ist eigentlich leer.
#define __global__
#endif

#ifndef __device__
// Falls der NVCC die Datei inkludiert, hat dieser __device__ bereits definiert und ignoriert das hier.
// Falls der Host-Compiler die Datei inkludiert, ist dieser von __device__ verwirrt, also sagen wir ihm, das ist eigentlich leer.
#define __device__
#endif

#ifndef __host__
// Falls der NVCC die Datei inkludiert, hat dieser __host__ bereits definiert und ignoriert das hier.
// Falls der Host-Compiler die Datei inkludiert, ist dieser von __host__ verwirrt, also sagen wir ihm, das ist eigentlich leer.
#define __host__
#endif

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// curand API error checking
#define CURAND_CHECK(err)                                                        \
    do {                                                                         \
        curandStatus_t err_ = (err);                                             \
        if (err_ != CURAND_STATUS_SUCCESS) {                                     \
            std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("curand error");                            \
        }                                                                        \
    } while (0)

// Macro for cudaMalloc and cudaMemcpy from host to device
#define CUDA_MALLOC_AND_MEMCPY(dst, src, size, kind)          \
    CUDA_CHECK(cudaMalloc(&(dst), (size)));                   \
    if (src != nullptr) {                                     \
        CUDA_CHECK(cudaMemcpy((dst), (src), (size), (kind))); \
    }

// Macro for cudaMalloc and memset
#define CUDA_MALLOC_AND_MEMSET(dst, value, size) \
    CUDA_CHECK(cudaMalloc(&(dst), (size)));      \
    CUDA_CHECK(cudaMemset((dst), (value), (size)));

#define RELEARN_CUDA_CHECK(cond, msg)            \
    if (!cond) {                                 \
        std::printf("Runtime error: %s\n", msg); \
        throw std::runtime_error("CUDA error");  \
    }

#define CUDA_THREADS_PER_BLOCK 128U

// CONSTANTS
__device__ __constant__ constexpr static auto EPS = 0.00001;
__device__ __constant__ constexpr static auto NUMBER_OCT = 8;
// GAUSSIAN KERNEL
__device__ __constant__ constexpr static auto MU_FLOAT = 0.0f;                                                 // default in Sebastians work
__device__ __constant__ constexpr static auto MU = 0.0;                                                        // default in Sebastians work
__device__ __constant__ constexpr static auto SQUARED_SIGMA_INV = 1.0 / (12.0 * 12.0);                         // This is a param we used for calling the Kernel
__device__ __constant__ constexpr static auto SQUARED_SIGMA_INV_FLOAT = static_cast<float>(SQUARED_SIGMA_INV); // This is a param we used for calling the Kernel
// BH
// Spelled with RelearnTypes::as so that the constant has the type of the parameters it is the default of, in every configuration of CudaTypes::cuda_real
__device__ __constant__ constexpr static auto DEFAULT_THETA = RelearnTypes::as<CudaTypes::cuda_acceptance_criterion>(0.3);

inline std::uint32_t divup(const std::uint32_t n, const std::uint32_t d) {
    auto r = n / d;

    if (r * d < n) {
        r++;
    }
    return r;
}

/**
 * @brief Calculates the squared 2-norm (sum of squared differences) between two points
 * @param vec1 The first point
 * @param vec2 The second point
 * @return The sum of squared differences between vec1 and vec2
 */
inline __device__ double calculate_squared_2_norm(const double3& vec1, const double3& vec2) {
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
inline __device__ double calculate_2_norm(const double3& vec1, const double3& vec2) {
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
inline __device__ double calculate_attractiveness_to_connect(const double3& source_position, const double3& target_position, const double number_free_elements, const bool weighted_dendrites = false) {
    if (number_free_elements == 0) {
        return 0;
    }

    const auto x = calculate_2_norm(target_position, source_position);

    // prevent autapse
    if (x == 0) {
        return 0;
    }

    // when neurons are too far away, use weighted dendrites
    if (weighted_dendrites) {
        const auto ret_val = number_free_elements / x;
        return ret_val;
    }

    const auto numerator = (x - MU) * (x - MU);

    const auto exponent = -numerator * SQUARED_SIGMA_INV;

    // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
    const auto exp_val = exp(exponent);
    const auto ret_val = number_free_elements * exp_val;

    return ret_val;
}

/**
 * @brief Calculates the squared 2-norm (sum of squared differences) between two points
 * @param vec1 The first point
 * @param vec2 The second point
 * @return The sum of squared differences between vec1 and vec2
 */
inline __device__ float calculate_squared_2_norm(const float3& vec1, const float3& vec2) {
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
inline __device__ float calculate_2_norm(const float3& vec1, const float3& vec2) {
    const auto sum = calculate_squared_2_norm(vec1, vec2);
    const auto norm = sqrtf(sum);

    return norm;
}

/**
 * @brief Calculates the attractiveness of a source neuron to connect to a target neuron, weighted by the amount of free dendritic elements on the target
 * @param source_position The position of the source point
 * @param target_position The position of the target point
 * @param number_free_elements The number of free dendritic elements available for connection
 * @return The attractiveness to connect, or 0 if no free dendritic elements are available or the source position equals the target position
 */
inline __device__ float calculate_attractiveness_to_connect(const float3& source_position, const float3& target_position, const float number_free_elements, const bool weighted_dendrites = true) {
    if (number_free_elements == 0) {
        return 0;
    }

    const auto x = calculate_2_norm(target_position, source_position);

    // prevent autapse
    if (x == 0) {
        return 0;
    }

    // when neurons are too far away, use weighted dendrites
    if (weighted_dendrites) {
        const auto ret_val = number_free_elements / x;
        return ret_val;
    }

    const auto numerator = (x - MU_FLOAT) * (x - MU_FLOAT);

    const auto exponent = -numerator * SQUARED_SIGMA_INV_FLOAT;

    // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
    const auto exp_val = expf(exponent);
    const auto ret_val = number_free_elements * exp_val;

    return ret_val;
}