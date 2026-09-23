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

#include "cuda/memory/DeviceArray.h"

#include <cuda.h>

#include <cassert>
#include <iostream>

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
// CONSTANTS
__device__ __constant__ constexpr static auto EPS = 0.00001;
__device__ __constant__ constexpr static auto NUMBER_OCT = 8;

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

#define RELEARN_CUDA_CHECK(cond, msg)            \
    if (!(cond)) {                               \
        std::printf("Runtime error: %s\n", msg); \
        throw std::runtime_error("CUDA error");  \
    }

#define RELEARN_DEVICE_CUDA_CHECK(cond, fmt, ...)                   \
    do {                                                            \
        if (!(cond)) {                                              \
            printf("RELEARN ERROR at %s:%d\n", __FILE__, __LINE__); \
            printf("    " fmt "\n", ##__VA_ARGS__);                 \
            __trap();                                               \
        }                                                           \
    } while (0)

#define kernelErrCheck()                                                                                                              \
    {                                                                                                                                 \
        if (const cudaError err = cudaGetLastError(); err != cudaSuccess) {                                                           \
            std::cerr << "Cuda Kernel Error in file " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            cudaDeviceReset();                                                                                                        \
            abort();                                                                                                                  \
        }                                                                                                                             \
    }

/** CUDA launch configuration: number of blocks (grid size) and threads per block (block size). */
struct KernelLaunchConfig {
    int grid_size;
    int block_size;
};

template <typename Kernel>
KernelLaunchConfig get_grid_ands_block_size(std::size_t problem_size, Kernel kernel) {
    int min_grid_size = 0;
    int block_size = 0;

    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        kernel,
        0,
        0);

    block_size = std::min(block_size, 1024);
    const auto grid_size_for_problem = (problem_size + block_size - 1) / block_size;
    const auto grid_size = std::max<int>(grid_size_for_problem, min_grid_size / 3);
    return { grid_size, block_size };
}
