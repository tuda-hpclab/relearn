#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#define RELEARN_CUDA_FOUND CUDA_FOUND

// The floating point precision in which the simulation calculates, set by PRECISION in CMake:
// true is FP32, i.e., float, false is FP64, i.e., double. cmake/Precision.cmake defines it, and
// types/BasicTypes.h and cuda/CudaTypes.h pick the type of their aliases based on it.
#define RELEARN_PRECISION_FP32 PRECISION_FP32

#ifdef __CUDACC__
// File is compiled with CUDA compiler
#define CUDA_COMPILER
#else
// File is compiled wiith host compiler, e.g., g++
#define HOST_COMPILER
#endif

// Macro that enables execution of a function on gpu and device
#if __CUDACC__
#define GPU_AND_HOST __device__ __host__
#else
#define GPU_AND_HOST
#endif
