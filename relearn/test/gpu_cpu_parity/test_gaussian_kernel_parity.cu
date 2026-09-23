/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_gaussian_kernel_parity.h"

#ifdef RELEARN_CUDA_ENABLED

#include "cuda/algorithm/Kernel.cuh"
#include "cuda/util/Util.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void k_calculate_attractiveness(
    double source_x, double source_y, double source_z,
    double target_x, double target_y, double target_z,
    double number_free_elements, double squared_sigma_inv,
    double* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const auto source = SimpleVec3d{ source_x, source_y, source_z };
    const auto target = SimpleVec3d{ target_x, target_y, target_z };
    *out = calculate_attractiveness_to_connect(source, target, number_free_elements, /*weighted_dendrites=*/false, squared_sigma_inv);
}

} // namespace

double device_calculate_attractiveness_to_connect(
    double source_x, double source_y, double source_z,
    double target_x, double target_y, double target_z,
    double number_free_elements, double squared_sigma_inv) {
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));

    k_calculate_attractiveness<<<1, 1>>>(source_x, source_y, source_z, target_x, target_y, target_z, number_free_elements, squared_sigma_inv, d_out);
    cudaDeviceSynchronize();
    kernelErrCheck();

    double h_out{};
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    return h_out;
}

#endif
