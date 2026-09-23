/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_monitor_gpu.h"

namespace {

constexpr unsigned BLOCK = 256;

unsigned grid(unsigned n) {
    return (n + BLOCK - 1) / BLOCK;
}

__global__ void k_write_synaptic_elements(SynapticElementsBaseCudaHandle handle, float grown_offset, std::uint32_t connected_offset) {
    const auto nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= handle.number_neurons)
        return;

    handle.grown_elements[nid] = static_cast<CudaConfig::synaptic_grown_type>(2.0f * static_cast<float>(nid) + grown_offset);
    handle.connected_elements[nid] = static_cast<CudaConfig::synaptic_count_type>(nid + connected_offset);
}

} // namespace

void device_write_synaptic_elements(SynapticElementsBaseCudaHandle handle, float grown_offset, std::uint32_t connected_offset) {
    k_write_synaptic_elements<<<grid(handle.number_neurons), BLOCK>>>(handle, grown_offset, connected_offset);
    cudaDeviceSynchronize();
}
