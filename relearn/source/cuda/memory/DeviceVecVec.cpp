/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "DeviceVecVec.h"

#include "cpp-utility/MemoryFootprint.hpp"
#include "cuda/memory/SharedBlockPool.h"
#include "cuda/util/SmallNeuronIdType.h"
#include "cuda/util/Util.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// DynamicVecVec<T> is only implemented in DeviceVecVec.cu (compiled into relearn_gpu, which is
// only built when RELEARN_CUDA_ENABLED). This file is always compiled, so on a CPU-only build
// these stubs are the only definitions available -- matching the NOT_SUPPORTED convention used
// for the CUDA bridge functions in CudaBridgeFunctions.cpp.
#ifndef RELEARN_CUDA_ENABLED

template <typename T>
DynamicVecVec<T>::DynamicVecVec(std::size_t) { CUDA_NOT_SUPPORTED }

template <typename T>
DynamicVecVec<T>::DynamicVecVec(std::size_t, std::size_t, SharedBlockPool*) { CUDA_NOT_SUPPORTED }

template <typename T>
void DynamicVecVec<T>::add_from_map(const std::unordered_map<std::uint32_t, std::vector<T>>&) { CUDA_NOT_SUPPORTED }

template <typename T>
std::uint64_t DynamicVecVec<T>::get_gpu_memory_footprint() const { CUDA_NOT_SUPPORTED }

template <typename T>
float DynamicVecVec<T>::get_average_filled_main_chunk() const { CUDA_NOT_SUPPORTED }

template <typename T>
ChunkUsageStats DynamicVecVec<T>::get_used_chunks_per_neuron() const { CUDA_NOT_SUPPORTED }

template <typename T>
std::vector<std::vector<T>> DynamicVecVec<T>::copy_to_host() const { CUDA_NOT_SUPPORTED }

template <typename T>
void DynamicVecVec<T>::record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>&,
                                              const std::string&) const { CUDA_NOT_SUPPORTED }

// Explicit instantiations for every T that GPUEdgesBase's GraphStorage implementations
// (MemoryPoolStorage / MemoryPoolNeuronIdStorage in cuda/network_graph/GPUEdges.h) actually use.
template class DynamicVecVec<std::uint32_t>;
template class DynamicVecVec<std::uint16_t>;
template class DynamicVecVec<std::int8_t>;
template class DynamicVecVec<SmallNeuronIdType>;

// bool specialisation (non-template, so its members are stubbed directly rather than via the
// template above).
DynamicVecVec<bool>::DynamicVecVec(std::size_t){ CUDA_NOT_SUPPORTED }

DynamicVecVec<bool>::DynamicVecVec(std::size_t, std::size_t, SharedBlockPool*) { CUDA_NOT_SUPPORTED }

void DynamicVecVec<bool>::add_from_map(const std::unordered_map<std::uint32_t, std::vector<bool>>&){ CUDA_NOT_SUPPORTED }

std::uint64_t DynamicVecVec<bool>::get_gpu_memory_footprint() const { CUDA_NOT_SUPPORTED }

float DynamicVecVec<bool>::get_average_filled_main_chunk() const { CUDA_NOT_SUPPORTED }

BoolChunkUsageStats DynamicVecVec<bool>::get_used_chunks_per_neuron() const { CUDA_NOT_SUPPORTED }

std::vector<std::vector<bool>> DynamicVecVec<bool>::copy_to_host() const { CUDA_NOT_SUPPORTED }

void DynamicVecVec<bool>::record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>&,
                                                 const std::string&) const { CUDA_NOT_SUPPORTED }

#endif
