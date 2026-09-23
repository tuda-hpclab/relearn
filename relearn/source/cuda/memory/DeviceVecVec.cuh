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

#include "DeviceVecVec.h"
#include "SharedBlockPool.cuh"

#include "util/Util.cuh"

#include <cuda.h>

template <typename T>
struct DeviceVecVecCursor {
    using index_type = uint32_t;
    Chunk<T>* current_chunk{};
    index_type current_local_offset{ 0U };

    __device__ bool operator==(const DeviceVecVecCursor& other) const {
        return current_chunk == nullptr && nullptr == other.current_chunk || current_chunk == other.current_chunk && current_local_offset == other.current_local_offset;
    }

    __device__ T next() {
        RELEARN_DEVICE_CUDA_CHECK(current_chunk != nullptr, "DeviceVecVecCursor::next: Iterator out of bounds");
        const auto item = current_chunk->begin[current_local_offset];
        ++current_local_offset;
        if (current_local_offset >= current_chunk->filled) {
            current_chunk = current_chunk->next;
            current_local_offset = 0U;
            // Skip trailing empty chunks left over after remove_edge_at + pop_back
            while (current_chunk != nullptr && current_chunk->filled == 0) {
                current_chunk = current_chunk->next;
            }
        }
        return item;
    }

    __device__ explicit DeviceVecVecCursor(Chunk<T>* current_chunk)
        : current_chunk(current_chunk) { }

    __device__ DeviceVecVecCursor() = default;
};

template <>
struct DeviceVecVecCursor<bool> {
    using index_type = uint32_t;

    Chunk<uint32_t>* current_chunk{};
    index_type current_local_offset{ 0U };

    __device__ bool operator==(const DeviceVecVecCursor<bool>& other) const {
        return current_chunk == nullptr && nullptr == other.current_chunk || current_chunk == other.current_chunk && current_local_offset == other.current_local_offset;
    }

    __device__ bool next() {
        RELEARN_DEVICE_CUDA_CHECK(current_chunk != nullptr, "DeviceVecVecCursor::next: Iterator out of bounds");
        const index_type word_idx = current_local_offset >> 5U;
        const index_type bit_pos = current_local_offset & 31U;
        const bool item = (current_chunk->begin[word_idx] >> bit_pos) & 1U;
        ++current_local_offset;
        if (current_local_offset >= current_chunk->filled) {
            current_chunk = current_chunk->next;
            current_local_offset = 0U;
            while (current_chunk != nullptr && current_chunk->filled == 0) {
                current_chunk = current_chunk->next;
            }
        }
        return item;
    }

    __device__ explicit DeviceVecVecCursor(Chunk<uint32_t>* c)
        : current_chunk(c) { }
    __device__ DeviceVecVecCursor() = default;
};

template <typename T>
class DynamicVecVecView {
public:
    using index_type = std::uint32_t;

    DynamicVecVecView() = default;

    __device__ [[nodiscard]] bool add(index_type neuron_id, T&& value);

    __device__ T get(index_type neuron_id, index_type element_idx) const;

    __device__ void set(index_type neuron_id, index_type element_idx, T value);

    __device__ void pop_back(index_type neuron_id);

    __device__ index_type get_size(index_type neuron_id) const;

    __device__ DeviceVecVecCursor<T> begin(index_type neuron_id) {
        auto* chunk = chunks + neuron_id;
        while (chunk != nullptr && chunk->filled == 0) {
            chunk = chunk->next;
        }
        if (chunk == nullptr)
            return DeviceVecVecCursor<T>{};
        return DeviceVecVecCursor<T>(chunk);
    }

    __device__ DeviceVecVecCursor<T> end([[maybe_unused]] index_type neuron_id) {
        return DeviceVecVecCursor<T>();
    }

    __device__ bool empty() const { return false; }

public:
    Chunk<T>* chunks{}; // main chunk descriptors [0..main_chunks-1]
    DeviceSharedBlockPool* pool{};
    index_type new_chunk_size{}; // data elements per overflow block
    index_type main_chunks{};

    __device__ Chunk<T>* get_new_chunk(Chunk<T>* prev);
};

template <>
class DynamicVecVecView<bool> {
public:
    using index_type = std::uint32_t;

    DynamicVecVecView() = default;

    __device__ [[nodiscard]] bool add(index_type neuron_id, bool value);
    __device__ bool get(index_type neuron_id, index_type bit_idx) const;
    __device__ void set(index_type neuron_id, index_type bit_idx, bool value);
    __device__ void pop_back(index_type neuron_id);
    __device__ index_type get_size(index_type neuron_id) const;

    __device__ DeviceVecVecCursor<bool> begin(index_type neuron_id) {
        RELEARN_DEVICE_CUDA_CHECK(neuron_id < main_chunks, "DynamicVecVecView::begin: Neuron id too large %u < %u", neuron_id, main_chunks);
        auto* chunk = chunks + neuron_id;
        while (chunk != nullptr && chunk->filled == 0) {
            chunk = chunk->next;
        }
        if (chunk == nullptr)
            return DeviceVecVecCursor<bool>{};
        return DeviceVecVecCursor<bool>(chunk);
    }

    __device__ DeviceVecVecCursor<bool> end([[maybe_unused]] index_type neuron_id) {
        return DeviceVecVecCursor<bool>();
    }

    __device__ bool empty() const { return false; }

    Chunk<std::uint32_t>* chunks{};
    DeviceSharedBlockPool* pool{};
    index_type new_chunk_size{}; // in bits for bool specialisation
    index_type main_chunks{};

    __device__ Chunk<std::uint32_t>* get_new_chunk(Chunk<std::uint32_t>* prev);
};
