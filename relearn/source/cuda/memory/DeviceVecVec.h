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

#include "SharedBlockPool.h"

#include "cpp-utility/MemoryFootprint.hpp"
#include "cuda/CudaBaseBridgeFunctions.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

template <typename T>
class DynamicVecVecView;

/**
 * Aggregate per-neuron chunk-usage statistics returned by DynamicVecVec<T>::get_used_chunks_per_neuron().
 */
struct ChunkUsageStats {
    std::size_t sum_used_chunks;     ///< Sum of used (main) chunks across all neurons.
    float avg_used_chunks;           ///< Average number of used chunks per neuron.
    std::size_t median_used_chunks;  ///< Median number of used chunks per neuron.
    std::size_t max_used_chunks;     ///< Maximum number of used chunks for any single neuron.
    float ratio_multi_chunk_neurons; ///< Fraction of neurons using more than one chunk.
};

/**
 * Same shape as ChunkUsageStats, but for the DynamicVecVec<bool> specialization, whose last field
 * is the raw count of multi-chunk neurons rather than a ratio (see get_used_chunks_per_neuron()).
 */
struct BoolChunkUsageStats {
    std::size_t sum_used_chunks;
    float avg_used_chunks;
    std::size_t median_used_chunks;
    std::size_t max_used_chunks;
    std::size_t count_multi_chunk_neurons; ///< Number of neurons using more than one chunk.
};

/**
 * A linked-list chunk for a single neuron's adjacency list in DynamicVecVec.
 *
 * Chunks are allocated from a shared pool; each neuron holds a singly-linked
 * list of Chunk objects growing on demand as elements are inserted on the device.
 */
template <typename T>
struct Chunk {
    T* begin{};               ///< Pointer to the first element in this chunk's data region.
    std::uint32_t chunk_id{}; ///< Index of this chunk in the global chunk array.
    std::uint32_t filled{};   ///< Number of elements currently stored in this chunk.
    std::uint32_t size{};     ///< Total capacity of this chunk in elements.
    Chunk<T>* prev{};         ///< Previous chunk in the per-neuron linked list (nullptr if first).
    Chunk<T>* next{};         ///< Next chunk in the per-neuron linked list (nullptr if last).
};

/**
 * GPU-resident dynamic vector-of-vectors for type T, grown on the device without host involvement.
 *
 * Memory is organised in fixed-size chunks; when a neuron's active chunk is full, the kernel
 * allocates a new chunk from a shared pool.  Overflow chunks are returned to the pool when no
 * longer needed.
 */
template <typename T>
class DynamicVecVec {
public:
    /**
     * @brief Constructs with a private pool; derives all sizes from CudaConfig.
     * @param number_elements Number of neurons (outer dimension).
     */
    explicit DynamicVecVec(std::size_t number_elements);

    /**
     * @brief Constructs with a shared overflow pool.
     * @param number_elements Number of neurons (outer dimension).
     * @param init_size       Initial capacity per neuron in elements (main chunk size).
     * @param shared_pool     Shared pool from which overflow chunks are drawn.
     */
    DynamicVecVec(std::size_t number_elements, std::size_t init_size, SharedBlockPool* shared_pool);

    DynamicVecVec(const DynamicVecVec&) = delete;
    DynamicVecVec& operator=(const DynamicVecVec&) = delete;
    DynamicVecVec(DynamicVecVec&&) = delete;
    DynamicVecVec& operator=(DynamicVecVec&&) = delete;

    ~DynamicVecVec() {
        cudaFree_bridge(data);
        cudaFree_bridge(chunks);
        cudaFree_bridge(device_class);
    }

    /**
     * @brief Returns a device pointer to the DynamicVecVecView that kernels use to access and modify the structure.
     */
    [[nodiscard]] DynamicVecVecView<T>* get_device_view() const { return device_class; }

    /**
     * @brief Initialises the device structure from a host-side map of neuron-ID → element lists.
     * @param map Host-side adjacency map; each entry populates one neuron's inner vector.
     */
    void add_from_map(const std::unordered_map<std::uint32_t, std::vector<T>>& map);

    /**
     * @brief Records per-field memory usage into @p footprint with the given name prefix.
     * @param footprint Target memory footprint tracker.
     * @param prefix    String prefix for field names.
     */
    void record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const;

    /**
     * @brief Returns total device memory in bytes occupied by data and metadata arrays.
     */
    [[nodiscard]] std::uint64_t get_gpu_memory_footprint() const;

    /**
     * @brief Returns the average fill fraction of main (non-overflow) chunks across all neurons.
     */
    [[nodiscard]] float get_average_filled_main_chunk() const;

    /**
     * @brief Returns the average number of used chunks per neuron (including overflow).
     */
    [[nodiscard]] ChunkUsageStats get_used_chunks_per_neuron() const;

    /**
     * @brief Records byte-level memory usage into @p footprint with the given name prefix.
     * @param footprint Target memory footprint tracker.
     * @param prefix    String prefix for field names.
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const {
        footprint->emplace(prefix + " total data", total_size * sizeof(T));
        footprint->emplace(prefix + " chunk storage", main_chunks * sizeof(Chunk<T>));
    }

    /**
     * @brief Copies the entire structure from device to a host vector-of-vectors.
     */
    [[nodiscard]] std::vector<std::vector<T>> copy_to_host() const;

private:
    void init_main(std::size_t number_elements, std::size_t init_size, SharedBlockPool* pool);

    T* data{};
    Chunk<T>* chunks{};
    DynamicVecVecView<T>* device_class{};
    std::size_t total_size{};
    std::size_t main_chunks{};
    std::unique_ptr<SharedBlockPool> private_pool;
};

/**
 * Specialisation of DynamicVecVec for bool: bits are packed into 32-bit words to reduce memory.
 */
template <>
class DynamicVecVec<bool> {
public:
    /**
     * @brief Constructs with a private pool; derives all sizes from CudaConfig.
     * @param number_elements Number of neurons (outer dimension).
     */
    explicit DynamicVecVec(std::size_t number_elements);

    /**
     * @brief Constructs with a shared overflow pool.
     * @param number_elements Number of neurons (outer dimension).
     * @param init_size       Initial capacity per neuron in bits.
     * @param shared_pool     Shared pool from which overflow chunks are drawn.
     */
    DynamicVecVec(std::size_t number_elements, std::size_t init_size, SharedBlockPool* shared_pool);

    DynamicVecVec(const DynamicVecVec&) = delete;
    DynamicVecVec& operator=(const DynamicVecVec&) = delete;
    DynamicVecVec(DynamicVecVec&&) = delete;
    DynamicVecVec& operator=(DynamicVecVec&&) = delete;

    ~DynamicVecVec() {
        cudaFree_bridge(data);
        cudaFree_bridge(chunks);
        cudaFree_bridge(device_class);
    }

    /**
     * @brief Returns a device pointer to the DynamicVecVecView that kernels use to access the bit-packed structure.
     */
    [[nodiscard]] DynamicVecVecView<bool>* get_device_view() const { return device_class; }

    /**
     * @brief Initialises the device structure from a host-side map of neuron-ID → bool lists.
     * @param map Host-side adjacency map.
     */
    void add_from_map(const std::unordered_map<std::uint32_t, std::vector<bool>>& map);

    /**
     * @brief Returns total device memory in bytes occupied by data and metadata arrays.
     */
    [[nodiscard]] std::uint64_t get_gpu_memory_footprint() const;

    /**
     * @brief Returns the average fill fraction of main chunks across all neurons.
     */
    [[nodiscard]] float get_average_filled_main_chunk() const;

    /**
     * @brief Returns the average number of used chunks per neuron (including overflow).
     */
    [[nodiscard]] BoolChunkUsageStats get_used_chunks_per_neuron() const;

    /**
     * @brief Records byte-level memory usage into @p footprint with the given name prefix.
     * @param footprint Target memory footprint tracker.
     * @param prefix    String prefix for field names.
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const {
        footprint->emplace(prefix + " total data", total_words * sizeof(std::uint32_t));
        footprint->emplace(prefix + " chunk storage", main_chunks * sizeof(Chunk<std::uint32_t>));
    }

    /**
     * @brief Records per-field usage statistics into @p footprint with the given name prefix.
     * @param footprint Target memory footprint tracker.
     * @param prefix    String prefix for field names.
     */
    void record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const;

    /**
     * @brief Copies the entire bit-packed structure back to a host vector-of-vectors of bool.
     */
    [[nodiscard]] std::vector<std::vector<bool>> copy_to_host() const;

private:
    void init_main(std::size_t number_elements, std::size_t init_size, SharedBlockPool* pool);

    std::uint32_t* data{};
    Chunk<std::uint32_t>* chunks{};
    DynamicVecVecView<bool>* device_class{};
    std::size_t total_words{};
    std::size_t main_chunks{};
    std::unique_ptr<SharedBlockPool> private_pool;
};
