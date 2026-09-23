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

#include "GPUEdges.h"

#include "cuda/CudaConfig.h"
#include "cuda/memory/SharedBlockPool.h"

namespace utility {
class MemoryFootprint;
}

struct DummyView;
struct NetworkHandle;
enum class LayoutType;
struct Features;
class GPUEdgesBase;
class NeuronID;
class RankNeuronId;

struct NetworkGraphGPUParams {
    bool has_local_edges{};
    LayoutType layout_local_incoming{};
    Features features_local_incoming{};
    LayoutType layout_local_outgoing{};
    Features features_local_outgoing{};
    LayoutType layout_distant_incoming{};
    Features features_distant_incoming{};
    LayoutType layout_distant_outgoing{};
    Features features_distant_outgoing{};
    std::size_t expected_synapses_per_neuron{};
    std::size_t number_neurons{};
    double overflow_chunk_size_factor{ CudaConfig::overflow_chunk_size_factor };
    double number_overflow_chunks_factor{ CudaConfig::number_overflow_chunks_factor };
    double local_edges_ratio{ CudaConfig::local_edges_ratio };
    double non_overflow_size_factor{ CudaConfig::non_overflow_size_factor };
};

class NetworkGraphGPUBase {
public:
    using mpi_rank_type = std::uint16_t;
    using neuron_id_type = std::uint32_t;
    using weight_type = std::int16_t;

    NetworkGraphGPUBase(const mpi_rank_type _number_ranks, const mpi_rank_type _my_rank);
    virtual ~NetworkGraphGPUBase();

    NetworkGraphGPUBase(const NetworkGraphGPUBase&) = delete;
    NetworkGraphGPUBase& operator=(const NetworkGraphGPUBase&) = delete;
    NetworkGraphGPUBase(NetworkGraphGPUBase&&) = delete;
    NetworkGraphGPUBase& operator=(NetworkGraphGPUBase&&) = delete;

    void init(NetworkGraphGPUParams&& network_params);

    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) const;

    void record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) const;

    [[nodiscard]] virtual std::uint64_t get_gpu_memory_footprint() const {
        return (shared_overflow_pool != nullptr ? shared_overflow_pool->get_gpu_memory_footprint() : 0)
               + (incoming_local_edges != nullptr ? incoming_local_edges->get_gpu_memory_footprint() : 0)
               + incoming_distant_edges->get_gpu_memory_footprint()
               + (outgoing_local_edges != nullptr ? outgoing_local_edges->get_gpu_memory_footprint() : 0)
               + outgoing_distant_edges->get_gpu_memory_footprint();
    }

    void update_edges(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_edges_in_host, const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_edges_out_host, const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_in_edges_host, const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_out_edges_host);

    [[nodiscard]] NetworkHandle get_handle();

    void rebuild();

    // How many rebuild() calls to skip between bloom filter rebuilds.
    // Set to 0 to disable periodic rebuilding entirely.
    std::uint32_t bloom_rebuild_interval{ 50 };

    using LocalEdges = std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>;

    using NeuronLocalInNeighborhood = std::vector<LocalEdges>;
    using NeuronLocalOutNeighborhood = std::vector<LocalEdges>;

    using DistantEdges = std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>;
    using DistantCount = std::vector<std::int32_t>;

    using NeuronDistantInNeighborhood = std::vector<DistantEdges>;
    using NeuronDistantOutNeighborhood = std::vector<DistantEdges>;

    [[nodiscard]] std::tuple<NeuronLocalInNeighborhood, NeuronLocalOutNeighborhood, NeuronDistantInNeighborhood, NeuronDistantOutNeighborhood> copy_to_host() const;

protected:
    mpi_rank_type number_ranks{};
    mpi_rank_type my_rank{};
    neuron_id_type number_neurons{};
    NetworkGraphGPUParams network_params{};

    std::uint32_t bloom_rebuild_counter{};

    std::unique_ptr<SharedBlockPool> shared_overflow_pool;
    std::unique_ptr<GPUEdgesBase> incoming_local_edges;
    std::unique_ptr<GPUEdgesBase> incoming_distant_edges;
    std::unique_ptr<GPUEdgesBase> outgoing_local_edges;
    std::unique_ptr<GPUEdgesBase> outgoing_distant_edges;
    void* dummy_view{};
};
