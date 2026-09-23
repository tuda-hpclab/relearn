/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NetworkGraphGPU.h"

#include "GPUEdges.h"

#include "cuda/memory/SharedBlockPool.h"
#include "cuda/network_graph/NetworkHandle.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/Timers.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

void NetworkGraphGPUBase::init(NetworkGraphGPUParams&& _network_params) {
    network_params = std::move(_network_params);
    const auto overflow_chunk_size_factor = network_params.overflow_chunk_size_factor;
    const auto number_overflow_chunks_factor = network_params.number_overflow_chunks_factor;
    const auto max_edges = network_params.expected_synapses_per_neuron;
    const auto non_overflow_size_factor = network_params.non_overflow_size_factor;

    // Main-chunk sizes: local gets ratio * max, distant gets the remainder.
    auto local_ratio = std::clamp(network_params.local_edges_ratio, 0.0, 1.0);
    if (number_ranks == 1 && network_params.has_local_edges) {
        local_ratio = 1.0;
    } else if (!network_params.has_local_edges) {
        local_ratio = 0.0;
    }
    const auto local_init_size = std::max(std::size_t{ 1 }, static_cast<std::size_t>(non_overflow_size_factor * local_ratio * static_cast<double>(max_edges)));
    const auto distant_init_size = std::max(std::size_t{ 1 }, max_edges - local_init_size + 1);

    // Phase 1: construct GPUEdges objects so we know their feature sets.
    if (network_params.has_local_edges) {
        incoming_local_edges = std::make_unique<GPUEdgesBase>(network_params.layout_local_incoming, network_params.features_local_incoming, number_ranks, my_rank);
        outgoing_local_edges = std::make_unique<GPUEdgesBase>(network_params.layout_local_outgoing, network_params.features_local_outgoing, number_ranks, my_rank);
    }
    incoming_distant_edges = std::make_unique<GPUEdgesBase>(network_params.layout_distant_incoming, network_params.features_distant_incoming, number_ranks, my_rank);
    outgoing_distant_edges = std::make_unique<GPUEdgesBase>(network_params.layout_distant_outgoing, network_params.features_distant_outgoing, number_ranks, my_rank);

    // Phase 2: count total DynamicVecVec instances across all edge graphs, then allocate.
    // Each block stores an embedded Chunk<uint32_t> header followed by data elements.
    // uint32_t is the widest element type, so block_bytes is divisible by sizeof(T) for all T.
    // block_bytes must also be a multiple of alignof(Chunk<T>) (8, from the T*/Chunk<T>* members):
    // every block is reinterpreted as a Chunk<T>, so a block at index i sits at data + i*block_bytes,
    // which must stay 8-byte aligned for all i -- otherwise the pointer stores in get_new_chunk()
    // fault with a CUDA "misaligned address" error.
    const auto data_bytes = static_cast<std::size_t>(overflow_chunk_size_factor * static_cast<double>(max_edges)) * sizeof(std::uint32_t);
    const auto raw_block_bytes = sizeof(Chunk<std::uint32_t>) + std::max(std::size_t{ 4 }, data_bytes);
    constexpr auto block_alignment = alignof(Chunk<std::uint32_t>);
    const auto block_bytes = ((raw_block_bytes + block_alignment - 1) / block_alignment) * block_alignment;
    std::size_t number_buffers = 0;
    if (incoming_local_edges) {
        number_buffers += incoming_local_edges->count_pool_consumers();
    }
    if (outgoing_local_edges) {
        number_buffers += outgoing_local_edges->count_pool_consumers();
    }
    number_buffers += incoming_distant_edges->count_pool_consumers();
    number_buffers += outgoing_distant_edges->count_pool_consumers();
    const auto num_blocks = static_cast<std::size_t>(static_cast<double>(number_buffers * network_params.number_neurons) * number_overflow_chunks_factor);
    shared_overflow_pool = std::make_unique<SharedBlockPool>(num_blocks, block_bytes);
    SharedBlockPool* const pool = shared_overflow_pool.get();

    // Phase 3: initialise all edge graphs with the shared pool.
    if (network_params.has_local_edges) {
        incoming_local_edges->init(static_cast<GPUEdgesBase::neuron_id_type>(network_params.number_neurons), local_init_size, pool);
        outgoing_local_edges->init(static_cast<GPUEdgesBase::neuron_id_type>(network_params.number_neurons), local_init_size, pool);
    }
    incoming_distant_edges->init(static_cast<GPUEdgesBase::neuron_id_type>(network_params.number_neurons), distant_init_size, pool);
    outgoing_distant_edges->init(static_cast<GPUEdgesBase::neuron_id_type>(network_params.number_neurons), distant_init_size, pool);
    number_neurons = static_cast<neuron_id_type>(network_params.number_neurons);
}

void NetworkGraphGPUBase::record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) const {
    if (shared_overflow_pool != nullptr) {
        shared_overflow_pool->record_usage_footprint(footprint, "shared overflow pool");
    }
    if (incoming_local_edges != nullptr) {
        incoming_local_edges->record_usage_footprint(footprint, "GPU incoming local edges");
    }
    incoming_distant_edges->record_usage_footprint(footprint, "GPU incoming distant edges");
    if (outgoing_local_edges != nullptr) {
        outgoing_local_edges->record_usage_footprint(footprint, "GPU outgoing local edges");
    }
    outgoing_distant_edges->record_usage_footprint(footprint, "GPU outgoing distant edges");
}

void NetworkGraphGPUBase::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) const {
    if (shared_overflow_pool != nullptr) {
        shared_overflow_pool->record_memory_footprint(footprint, "GPU shared overflow pool");
    }
    if (incoming_local_edges != nullptr) {
        incoming_local_edges->record_memory_footprint(footprint, "GPU incoming local edges");
    }
    incoming_distant_edges->record_memory_footprint(footprint, "GPU incoming distant edges");
    if (outgoing_local_edges != nullptr) {
        outgoing_local_edges->record_memory_footprint(footprint, "GPU outgoing local edges");
    }
    outgoing_distant_edges->record_memory_footprint(footprint, "GPU outgoing distant edges");
}

void NetworkGraphGPUBase::rebuild() {
    outgoing_distant_edges->rebuild();
    incoming_distant_edges->rebuild();
    // check_only_outgoing_view(   get_handle(), number_neurons);
    if (incoming_local_edges != nullptr) {
        incoming_local_edges->rebuild();
    }
    if (outgoing_local_edges != nullptr) {
        outgoing_local_edges->rebuild();
    }
}

std::tuple<NetworkGraphGPUBase::NeuronLocalInNeighborhood, NetworkGraphGPUBase::NeuronLocalOutNeighborhood,
           NetworkGraphGPUBase::NeuronDistantInNeighborhood, NetworkGraphGPUBase::NeuronDistantOutNeighborhood>
NetworkGraphGPUBase::copy_to_host() const {
    NeuronLocalInNeighborhood local_in_edges(number_neurons);
    NeuronLocalOutNeighborhood local_out_edges(number_neurons);
    NeuronDistantInNeighborhood distant_in_edges(number_neurons);
    NeuronDistantOutNeighborhood distant_out_edges(number_neurons);

    auto add_local_edges = [this](NeuronLocalOutNeighborhood& out_edges, NeuronLocalInNeighborhood& in_edges, const std::unique_ptr<GPUEdgesBase>& gpu_edges) {
        const auto data = gpu_edges->copy_to_host();
        for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
            std::unordered_map<NeuronID::value_type, RelearnTypes::plastic_synapse_weight> map;
            for (const auto& [rank, other_neuron_id, weight] : data[neuron_id]) {
                map[other_neuron_id] += weight;
            }

            for (const auto& [other_neuron_id, weight] : map) {
                out_edges[neuron_id].emplace_back(NeuronID{ other_neuron_id }, weight);
                in_edges[other_neuron_id].emplace_back(NeuronID{ neuron_id }, weight);
            }
        }
    };

    if (outgoing_local_edges != nullptr) {
        add_local_edges(local_out_edges, local_in_edges, outgoing_local_edges);
    }

    auto add_distant_edges = [this](NeuronLocalInNeighborhood& local_edges, NeuronDistantInNeighborhood& distant_edges, const std::unique_ptr<GPUEdgesBase>& gpu_edges) {
        const auto data = gpu_edges->copy_to_host();
        for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
            std::unordered_map<RankNeuronId, RelearnTypes::plastic_synapse_weight> map;
            for (const auto& [rank, other_neuron_id, weight] : data[neuron_id]) {
                map[RankNeuronId{ mpiPP::MPIRank{ rank }, NeuronID{ neuron_id } }] += weight;
            }

            for (const auto& [other_rni, weight] : map) {
                if (other_rni.get_rank().get_rank() == my_rank) {
                    local_edges[neuron_id].emplace_back(other_rni.get_neuron_id(), weight);
                } else {
                    distant_edges[neuron_id].emplace_back(other_rni, weight);
                }
            }
        }
    };

    add_distant_edges(local_in_edges, distant_in_edges, incoming_distant_edges);
    add_distant_edges(local_out_edges, distant_out_edges, outgoing_distant_edges);

    return { local_in_edges, local_out_edges, distant_in_edges, distant_out_edges };
}

// [[nodiscard]] std::uint64_t NetworkGraphGPUBase::get_gpu_memory_footprint() const {
//     return incoming_edges->get_gpu_memory_footprint() + outgoing_edges->get_gpu_memory_footprint();
// }

void NetworkGraphGPUBase::update_edges(
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_edges_in_host,
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_edges_out_host,
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>&
        distant_in_edges_host,
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>&
        distant_out_edges_host) {
    if (incoming_local_edges == nullptr) {
        incoming_distant_edges->update_edges(local_edges_in_host, distant_in_edges_host);
    } else {
        const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> empty_distant(distant_in_edges_host.size());
        const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> empty_local(distant_in_edges_host.size());
        incoming_local_edges->update_edges(local_edges_in_host, empty_distant);
        incoming_distant_edges->update_edges(empty_local, distant_in_edges_host);
    }
    if (outgoing_local_edges == nullptr) {
        outgoing_distant_edges->update_edges(local_edges_out_host, distant_out_edges_host);
    } else {
        const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> empty_local(distant_in_edges_host.size());
        const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> empty_distant(distant_in_edges_host.size());
        outgoing_local_edges->update_edges(local_edges_out_host, empty_distant);
        outgoing_distant_edges->update_edges(empty_local, distant_out_edges_host);
    }
}

NetworkHandle NetworkGraphGPUBase::get_handle() {
    auto inc_l = incoming_local_edges != nullptr ? incoming_local_edges->get_handle() : EdgeHandle{ LayoutType::Dummy, dummy_view };
    auto inc_d = incoming_distant_edges->get_handle();
    auto outg_l = outgoing_local_edges != nullptr ? outgoing_local_edges->get_handle() : EdgeHandle{ LayoutType::Dummy, dummy_view };
    auto outg_d = outgoing_distant_edges->get_handle();
    return NetworkHandle{ inc_l,
                          inc_d,
                          outg_l,
                          outg_d,
                          shared_overflow_pool.get() };
}
