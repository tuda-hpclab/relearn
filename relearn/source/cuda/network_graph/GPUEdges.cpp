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

#include "cpp-utility/MemoryFootprint.hpp"
#include "cuda/memory/SharedBlockPool.h"
#include "cuda/network_graph/NetworkHandle.h"
#include "cuda/util/Util.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/Timers.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

void GPUEdgesBase::init(const neuron_id_type _number_neurons,
                        const std::size_t _expected_synapses_per_neuron,
                        SharedBlockPool* const shared_pool) {
    number_neurons = _number_neurons;
    expected_synapses_per_neuron = _expected_synapses_per_neuron;

    if (other_neurons_storage != nullptr) {
        other_neurons_storage->init(_number_neurons, _expected_synapses_per_neuron, shared_pool);
    }
    if (other_ranks.enabled) {
        other_ranks.other_ranks_storage->init(_number_neurons, _expected_synapses_per_neuron, shared_pool);
    }
    if (weighted.enabled) {
        weighted.weight_storage->init(_number_neurons, _expected_synapses_per_neuron, shared_pool);
    }
    if (excitatory.enabled) {
        excitatory.excitatory_storage->init(_number_neurons, _expected_synapses_per_neuron, shared_pool);
    }

    if (incoming_count_data.enabled) {
        incoming_count_data.count.resize(_number_neurons, 0U);
        incoming_count_data.count_excitatory.resize(_number_neurons, 0U);
    }

    _init = true;
}

std::size_t GPUEdgesBase::count_pool_consumers() const {
    std::size_t n = 0;
    if (other_neurons_storage && other_neurons_storage->uses_shared_pool()) {
        n++;
    }
    if (other_ranks.enabled && other_ranks.other_ranks_storage && other_ranks.other_ranks_storage->uses_shared_pool()) {
        n++;
    }
    if (weighted.enabled && weighted.weight_storage && weighted.weight_storage->uses_shared_pool()) {
        n++;
    }
    if (excitatory.enabled && excitatory.excitatory_storage && excitatory.excitatory_storage->uses_shared_pool()) {
        n++;
    }
    return n;
}

void GPUEdgesBase::rebuild() {
    RelearnException::check(_init, "Not initialize r");
}

std::vector<std::vector<std::tuple<GPUEdgesBase::mpi_rank_type, GPUEdgesBase::neuron_id_type, GPUEdgesBase::weight_type>>> GPUEdgesBase::copy_to_host() const {
    auto other_neurons_data = other_neurons_storage->copy_to_cpu();

    std::optional<std::vector<std::vector<internal_weight_type>>> data_weight{};
    std::optional<std::vector<std::vector<bool>>> data_exc{};
    if (weighted.enabled) {
        data_weight = weighted.weight_storage->copy_to_cpu();
    } else {
        data_exc = excitatory.excitatory_storage->copy_to_cpu();
    }

    std::optional<std::vector<std::vector<mpi_rank_type>>> data_ranks{};
    if (other_ranks.enabled) {
        data_ranks = other_ranks.other_ranks_storage->copy_to_cpu();
    }

    std::vector<std::vector<std::tuple<mpi_rank_type, neuron_id_type, weight_type>>> result(number_neurons);

    // Process per-neuron and free source vectors immediately to keep peak RAM
    // at max(tuple_result_size) rather than source_size + tuple_result_size.
    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        auto& data_for_neuron = other_neurons_data[neuron_id];
        std::vector<std::tuple<mpi_rank_type, neuron_id_type, weight_type>> local_result;
        local_result.reserve(data_for_neuron.size());
        for (auto i = 0UL; i < data_for_neuron.size(); i++) {
            const auto other_neuron_id = data_for_neuron[i];
            auto weight = 0;
            if (weighted.enabled) {
                weight = static_cast<int>(data_weight.value()[neuron_id][i]);
            } else {
                weight = data_exc.value()[neuron_id][i] ? 1 : -1;
            }
            auto rank = my_rank;
            if (other_ranks.enabled) {
                rank = data_ranks.value()[neuron_id][i];
            }
            local_result.emplace_back(rank, other_neuron_id, weight);
        }
        result[neuron_id] = std::move(local_result);

        // Free source data immediately after this neuron is processed
        data_for_neuron = {};
        if (data_weight) {
            (*data_weight)[neuron_id] = {};
        }
        if (data_exc) {
            (*data_exc)[neuron_id] = {};
        }
        if (data_ranks) {
            (*data_ranks)[neuron_id] = {};
        }
    }
    return result;
}

void GPUEdgesBase::update_edges(
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_edges,
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_edges) {

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& host_local_edges = local_edges[neuron_id];
        const auto& host_distant_edges = distant_edges[neuron_id];
        const auto size = host_local_edges.size() + host_distant_edges.size();
        RelearnException::check(size <= expected_synapses_per_neuron, "Too many edges: neuron {} has {} edges, but only {} were expected (see --expected-synapses-per-neuron)", neuron_id, size, expected_synapses_per_neuron);

        for (auto [other_neuron_id, weight] : host_local_edges) {
            RelearnException::check(weight != 0, "NetworkGraphBase::update_d_in_edges: weight is 0");
            RelearnException::check(other_neuron_id.get_neuron_id() != neuron_id,
                                    "NetworkGraphBase::update_d_in_edges: source and target are equal {} {}", neuron_id, other_neuron_id.get_neuron_id());

            const auto max_weight = std::numeric_limits<internal_weight_type>::max();
            const auto min_weight = std::numeric_limits<internal_weight_type>::min();
            auto excitatory_edge = weight > 0;
            auto it = 0UL;
            if (weighted.enabled) {
                if (weight > 0) {
                    it = static_cast<std::size_t>(std::ceil(weight / static_cast<double>(std::numeric_limits<internal_weight_type>::max())));
                } else {
                    it = static_cast<std::size_t>(std::ceil(weight / static_cast<double>(std::numeric_limits<internal_weight_type>::min())));
                }
            } else {
                it = static_cast<std::size_t>(std::abs(weight));
            }
            auto weight_left = weight;
            for (auto j = 0U; j < it; j++) {

                if (other_ranks.enabled) {
                    other_ranks.other_ranks_storage->add(static_cast<neuron_id_type>(neuron_id), my_rank);
                }
                other_neurons_storage->add(static_cast<neuron_id_type>(neuron_id), static_cast<uint32_t>(other_neuron_id.get_neuron_id()));
                if (excitatory.enabled) {
                    excitatory.excitatory_storage->add(static_cast<neuron_id_type>(neuron_id), excitatory_edge);
                }
                if (weighted.enabled) {
                    auto w = static_cast<internal_weight_type>(weight_left);
                    if (weight_left > 0) {
                        if (weight_left > max_weight) {
                            w = max_weight;
                            weight_left -= max_weight;
                        }
                    } else {
                        RelearnException::check(weight_left != 0, "Weight is 0");
                        if (weight_left < min_weight) {
                            w = min_weight;
                            weight_left -= min_weight;
                        }
                    }
                    weighted.weight_storage->add(static_cast<neuron_id_type>(neuron_id), w);
                }
            }
        }

        [[maybe_unused]] const auto offset = host_local_edges.size();
        for (auto [other_rni, weight] : host_distant_edges) {
            const auto& [other_rank, other_neuron_id] = other_rni;
            RelearnException::check(weight != 0, "NetworkGraphBase::update_d_in_edges: weight is 0");
            RelearnException::check(static_cast<mpi_rank_type>(other_rank.get_rank()) != my_rank,
                                    "NetworkGraphBase::update_d_in_edges: Distant connection that is local?");

            auto excitatory_edge = weight > 0;
            const auto it = static_cast<std::size_t>(weighted.enabled ? 1 : std::abs(weight));
            for (auto j = 0U; j < it; j++) {
                if (other_ranks.enabled) {
                    other_ranks.other_ranks_storage->add(static_cast<neuron_id_type>(neuron_id), static_cast<mpi_rank_type>(other_rank.get_rank()));
                }
                other_neurons_storage->add(static_cast<neuron_id_type>(neuron_id), static_cast<uint32_t>(other_neuron_id.get_neuron_id()));
                if (excitatory.enabled) {
                    excitatory.excitatory_storage->add(static_cast<neuron_id_type>(neuron_id), excitatory_edge);
                }
                if (weighted.enabled) {
                    weighted.weight_storage->add(static_cast<neuron_id_type>(neuron_id), static_cast<internal_weight_type>(weight));
                }
            }
        }
    }

    if (other_ranks.enabled) {
        other_ranks.other_ranks_storage->force_update();
    }
    other_neurons_storage->force_update();
    if (excitatory.enabled) {
        excitatory.excitatory_storage->force_update();
    }
    if (weighted.enabled) {
        weighted.weight_storage->force_update();
    }

    if (layout == LayoutType::MemoryPool && incoming_count_data.enabled) {
        for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
            std::uint32_t total = 0;
            std::uint32_t exc = 0;
            for (const auto& [nid, weight] : local_edges[neuron_id]) {
                const auto w = static_cast<std::uint32_t>(std::abs(weight));
                total += w;
                if (weight > 0) {
                    exc += w;
                }
            }
            for (const auto& [rni, weight] : distant_edges[neuron_id]) {
                const auto w = static_cast<std::uint32_t>(std::abs(weight));
                total += w;
                if (weight > 0) {
                    exc += w;
                }
            }
            incoming_count_data.count[neuron_id] = total;
            incoming_count_data.count_excitatory[neuron_id] = exc;
        }
#ifdef RELEARN_CUDA_ENABLED
        incoming_count_data.count.force_update();
        incoming_count_data.count_excitatory.force_update();
#endif
    }
}

void GPUEdgesBase::record_memory_footprint([[maybe_unused]] const std::unique_ptr<utility::MemoryFootprint>& footprint, [[maybe_unused]] const std::string& prefix) {
#ifdef RELEARN_CUDA_ENABLED
    if (other_neurons_storage != nullptr) {
        other_neurons_storage->record_memory_footprint(footprint, prefix + " other neurons");
    }
    if (other_ranks.enabled) {
        other_ranks.other_ranks_storage->record_memory_footprint(footprint, prefix + " other ranks");
    }
    if (weighted.enabled) {
        weighted.weight_storage->record_memory_footprint(footprint, prefix + " weights");
    }
    if (incoming_count_data.enabled) {
        footprint->emplace(prefix + " count data", incoming_count_data.count.get_memory_footprint() + incoming_count_data.count_excitatory.get_memory_footprint());
    }

    if (other_neurons_storage != nullptr) {
        other_neurons_storage->record_usage_footprint(footprint, prefix + " neurons");
    }
#endif
}

void GPUEdgesBase::record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint,
                                          const std::string& prefix) const {
    if (other_neurons_storage != nullptr) {
        other_neurons_storage->record_usage_footprint(footprint, prefix);
    }
}

// The following members are only implemented in GPUEdges.cu (compiled into relearn_gpu, which is
// only built when RELEARN_CUDA_ENABLED). GPUEdges.cpp itself is always compiled, so on a CPU-only
// build these stubs are the only definitions available -- matching the NOT_SUPPORTED convention
// used for the CUDA bridge functions in CudaBridgeFunctions.cpp.
#ifndef RELEARN_CUDA_ENABLED

GPUEdgesBase::~GPUEdgesBase() = default;

std::uint64_t GPUEdgesBase::get_gpu_memory_footprint() const { CUDA_NOT_SUPPORTED }

EdgeHandle GPUEdgesBase::get_handle() { CUDA_NOT_SUPPORTED }

#endif
