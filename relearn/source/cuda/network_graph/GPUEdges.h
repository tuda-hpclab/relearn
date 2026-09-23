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

#include "LayoutType.h"

#include "cpp-utility/MemoryFootprint.hpp"
#include "cuda/CudaConfig.h"
#include "cuda/memory//DeviceArray.h"
#include "cuda/memory/DeviceVecVec.h"
#include "cuda/memory/LazySyncedArray.h"
#include "cuda/memory/SharedBlockPool.h"
#include "cuda/network_graph/BloomFilter.h"
#include "cuda/util/SmallNeuronIdType.h"
#include "util/RelearnException.h"

#include <functional>
#include <iostream>

class RankNeuronId;
class NeuronID;

struct EdgeHandle;

template <typename NeuronIDType, typename T>
class GraphStorage {
public:
    GraphStorage() = default;
    GraphStorage(const GraphStorage&) = default;
    GraphStorage& operator=(const GraphStorage&) = default;
    GraphStorage(GraphStorage&&) = default;
    GraphStorage& operator=(GraphStorage&&) = default;

    virtual void add(NeuronIDType neuron_id, T value) = 0;

    virtual void force_update() = 0;

    virtual void init(NeuronIDType _number_neurons,
                      std::size_t _expected_synapses_per_neuron,
                      SharedBlockPool* shared_pool = nullptr) = 0;

    // Returns true if this storage allocates a DynamicVecVec (i.e. MemoryPoolStorage).
    [[nodiscard]] virtual bool uses_shared_pool() const { return false; }

    [[nodiscard]] virtual std::uint64_t get_gpu_memory_footprint() const = 0;

    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const = 0;

    virtual void record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const = 0;

    [[nodiscard]] virtual std::vector<std::vector<T>> copy_to_cpu() const = 0;

    virtual ~GraphStorage() = default;
};

template <typename NeuronIDType, typename T>
class MemoryPoolStorage : public GraphStorage<NeuronIDType, T> {
public:
    MemoryPoolStorage() = default;
    std::unique_ptr<DynamicVecVec<T>> data{};
    NeuronIDType number_neurons{};

    std::unordered_map<NeuronIDType, std::vector<T>> host_temp_data{};

    void add(NeuronIDType neuron_id, const T value) override {
        host_temp_data[neuron_id].emplace_back(value);
    }

    void force_update() override {
        data->add_from_map(host_temp_data);
        host_temp_data.clear();
    }

    [[nodiscard]] bool uses_shared_pool() const override { return true; }

    void init(const NeuronIDType _number_neurons,
              const std::size_t _expected_synapses_per_neuron,
              SharedBlockPool* shared_pool = nullptr) override {
        number_neurons = _number_neurons;
        if (shared_pool != nullptr) { // NOLINT(bugprone-branch-clone) - different DynamicVecVec constructor overloads (with/without shared pool), not identical branches
            data = std::make_unique<DynamicVecVec<T>>(_number_neurons, _expected_synapses_per_neuron, shared_pool);
        } else {
            data = std::make_unique<DynamicVecVec<T>>(_number_neurons);
        }
    }

    void record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const override {
        data->record_usage_footprint(footprint, prefix);
    }

    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const override {
        data->record_memory_footprint(footprint, prefix);
    }

    [[nodiscard]] std::uint64_t get_gpu_memory_footprint() const override {
        return data->get_gpu_memory_footprint();
    }

    [[nodiscard]] std::vector<std::vector<T>> copy_to_cpu() const override {
        return data->copy_to_host();
    }
};

// Like MemoryPoolStorage, but the internal GPU packing (PackedT: SmallNeuronIdType or std::uint32_t)
// is independent of the value type exposed through GraphStorage's virtual interface (always
// std::uint32_t, the logical neuron ID). This lets GPUEdgesBase::other_neurons_storage hold either
// packing behind one polymorphic pointer, so callers don't need to know which one was chosen -- see
// CudaConfig::use_wide_neuron_ids.
template <typename NeuronIDType, typename PackedT>
class MemoryPoolNeuronIdStorage : public GraphStorage<NeuronIDType, std::uint32_t> {
public:
    MemoryPoolNeuronIdStorage() = default;
    std::unique_ptr<DynamicVecVec<PackedT>> data{};
    NeuronIDType number_neurons{};

    std::unordered_map<NeuronIDType, std::vector<PackedT>> host_temp_data{};

    void add(NeuronIDType neuron_id, const std::uint32_t value) override {
        host_temp_data[neuron_id].emplace_back(static_cast<PackedT>(value));
    }

    void force_update() override {
        data->add_from_map(host_temp_data);
        host_temp_data.clear();
    }

    [[nodiscard]] bool uses_shared_pool() const override { return true; }

    void init(const NeuronIDType _number_neurons,
              const std::size_t _expected_synapses_per_neuron,
              SharedBlockPool* shared_pool = nullptr) override {
        number_neurons = _number_neurons;
        if (shared_pool != nullptr) { // NOLINT(bugprone-branch-clone) - different DynamicVecVec constructor overloads (with/without shared pool), not identical branches
            data = std::make_unique<DynamicVecVec<PackedT>>(_number_neurons, _expected_synapses_per_neuron, shared_pool);
        } else {
            data = std::make_unique<DynamicVecVec<PackedT>>(_number_neurons);
        }
    }

    void record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const override {
        data->record_usage_footprint(footprint, prefix);
    }

    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const override {
        data->record_memory_footprint(footprint, prefix);
    }

    [[nodiscard]] std::uint64_t get_gpu_memory_footprint() const override {
        return data->get_gpu_memory_footprint();
    }

    [[nodiscard]] std::vector<std::vector<std::uint32_t>> copy_to_cpu() const override {
        auto raw = data->copy_to_host();
        std::vector<std::vector<std::uint32_t>> result(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            result[i].reserve(raw[i].size());
            for (const auto& v : raw[i]) {
                result[i].push_back(static_cast<std::uint32_t>(v));
            }
        }
        return result;
    }
};

struct DefaultView;

struct Features {
    bool has_weights{};
    bool has_ranks{};
};

class GPUEdgesBase {
public:
    virtual ~GPUEdgesBase();

    GPUEdgesBase(const GPUEdgesBase&) = delete;
    GPUEdgesBase& operator=(const GPUEdgesBase&) = delete;
    GPUEdgesBase(GPUEdgesBase&&) = delete;
    GPUEdgesBase& operator=(GPUEdgesBase&&) = delete;

    using mpi_rank_type = std::uint16_t;
    using neuron_id_type = std::uint32_t;
    using weight_type = std::int16_t;
    using internal_weight_type = std::int8_t;

    using memory_pool_neuron_small = MemoryPoolNeuronIdStorage<neuron_id_type, SmallNeuronIdType>;
    using memory_pool_neuron_wide = MemoryPoolNeuronIdStorage<neuron_id_type, std::uint32_t>;
    using memory_pool_weight = MemoryPoolStorage<neuron_id_type, std::int8_t>;

    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix);
    void record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const;

    GPUEdgesBase(LayoutType layout_, Features features, const mpi_rank_type number_ranks_, const mpi_rank_type my_rank_)
        : wide_ids(CudaConfig::use_wide_neuron_ids)
        , number_ranks(number_ranks_)
        , my_rank(my_rank_)
        , layout(layout_) {
        if (layout == LayoutType::MemoryPool) {
            other_neurons_storage = wide_ids
                                        ? std::unique_ptr<GraphStorage<neuron_id_type, std::uint32_t>>(std::make_unique<memory_pool_neuron_wide>())
                                        : std::unique_ptr<GraphStorage<neuron_id_type, std::uint32_t>>(std::make_unique<memory_pool_neuron_small>());
            if (features.has_weights) { // NOLINT(bugprone-branch-clone) - mutually exclusive weighted vs excitatory feature setup, not identical branches
                weighted.enabled = true;
                weighted.weight_storage = std::make_unique<memory_pool_weight>();
                excitatory.enabled = false;
            } else {
                excitatory.enabled = true;
                excitatory.excitatory_storage = std::make_unique<MemoryPoolStorage<neuron_id_type, bool>>();
            }
            if (features.has_ranks) {
                other_ranks.enabled = true;
                other_ranks.other_ranks_storage = std::make_unique<MemoryPoolStorage<neuron_id_type, mpi_rank_type>>();
            }
            incoming_count_data.enabled = true;
        } else {
            RelearnException::fail("Not supported yet");
        }
    }

    std::unique_ptr<GraphStorage<neuron_id_type, std::uint32_t>> other_neurons_storage{};

    // Set once in the constructor from CudaConfig::use_wide_neuron_ids. Selects which concrete
    // MemoryPoolNeuronIdStorage<...> other_neurons_storage holds, and which MemoryPoolView/OnlyOutgoingView
    // template instantiation (and IdT-dependent kernels) this instance's edges use downstream.
    bool wide_ids{ false };

    struct OtherRankFeature {
        std::unique_ptr<GraphStorage<neuron_id_type, mpi_rank_type>> other_ranks_storage{};
        bool enabled{};
    } other_ranks;

    struct WeightFeature {
        std::unique_ptr<GraphStorage<neuron_id_type, internal_weight_type>> weight_storage{};
        bool enabled{};
    } weighted;

    struct ExcitatoryFeature {
        std::unique_ptr<GraphStorage<neuron_id_type, bool>> excitatory_storage{};
        bool enabled{};
    } excitatory;

    struct IncomingCountFeature {
        bool enabled{};
        LazySyncedArray<std::uint32_t> count{};
        LazySyncedArray<std::uint32_t> count_excitatory{};
    } incoming_count_data;

    // Builds active_neuron_ids/active_neuron_count from dirty_flags (stream compaction). Must be
    // called once per rebuild() cycle before remove_deleted_entries_from_memory_pool()/sort(), which
    // both read active_neuron_ids/active_neuron_count instead of iterating all number_neurons.
    void build_dirty_active_list();

    // Resets dirty_flags to all-false. Call once per rebuild() cycle, after both
    // remove_deleted_entries_from_memory_pool() and sort() have consumed the current active list.
    void clear_dirty_flags();

    [[nodiscard]] EdgeHandle get_handle();

    virtual void init(neuron_id_type _number_neurons, std::size_t _expected_synapses_per_neuron,
                      SharedBlockPool* shared_pool = nullptr);

    // Returns the number of DynamicVecVec instances this GPUEdges will create on init().
    [[nodiscard]] std::size_t count_pool_consumers() const;

    [[nodiscard]] virtual std::uint64_t get_gpu_memory_footprint() const;

    void update_edges(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_edges, const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_edges);

    // virtual void check_gpu_cpu_equivalence(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_edges_host, const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_edges_host, const mpi_rank_type my_rank) const {}

    void rebuild();

    std::vector<std::vector<std::tuple<mpi_rank_type, neuron_id_type, weight_type>>> copy_to_host() const;

protected:
    neuron_id_type number_neurons{};
    mpi_rank_type number_ranks{};
    mpi_rank_type my_rank{};
    std::size_t expected_synapses_per_neuron{};

    void* my_view{};

    LayoutType layout{};
    bool _init{ false };

private:
    // IdT-templated implementations, dispatched on wide_ids by the public (non-template) methods
    // above -- IdT is either SmallNeuronIdType or std::uint32_t, matching whichever concrete
    // MemoryPoolNeuronIdStorage<...>/MemoryPoolView<...>/OnlyOutgoingView<...> instantiation this instance's
    // wide_ids selected. Defined (and explicitly instantiated for both IdT) in GPUEdges.cu.
    template <typename IdT>
    [[nodiscard]] void* build_handle_impl();
    template <typename IdT>
    void delete_view_impl();
};
