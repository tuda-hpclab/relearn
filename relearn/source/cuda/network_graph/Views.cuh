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

#include "BloomFilter.cuh"

#include "cuda/util/SmallNeuronIdType.h"
#include "memory/DeviceVecVec.cuh"

#include <type_traits>

struct BloomFilterView;
enum class LayoutType;
using neuron_type = std::uint32_t;
using weight_type = std::int16_t;
using rank_type = std::uint16_t;
using internal_weight_type = std::int8_t;

// ── Edge tuple returned by all edge cursors ───────────────────────────────────
struct EdgeData {
    std::uint32_t other_neuron_id;
    std::uint16_t other_rank;
    std::int16_t weight; // actual weight for weighted; +1/-1 for unweighted
};

// ── Dynamic-storage edge cursor (DynamicStorageView) ─────────────────────────
// Walks up to four DeviceVecVecCursor instances in lockstep.
// IdT: SmallNeuronIdType or std::uint32_t, matching the paired MemoryPoolView<IdT>'s neuron-ID storage
// width (see CudaConfig::use_wide_neuron_ids).
template <typename IdT>
struct DynamicEdgeCursor {
    DeviceVecVecCursor<IdT> neuron_it;
    DeviceVecVecCursor<rank_type> rank_it;
    DeviceVecVecCursor<internal_weight_type> weight_it;
    DeviceVecVecCursor<bool> exc_it;
    bool has_ranks;
    bool has_weights;
    rank_type my_rank;

    __device__ bool operator==(const DynamicEdgeCursor& o) const;
    __device__ EdgeData next();
};

struct DummyCursor {
    __device__ bool operator==(const DummyCursor&) const;
    __device__ EdgeData next();
};

struct DummyView {
    __device__ weight_type get_weight(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ bool has_local_edges() const;
    __device__ neuron_type get_other_neuron(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ rank_type get_other_rank(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ char get_excitatory(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ std::size_t size(neuron_type neuron_id) const;
    __device__ [[nodiscard]] bool add_synapse(neuron_type my_neuron_id, rank_type other_rank, neuron_type other_neuron_id, bool excitatory);
    __device__ void remove_synapse(neuron_type my_neuron_id, rank_type other_rank, neuron_type other_neuron_id, bool is_excitatory);
    __device__ bool is_excitatory(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ DummyCursor begin(neuron_type) const;
    __device__ DummyCursor end(neuron_type) const;
    __device__ DummyCursor edge_begin(neuron_type) const;
    __device__ DummyCursor edge_end(neuron_type) const;
};

// IdT: SmallNeuronIdType (default, 3-byte packed) or std::uint32_t (4-byte, used when the run has
// more than max_small_neuron_id neurons -- see CudaConfig::use_wide_neuron_ids). Instantiated for
// both in Views.cu; which one a given GPUEdgesBase instance uses is recorded in its wide_ids flag
// and propagated through EdgeHandle::wide_ids for every cross-TU consumer.
template <typename IdT>
struct MemoryPoolView {
    DynamicVecVecView<IdT>* other_neuron_ids;
    DynamicVecVecView<std::int8_t>* weights;   // Can be null
    DynamicVecVecView<rank_type>* other_ranks; // Can be null
    DynamicVecVecView<bool>* excitatory;
    std::uint32_t* incoming_count;
    std::uint32_t* incoming_count_excitatory;
    std::uint32_t number_neurons;
    rank_type number_ranks;
    rank_type my_rank{};
    bool has_weights{};

    MemoryPoolView(DynamicVecVecView<IdT>* other_neuron_ids,
                   DynamicVecVecView<std::int8_t>* weights,
                   DynamicVecVecView<rank_type>* other_ranks,
                   DynamicVecVecView<bool>* excitatory,
                   std::uint32_t* incoming_count,
                   std::uint32_t* incoming_count_excitatory,
                   std::uint32_t number_neurons,
                   const rank_type& number_ranks, const rank_type& my_rank,
                   bool has_weights);

    __device__ bool has_local_edges() const;
    __device__ weight_type get_weight(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ neuron_type get_other_neuron(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ rank_type get_other_rank(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ char get_excitatory(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ std::size_t size(neuron_type neuron_id) const;
    __device__ [[nodiscard]] bool add_synapse(neuron_type my_neuron_id, rank_type other_rank, neuron_type other_neuron_id, bool excitatory_synapse);
    __device__ void remove_synapse(neuron_type my_neuron_id, rank_type other_rank, neuron_type other_neuron_id, bool is_excitatory);
    __device__ bool is_excitatory(neuron_type neuron_id, std::size_t edge_idx) const;
    __device__ DeviceVecVecCursor<IdT> begin(neuron_type neuron_id) const;
    __device__ DeviceVecVecCursor<IdT> end(neuron_type neuron_id) const;

    using EdgeCursor = DynamicEdgeCursor<IdT>;

    __device__ EdgeCursor edge_begin(neuron_type neuron_id) const;
    __device__ EdgeCursor edge_end(neuron_type neuron_id) const;
};

// ── Compile-time-typed cursor and view ───────────────────────────────────────
// DynamicEdgeCursorT<HasRanks, HasWeights>: rank_it is zero-size when HasRanks=false;
// weight_or_exc_it is the weight cursor when HasWeights=true, else the excitatory cursor.
// All branches in next() are if constexpr so unused paths are compiled away entirely.

struct CursorEmpty { };

template <bool HasRanks, bool HasWeights, typename IdT>
struct DynamicEdgeCursorT {
    DeviceVecVecCursor<IdT> neuron_it;
    [[no_unique_address]] std::conditional_t<HasRanks,
                                             DeviceVecVecCursor<rank_type>, CursorEmpty> rank_it;
    std::conditional_t<HasWeights,
                       DeviceVecVecCursor<internal_weight_type>,
                       DeviceVecVecCursor<bool>>
        weight_or_exc_it;
    rank_type my_rank;

    __device__ bool operator==(const DynamicEdgeCursorT& o) const {
        return neuron_it == o.neuron_it;
    }

    __device__ EdgeData next() {
        const auto n = neuron_it.next();
        rank_type r;
        if constexpr (HasRanks)
            r = static_cast<rank_type>(rank_it.next());
        else
            r = my_rank;
        weight_type w;
        if constexpr (HasWeights) {
            w = static_cast<weight_type>(weight_or_exc_it.next());
        } else {
            w = weight_or_exc_it.next() ? weight_type{ 1 } : weight_type{ -1 };
        }
        return { static_cast<std::uint32_t>(n), static_cast<std::uint16_t>(r), w };
    }
};

// MemoryPoolViewTyped<HasRanks, HasWeights>: unused pointer fields are zero-size.
// edge_begin/edge_end are defined inline here so they compile into the calling
// kernel TU — no separate-TU nvlink register overhead.
template <bool HasRanks, bool HasWeights, typename IdT>
struct MemoryPoolViewTyped {
    DynamicVecVecView<IdT>* other_neuron_ids;
    [[no_unique_address]] std::conditional_t<HasWeights,
                                             DynamicVecVecView<internal_weight_type>*, CursorEmpty> weights;
    [[no_unique_address]] std::conditional_t<HasRanks,
                                             DynamicVecVecView<rank_type>*, CursorEmpty> other_ranks;
    [[no_unique_address]] std::conditional_t<!HasWeights,
                                             DynamicVecVecView<bool>*, CursorEmpty> excitatory;
    std::uint32_t number_neurons;
    rank_type number_ranks;
    rank_type my_rank;

    using EdgeCursor = DynamicEdgeCursorT<HasRanks, HasWeights, IdT>;

    template <bool Cond, typename T>
    static std::conditional_t<Cond, T*, CursorEmpty> opt_ptr(T* p) noexcept {
        if constexpr (Cond)
            return p;
        else
            return CursorEmpty{};
    }

    explicit MemoryPoolViewTyped(const MemoryPoolView<IdT>& v) noexcept
        : other_neuron_ids(v.other_neuron_ids)
        , weights(opt_ptr<HasWeights>(v.weights))
        , other_ranks(opt_ptr<HasRanks>(v.other_ranks))
        , excitatory(opt_ptr<!HasWeights>(v.excitatory))
        , number_neurons(v.number_neurons)
        , number_ranks(v.number_ranks)
        , my_rank(v.my_rank)

    { }

    __device__ EdgeCursor edge_begin(neuron_type neuron_id) const {
        if constexpr (!HasRanks && HasWeights)
            return EdgeCursor{ other_neuron_ids->begin(neuron_id), {}, weights->begin(neuron_id), my_rank };
        else if constexpr (!HasRanks && !HasWeights)
            return EdgeCursor{ other_neuron_ids->begin(neuron_id), {}, excitatory->begin(neuron_id), my_rank };
        else if constexpr (HasRanks && HasWeights)
            return EdgeCursor{ other_neuron_ids->begin(neuron_id), other_ranks->begin(neuron_id), weights->begin(neuron_id), my_rank };
        else
            return EdgeCursor{ other_neuron_ids->begin(neuron_id), other_ranks->begin(neuron_id), excitatory->begin(neuron_id), my_rank };
    }

    __device__ EdgeCursor edge_end(neuron_type neuron_id) const {
        return EdgeCursor{ other_neuron_ids->end(neuron_id), {}, {}, my_rank };
    }
};

using mpi_rank_type = std::uint16_t;
using neuron_id_type = std::uint32_t;
using weight_type = std::int16_t;

// Type traits so code generic over a view type (e.g. templated on LocalEdges/OutgoingNetwork) can
// ask "is this a MemoryPoolView/OnlyOutgoingView, for any IdT?" without needing to know IdT itself -- see
// ExchangeAlgorithm.cu's find_synapses_to_delete_random_internal_entry and friends.
template <typename T>
struct is_memory_pool_view : std::false_type { };
template <typename IdT>
struct is_memory_pool_view<MemoryPoolView<IdT>> : std::true_type { };
template <typename T>
inline constexpr bool is_memory_pool_view_v = is_memory_pool_view<T>::value;