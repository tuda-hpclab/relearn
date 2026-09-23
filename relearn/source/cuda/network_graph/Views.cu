/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BloomFilter.h"
#include "Views.cuh"

#include <chrono>

// ── DynamicEdgeCursor ─────────────────────────────────────────────────────────

template <typename IdT>
__device__ bool DynamicEdgeCursor<IdT>::operator==(const DynamicEdgeCursor& o) const {
    return neuron_it == o.neuron_it;
}

template <typename IdT>
__device__ EdgeData DynamicEdgeCursor<IdT>::next() {
    const auto n = neuron_it.next();
    const auto r = has_ranks ? rank_it.next() : my_rank;
    weight_type w;
    if (has_weights) {
        w = weight_it.next();
    } else {
        const char e = exc_it.next();
        w = e ? weight_type{ 1 } : weight_type{ -1 };
    }
    return { static_cast<std::uint32_t>(n), static_cast<std::uint16_t>(r), static_cast<std::int16_t>(w) };
}

// ── DummyCursor ───────────────────────────────────────────────────────────────

__device__ bool DummyCursor::operator==(const DummyCursor&) const {
    return true;
}

__device__ EdgeData DummyCursor::next() {
    RELEARN_DEVICE_CUDA_CHECK(false, "DummyCursor::next: Not implemented");
    return {};
}

// ── DummyView ─────────────────────────────────────────────────────────────────

__device__ weight_type DummyView::get_weight(neuron_type neuron_id, std::size_t edge_idx) const {
    RELEARN_DEVICE_CUDA_CHECK(false, "DummyView::get_weight: Not implemented");
    return 0;
}

__device__ bool DummyView::has_local_edges() const {
    return false;
}

__device__ neuron_type DummyView::get_other_neuron(neuron_type neuron_id, std::size_t edge_idx) const {
    RELEARN_DEVICE_CUDA_CHECK(false, "DummyView::get_other_neuron: Not implemented");
    return 0;
}

__device__ rank_type DummyView::get_other_rank(neuron_type neuron_id, std::size_t edge_idx) const {
    RELEARN_DEVICE_CUDA_CHECK(false, "DummyView::get_other_rank: Not implemented");
    return 0;
}

__device__ char DummyView::get_excitatory(neuron_type neuron_id, std::size_t edge_idx) const {
    RELEARN_DEVICE_CUDA_CHECK(false, "DummyView::get_excitatory: Not implemented");
    return false;
}

__device__ std::size_t DummyView::size(neuron_type neuron_id) const {
    return 0;
}

__device__ bool DummyView::add_synapse(neuron_type my_neuron_id, rank_type other_rank, neuron_type other_neuron_id,
                                       bool excitatory) {
    RELEARN_DEVICE_CUDA_CHECK(false, "DummyView::add_synapse: Not implemented");
}

__device__ void DummyView::remove_synapse(neuron_type my_neuron_id, rank_type other_rank, neuron_type other_neuron_id, bool is_excitatory) {
    RELEARN_DEVICE_CUDA_CHECK(false, "DummyView::remove_synapse: Not implemented");
}

__device__ bool DummyView::is_excitatory(neuron_type neuron_id, std::size_t edge_idx) const {
    RELEARN_DEVICE_CUDA_CHECK(false, "DummyView::is_excitatory: Not implemented");
    return false;
}

__device__ DummyCursor DummyView::begin(neuron_type) const {
    return DummyCursor{};
}

__device__ DummyCursor DummyView::end(neuron_type) const {
    return DummyCursor{};
}

__device__ DummyCursor DummyView::edge_begin(neuron_type) const {
    return DummyCursor{};
}

__device__ DummyCursor DummyView::edge_end(neuron_type) const {
    return DummyCursor{};
}

// ── MemoryPoolView ─────────────────────────────────────────────────────────────────

template <typename IdT>
MemoryPoolView<IdT>::MemoryPoolView(DynamicVecVecView<IdT>* other_neuron_ids,
                                    DynamicVecVecView<std::int8_t>* weights,
                                    DynamicVecVecView<rank_type>* other_ranks,
                                    DynamicVecVecView<bool>* excitatory,
                                    std::uint32_t* incoming_count,
                                    std::uint32_t* incoming_count_excitatory,
                                    std::uint32_t number_neurons,
                                    const rank_type& number_ranks, const rank_type& my_rank,
                                    bool has_weights)
    : other_neuron_ids(other_neuron_ids)
    , weights(weights)
    , other_ranks(other_ranks)
    , excitatory(excitatory)
    , incoming_count(incoming_count)
    , incoming_count_excitatory(incoming_count_excitatory)
    , my_rank(my_rank)
    , number_neurons(number_neurons)
    , number_ranks(number_ranks)
    , has_weights(has_weights) {
}

template <typename IdT>
__device__ bool MemoryPoolView<IdT>::has_local_edges() const {
    return true;
}

template <typename IdT>
__device__ weight_type MemoryPoolView<IdT>::get_weight(neuron_type neuron_id, std::size_t edge_idx) const {
    if (weights == nullptr) {
        RELEARN_DEVICE_CUDA_CHECK(excitatory != nullptr, "MemoryPoolView::get_weight: No excitatory");
        return excitatory->get(neuron_id, edge_idx) ? 1 : -1;
    }
    return weights->get(neuron_id, static_cast<std::uint32_t>(edge_idx));
}

template <typename IdT>
__device__ neuron_type MemoryPoolView<IdT>::get_other_neuron(neuron_type neuron_id, std::size_t edge_idx) const {
    return other_neuron_ids->get(neuron_id, static_cast<std::uint32_t>(edge_idx));
}

template <typename IdT>
__device__ rank_type MemoryPoolView<IdT>::get_other_rank(neuron_type neuron_id, std::size_t edge_idx) const {
    if (other_ranks == nullptr) {
        return my_rank;
    }
    return other_ranks->get(neuron_id, static_cast<std::uint32_t>(edge_idx));
}

template <typename IdT>
__device__ char MemoryPoolView<IdT>::get_excitatory(neuron_type neuron_id, std::size_t edge_idx) const {
    RELEARN_DEVICE_CUDA_CHECK(false, "MemoryPoolView::get_excitatory: No excitatory infos");
    return false;
}

template <typename IdT>
__device__ std::size_t MemoryPoolView<IdT>::size(neuron_type neuron_id) const {
    return other_neuron_ids->get_size(neuron_id);
}

template <typename IdT>
__device__ bool MemoryPoolView<IdT>::add_synapse(neuron_type my_neuron_id, rank_type other_rank, neuron_type other_neuron_id,
                                                 bool excitatory_synapse) {
    atomicAdd(incoming_count + my_neuron_id, 1U);
    if (excitatory_synapse) {
        atomicAdd(incoming_count_excitatory + my_neuron_id, 1U);
    }
    if (weights != nullptr) {
        const auto n = size(my_neuron_id);
        RELEARN_DEVICE_CUDA_CHECK(other_neuron_ids->get_size(my_neuron_id) == weights->get_size(my_neuron_id),
                                  "MemoryPoolView::add_synapse: other_neuron_ids size %u != weights size %u for neuron %u", other_neuron_ids->get_size(my_neuron_id),
                                  weights->get_size(my_neuron_id), my_neuron_id);
        for (auto i = 0U; i < n; i++) {
            if (static_cast<neuron_type>(other_neuron_ids->get(my_neuron_id, i)) == other_neuron_id && get_other_rank(my_neuron_id, i) == other_rank) {
                const auto cur = weights->get(my_neuron_id, static_cast<std::uint32_t>(i));
                RELEARN_DEVICE_CUDA_CHECK(cur > 0 && excitatory_synapse || cur < 0 && !excitatory_synapse,
                                          "MemoryPoolView::add_synapse: Mixing excitatory and inhibitory connections %u %u %d", my_neuron_id, other_neuron_id, cur);
                if (cur == std::numeric_limits<std::int8_t>::max() || cur == std::numeric_limits<std::int8_t>::min()) {
                    continue;
                }
                const auto new_weight = static_cast<std::int8_t>(cur + (excitatory_synapse ? 1 : -1));
                weights->set(my_neuron_id, static_cast<std::uint32_t>(i),
                             static_cast<std::int8_t>(new_weight));
                RELEARN_DEVICE_CUDA_CHECK(other_neuron_ids->get_size(my_neuron_id) == weights->get_size(my_neuron_id),
                                          "MemoryPoolView::add_synapse: other_neuron_ids size %u != weights size %u", other_neuron_ids->get_size(my_neuron_id),
                                          weights->get_size(my_neuron_id));
                return true;
            }
        }
    }
    auto successful = other_neuron_ids->add(my_neuron_id, static_cast<IdT>(other_neuron_id));
    if (!successful) {
        return false;
    }
    RELEARN_DEVICE_CUDA_CHECK(my_rank == other_rank && other_ranks == nullptr || my_rank != other_rank && other_ranks != nullptr, "MemoryPoolView::add_synapse: Mixing local and distant edges");
    if (other_ranks != nullptr) {
        successful = other_ranks->add(my_neuron_id, std::move(other_rank));
        if (!successful) {
            return false;
        }
    }
    if (weights != nullptr) {
        successful = weights->add(my_neuron_id, static_cast<std::int8_t>(excitatory_synapse ? 1 : -1));
        if (!successful) {
            return false;
        }
        RELEARN_DEVICE_CUDA_CHECK(other_neuron_ids->get_size(my_neuron_id) == weights->get_size(my_neuron_id),
                                  "MemoryPoolView::add_synapse: other_neuron_ids size %u != weights size %u", other_neuron_ids->get_size(my_neuron_id),
                                  weights->get_size(my_neuron_id));
    } else {
        RELEARN_DEVICE_CUDA_CHECK(excitatory != nullptr, "MemoryPoolView::add_synapse: No excitatory");
        successful = excitatory->add(my_neuron_id, excitatory_synapse);
        if (!successful) {
            return false;
        }
        RELEARN_DEVICE_CUDA_CHECK(excitatory->get(my_neuron_id, excitatory->get_size(my_neuron_id) - 1) == excitatory_synapse, "MemoryPoolView::add_synapse: excitatory flag mismatch after add");
    }

    if (weights != nullptr) {
        RELEARN_DEVICE_CUDA_CHECK(
            other_neuron_ids->get_size(my_neuron_id) == weights->get_size(my_neuron_id),
            "MemoryPoolView::add_synapse: other_neuron_ids size %u != weights size %u", other_neuron_ids->get_size(my_neuron_id),
            weights->get_size(my_neuron_id));
    } else {
        RELEARN_DEVICE_CUDA_CHECK(
            other_neuron_ids->get_size(my_neuron_id) == excitatory->get_size(my_neuron_id),
            "MemoryPoolView::add_synapse: other_neuron_ids size %u != excitatory size %u", other_neuron_ids->get_size(my_neuron_id),
            excitatory->get_size(my_neuron_id));
    }
    return true;
}

template <typename IdT>
__device__ void MemoryPoolView<IdT>::remove_synapse(neuron_type my_neuron_id, rank_type other_rank, neuron_type other_neuron_id, bool is_excitatory) {
    const auto n = size(my_neuron_id);
    for (auto i = 0U; i < n; i++) {
        if ((static_cast<neuron_type>(other_neuron_ids->get(my_neuron_id, i)) == other_neuron_id && get_other_rank(my_neuron_id, i) == other_rank)) {
            if (weights != nullptr) {
                const auto cur = weights->get(my_neuron_id, static_cast<std::uint32_t>(i));
                RELEARN_DEVICE_CUDA_CHECK(cur > 0 && is_excitatory || cur < 0 && !is_excitatory, "MemoryPoolView::remove_synapse: weight sign does not match is_excitatory: %i %d", is_excitatory, cur);
                const auto delta = static_cast<std::int8_t>(is_excitatory ? -1 : 1);
                const auto new_weight = static_cast<std::int8_t>(cur + delta);
                if (new_weight != 0) {
                    weights->set(my_neuron_id, static_cast<std::uint32_t>(i), new_weight);
                    RELEARN_DEVICE_CUDA_CHECK(
                        other_neuron_ids->get_size(my_neuron_id) == weights->get_size(my_neuron_id),
                        "MemoryPoolView::remove_synapse: other_neuron_ids size %u != weights size %u (neuron %u, edge %u:%u)", other_neuron_ids->get_size(my_neuron_id),
                        weights->get_size(my_neuron_id), my_neuron_id, other_rank, other_neuron_id);
                    auto prev = atomicSub(incoming_count + my_neuron_id, 1U);
                    RELEARN_DEVICE_CUDA_CHECK(prev > 0, "MemoryPoolView::remove_synapse: incoming_count was already 0 before decrement");
                    if (is_excitatory) {
                        prev = atomicSub(incoming_count_excitatory + my_neuron_id, 1U);
                        RELEARN_DEVICE_CUDA_CHECK(prev > 0, "MemoryPoolView::remove_synapse: incoming_count_excitatory was already 0 before decrement");
                    }

                    if (weights != nullptr) {
                        RELEARN_DEVICE_CUDA_CHECK(
                            other_neuron_ids->get_size(my_neuron_id) == weights->get_size(my_neuron_id),
                            "MemoryPoolView::remove_synapse: other_neuron_ids size %u != weights size %u", other_neuron_ids->get_size(my_neuron_id),
                            weights->get_size(my_neuron_id));
                    } else {
                        RELEARN_DEVICE_CUDA_CHECK(
                            other_neuron_ids->get_size(my_neuron_id) == excitatory->get_size(my_neuron_id),
                            "MemoryPoolView::remove_synapse: other_neuron_ids size %u != excitatory size %u", other_neuron_ids->get_size(my_neuron_id),
                            excitatory->get_size(my_neuron_id));
                    }

                    return;
                }
            }

            auto prev = atomicSub(incoming_count + my_neuron_id, 1U);
            RELEARN_DEVICE_CUDA_CHECK(prev > 0, "MemoryPoolView::remove_synapse: incoming_count was already 0 before decrement");
            if (is_excitatory) {
                prev = atomicSub(incoming_count_excitatory + my_neuron_id, 1U);
                RELEARN_DEVICE_CUDA_CHECK(prev > 0, "MemoryPoolView::remove_synapse: incoming_count_excitatory was already 0 before decrement");
            }

            const auto cur_size = other_neuron_ids->get_size(my_neuron_id);
            const auto back_neuron_id = other_neuron_ids->get(my_neuron_id, cur_size - 1);
            other_neuron_ids->set(my_neuron_id, i, back_neuron_id);
            other_neuron_ids->pop_back(my_neuron_id);
            if (other_ranks != nullptr) {

                const auto back_rank = other_ranks->get(my_neuron_id, cur_size - 1);
                other_ranks->set(my_neuron_id, i, back_rank);
                other_ranks->pop_back(my_neuron_id);
            }
            if (weights != nullptr) {
                const auto back_weight = weights->get(my_neuron_id, cur_size - 1);
                weights->set(my_neuron_id, i, back_weight);
                weights->pop_back(my_neuron_id);
                RELEARN_DEVICE_CUDA_CHECK(other_neuron_ids->get_size(my_neuron_id) == weights->get_size(my_neuron_id),
                                          "MemoryPoolView::remove_synapse: other_neuron_ids size %u != weights size %u", other_neuron_ids->get_size(my_neuron_id),
                                          weights->get_size(my_neuron_id));
            } else {
                RELEARN_DEVICE_CUDA_CHECK(excitatory != nullptr, "MemoryPoolView::remove_synapse: No excitatory");
                const auto back_exc = excitatory->get(my_neuron_id, cur_size - 1);
                excitatory->set(my_neuron_id, i, back_exc);
                excitatory->pop_back(my_neuron_id);
            }

            if (weights != nullptr) {
                RELEARN_DEVICE_CUDA_CHECK(
                    other_neuron_ids->get_size(my_neuron_id) == weights->get_size(my_neuron_id),
                    "MemoryPoolView::remove_synapse: other_neuron_ids size %u != weights size %u", other_neuron_ids->get_size(my_neuron_id),
                    weights->get_size(my_neuron_id));
            } else {
                RELEARN_DEVICE_CUDA_CHECK(
                    other_neuron_ids->get_size(my_neuron_id) == excitatory->get_size(my_neuron_id),
                    "MemoryPoolView::remove_synapse: other_neuron_ids size %u != excitatory size %u", other_neuron_ids->get_size(my_neuron_id),
                    excitatory->get_size(my_neuron_id));
            }

            return;
        }
    }
    RELEARN_DEVICE_CUDA_CHECK(false, "MemoryPoolView::remove_synapse: edge not found %u:%u %u:%u", my_rank, my_neuron_id, other_rank, other_neuron_id);
}

template <typename IdT>
__device__ bool MemoryPoolView<IdT>::is_excitatory(neuron_type neuron_id, std::size_t edge_idx) const {
    if (weights != nullptr) {
        return weights->get(neuron_id, static_cast<std::uint32_t>(edge_idx)) > 0;
    }
    return excitatory->get(neuron_id, edge_idx);
}

template <typename IdT>
__device__ DeviceVecVecCursor<IdT> MemoryPoolView<IdT>::begin(neuron_type neuron_id) const {
    return other_neuron_ids->begin(neuron_id);
}

template <typename IdT>
__device__ DeviceVecVecCursor<IdT> MemoryPoolView<IdT>::end(neuron_type neuron_id) const {
    return other_neuron_ids->end(neuron_id);
}

template <typename IdT>
__device__ typename MemoryPoolView<IdT>::EdgeCursor MemoryPoolView<IdT>::edge_begin(neuron_type neuron_id) const {
    const bool hr = other_ranks != nullptr;
    const bool hw = weights != nullptr;
    RELEARN_DEVICE_CUDA_CHECK(neuron_id < number_neurons, "MemoryPoolView::edge_begin: neuron_id too large");
    RELEARN_DEVICE_CUDA_CHECK(weights != nullptr || excitatory->main_chunks > 0, "MemoryPoolView::edge_begin: no weights and excitatory not initialized");
    return EdgeCursor{
        other_neuron_ids->begin(neuron_id),
        hr ? other_ranks->begin(neuron_id) : DeviceVecVecCursor<rank_type>{},
        hw ? weights->begin(neuron_id) : DeviceVecVecCursor<std::int8_t>{},
        !hw ? excitatory->begin(neuron_id) : DeviceVecVecCursor<bool>{},
        hr, hw, my_rank
    };
}

template <typename IdT>
__device__ typename MemoryPoolView<IdT>::EdgeCursor MemoryPoolView<IdT>::edge_end(neuron_type neuron_id) const {
    return EdgeCursor{
        other_neuron_ids->end(neuron_id),
        {},
        {},
        {},
        false,
        false
    };
}

// Explicit instantiations -- both packed neuron-ID widths are compiled into this TU; which one a
// given run uses is chosen at runtime via GPUEdgesBase::wide_ids (see CudaConfig::use_wide_neuron_ids).
template struct DynamicEdgeCursor<SmallNeuronIdType>;
template struct DynamicEdgeCursor<std::uint32_t>;
template class MemoryPoolView<SmallNeuronIdType>;
template class MemoryPoolView<std::uint32_t>;
