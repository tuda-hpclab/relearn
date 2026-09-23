/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BlockCountAtomics.cuh"
#include "BloomFilter.h"
#include "GPUEdges.h"
#include "NetworkHandle.h"
#include "Views.cuh"

#include "cuda/spikes/ExchangeAlgorithm.h"
#include "cuda/util/SmallNeuronIdType.h"
#include "util/Timers.h"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

GPUEdgesBase::~GPUEdgesBase() {
    if (my_view == nullptr)
        return;
    if (wide_ids)
        delete_view_impl<std::uint32_t>();
    else
        delete_view_impl<SmallNeuronIdType>();
}

template <typename IdT>
void GPUEdgesBase::delete_view_impl() {
    if (layout == LayoutType::MemoryPool) {
        delete static_cast<MemoryPoolView<IdT>*>(my_view);
    }
}

EdgeHandle GPUEdgesBase::get_handle() {

    RELEARN_CUDA_CHECK(_init, "GPUEdgesBase::get_handle: get_handle called before init");
    if (my_view == nullptr) {
        my_view = wide_ids ? build_handle_impl<std::uint32_t>() : build_handle_impl<SmallNeuronIdType>();
    }
    return { layout, my_view, wide_ids };
}

template <typename IdT>
void* GPUEdgesBase::build_handle_impl() {
    using memory_pool_neuron_t = MemoryPoolNeuronIdStorage<neuron_id_type, IdT>;
    if (layout == LayoutType::MemoryPool) {
        auto* p = dynamic_cast<memory_pool_neuron_t*>(other_neurons_storage.get());
        auto dd1 = p->data->get_device_view();
        auto dd2 = weighted.enabled ? dynamic_cast<memory_pool_weight*>(weighted.weight_storage.get())->data->get_device_view() : nullptr;
        auto dd3 = other_ranks.enabled ? dynamic_cast<MemoryPoolStorage<neuron_id_type, mpi_rank_type>*>(other_ranks.other_ranks_storage.get())->data->get_device_view() : nullptr;
        auto dd4 = excitatory.enabled ? dynamic_cast<MemoryPoolStorage<neuron_id_type, bool>*>(excitatory.excitatory_storage.get())->data->get_device_view() : nullptr;
        auto* dd_count = incoming_count_data.count.get_device_ptr();
        auto* dd_count_exc = incoming_count_data.count_excitatory.get_device_ptr();
        return new MemoryPoolView<IdT>(
            dd1, dd2, dd3, dd4,
            dd_count, dd_count_exc,
            number_neurons,
            number_ranks,
            my_rank,
            weighted.enabled);
    } else {
        RELEARN_CUDA_CHECK(false, "GPUEdgesBase::build_handle_impl: Invalid layout");
        return nullptr;
    }
}

std::uint64_t GPUEdgesBase::get_gpu_memory_footprint() const {
    std::uint64_t sum = 0;
    if (other_neurons_storage != nullptr) {
        sum += other_neurons_storage->get_gpu_memory_footprint();
    }
    if (other_ranks.enabled) {
        sum += other_ranks.other_ranks_storage->get_gpu_memory_footprint();
    }
    if (weighted.enabled) {
        sum += weighted.weight_storage->get_gpu_memory_footprint();
    }
    if (incoming_count_data.enabled) {
        sum += incoming_count_data.count.get_memory_footprint();
    }
    return sum;
}