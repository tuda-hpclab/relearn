/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#ifdef RELEARN_CUDA_ENABLED

#include "benchmark_network_graph_gpu_edge_iteration.h"

#include "cuda/network_graph/NetworkHandle.h"
#include "cuda/network_graph/NetworkGraph.cuh"
#include "cuda/util/SmallNeuronIdType.h"

#include <cstdint>

template <typename IncomingLocalView, typename IncomingDistantView>
__global__ void kernel_iterate_edges(const IncomingLocalView local, const IncomingDistantView distant,
                                     const uint32_t test_neuron_id, uint64_t* d_edge_count) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint64_t total_edges = 0;

        {
            auto it = local.edge_begin(test_neuron_id);
            const auto sentinel = local.edge_end(test_neuron_id);
            while (!(it == sentinel)) {
                const auto [other_neuron_id, other_rank, weight] = it.next();
                total_edges++;
            }
        }

        {
            auto it = distant.edge_begin(test_neuron_id);
            const auto sentinel = distant.edge_end(test_neuron_id);
            while (!(it == sentinel)) {
                const auto [other_neuron_id, other_rank, weight] = it.next();
                total_edges++;
            }
        }

        d_edge_count[0] = total_edges;
    }
}

template <typename IncomingLocalView, typename IncomingDistantView>
static void launch_typed(const IncomingLocalView local, const IncomingDistantView distant,
                         const uint32_t test_neuron_id, uint64_t* d_edge_count) {
    kernel_iterate_edges<<<1, 1>>>(local, distant, test_neuron_id, d_edge_count);
}

// Distant incoming edges are always LayoutType::MemoryPool in every NetworkGraphGPUParams config this
// benchmark builds (OnlyOutgoing/Dummy are only used for local edges), so distant is dispatched on
// IdT alone; local additionally branches over OnlyOutgoing/MemoryPool/Dummy.
template <typename IdT>
static void launch_with_distant_memory_pool(const EdgeHandle& local_handle, const EdgeHandle& distant_handle,
                                            const uint32_t test_neuron_id, uint64_t* d_edge_count) {
    auto& distant = *static_cast<MemoryPoolView<IdT>*>(distant_handle.view_impl);
    switch (local_handle.layout_type) {
    case LayoutType::MemoryPool:
        launch_typed(*static_cast<MemoryPoolView<IdT>*>(local_handle.view_impl), distant, test_neuron_id, d_edge_count);
        return;
    case LayoutType::Dummy:
        launch_typed(*static_cast<DummyView*>(local_handle.view_impl), distant, test_neuron_id, d_edge_count);
        return;
    }
}

void launch_kernel_iterate_edges(const NetworkHandle& network, const uint32_t test_neuron_id,
                                 uint64_t* d_edge_count) {
    const auto& local_handle = network.incoming_local_handle;
    const auto& distant_handle = network.incoming_distant_handle;

    // Both handles come from GPUEdgesBase instances built from the same global
    // CudaConfig::use_wide_neuron_ids setting, so they always agree on wide_ids.
    if (distant_handle.wide_ids) {
        launch_with_distant_memory_pool<std::uint32_t>(local_handle, distant_handle, test_neuron_id, d_edge_count);
    } else {
        launch_with_distant_memory_pool<SmallNeuronIdType>(local_handle, distant_handle, test_neuron_id, d_edge_count);
    }
}

#endif // RELEARN_CUDA_ENABLED
