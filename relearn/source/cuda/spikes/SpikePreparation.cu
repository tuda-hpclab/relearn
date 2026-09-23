/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CudaBaseBridgeFunctions.h"
#include "SpikePreparation.h"

#include "network_graph/GPUEdges.h"
#include "network_graph/NetworkHandle.h"
#include "memory/DeviceArray.h"
#include "network_graph/Views.cuh"
#include "util/PrefixSum.cuh"
#include "util/Timers.h"
#include "util/UpperBound.cuh"
#include "util/Util.cuh"
#include "wrapper/EventWrapper.h"
#include "wrapper/StreamWrapper.cuh"

#include <cub/device/device_segmented_radix_sort.cuh>

template <typename Net>
__global__ void prepare_sending_spikes_kernel(const FiredStatus* d_fired, Net network, bool* flags, const CudaConfig::mpi_rank_type my_rank, std::size_t* network_displ) {

    const auto thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    const auto network_size = network_displ[network.number_neurons];
    if (thread_id >= network_size) {
        return;
    }

    auto my_neuron_id = upper_bound<std::size_t>(0,
                                                 network.number_neurons,
                                                 network_displ,
                                                 thread_id);
    const auto offset = thread_id - network_displ[my_neuron_id];

    const auto rank = network.get_other_rank(my_neuron_id, offset);
    if (rank == my_rank) {
        flags[thread_id] = 0;
        return;
    }

    if (d_fired[my_neuron_id] != FiredStatus::Fired) {
        flags[thread_id] = 0;
        return;
    }

    flags[thread_id] = 1;
}

__global__ void extractSegmentCounts(const bool* flags, const std::size_t* d_scan, const std::size_t* network_displ, int* d_segment_counts, const CudaConfig::mpi_rank_type number_ranks) {
    const auto thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (thread_id >= number_ranks) {
        return;
    }

    const auto start = network_displ[thread_id];
    const auto end = network_displ[thread_id + 1];

    if (end > start) {
        d_segment_counts[thread_id] = d_scan[end - 1] + flags[end - 1];
    } else {
        d_segment_counts[thread_id] = 0;
    }
}

template <typename Network>
__global__ void scatter_kernel(const bool* flags, const std::size_t* offset_on_rank, const std::size_t* network_displ, const Network& network, const int* new_rank_displ, CudaConfig::number_neurons_type* out_buffer, const int old_size) {
    const auto thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (thread_id >= old_size) {
        return;
    }

    if (!flags[thread_id]) {
        return;
    }

    auto my_neuron_id = upper_bound<std::size_t>(0,
                                                 network.number_neurons,
                                                 network_displ,
                                                 thread_id);
    const auto offset = thread_id - network_displ[my_neuron_id];

    const auto rank = network.get_other_rank(my_neuron_id, offset);
    const auto new_offset = offset_on_rank[rank];

    out_buffer[new_rank_displ[rank] + new_offset] = my_neuron_id;
}

template <typename Net>
__global__ void count_network_size_kernel(Net edges, std::size_t* sizes) {
    const auto thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (thread_id >= edges.number_neurons) {
        return;
    }

    sizes[thread_id] = edges.size(thread_id);
}

template <typename Net>
__global__ void count_spikes_per_rank_kernel(
    const FiredStatus* fired,
    Net network,
    const std::size_t* network_displ,
    int* spikes_per_rank,
    int my_rank) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const auto network_size = network_displ[network.number_neurons];

    if (tid >= network_size)
        return;

    auto neuron = upper_bound<std::size_t>(
        0,
        network.number_neurons,
        network_displ,
        tid);

    const auto offset = tid - network_displ[neuron];

    const auto rank = network.get_other_rank(neuron, offset);

    if (rank == my_rank)
        return;

    if (fired[neuron] != FiredStatus::Fired)
        return;

    atomicAdd(&spikes_per_rank[rank], 1);
}

template <typename Net>
__global__ void fill_spikes_kernel(
    const FiredStatus* fired,
    Net network,
    const std::size_t* network_displ,
    int* write_offsets,
    neuron_id_type* out,
    int my_rank) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const auto network_size = network_displ[network.number_neurons];

    if (tid >= network_size)
        return;

    auto neuron = upper_bound<std::size_t>(
        0,
        network.number_neurons,
        network_displ,
        tid);

    const auto offset = tid - network_displ[neuron];

    const auto rank = network.get_other_rank(neuron, offset);

    if (rank == my_rank)
        return;

    if (fired[neuron] != FiredStatus::Fired)
        return;

    const int pos = atomicAdd(&write_offsets[rank], 1);

    out[pos] = neuron;
}

template <typename OutgoingDistantEdges>
PreparedSpikes prepare_spikes_atomic_without_sorting_internal(const FiredStatus* d_fired, OutgoingDistantEdges edges, const CudaConfig::mpi_rank_type my_rank, bool sort_spikes, std::shared_ptr<StreamWrapper> sort_stream) {

    // All three kernels below run on the (default/null) CUDA stream with no sync in between --
    // interleaved with prefixSum/thrust calls that make a plain Timers::start/stop discipline
    // fragile here, so time each via CUDA events instead, resolved together at the full-device
    // sync just below (line ~211).
    static auto default_stream = StreamWrapper::default_stream();

    static DeviceArray<std::size_t> sizes(edges.number_neurons);
    const auto& [blocks0, threads0] = get_grid_ands_block_size(edges.number_neurons, count_network_size_kernel<OutgoingDistantEdges>);
    auto* count_network_size_timer = cuda_start_gpu_timer(TimerRegion::CUDA_SPIKE_COUNT_NETWORK_SIZE_KERNEL, *default_stream);
    count_network_size_kernel<<<blocks0, threads0>>>(edges, sizes.device_ptr());
    cuda_stop_gpu_timer(count_network_size_timer, *default_stream);

    static DeviceArray<char> temp_storage(0);
    static DeviceArray<std::size_t> network_displ(0);
    prefixSum<>(sizes.device_ptr(), sizes.size(), network_displ, temp_storage);

    static DeviceArray<int> spikes_per_rank{ edges.number_ranks };
    cudaMemset(spikes_per_rank.device_ptr(), 0, edges.number_ranks * sizeof(int));

    std::size_t network_size;
    cudaMemcpy_to_host_bridge(&network_size, network_displ.device_ptr() + edges.number_neurons, sizeof(std::size_t));
    const auto& [blocks1, threads1] = get_grid_ands_block_size(network_size, count_spikes_per_rank_kernel<OutgoingDistantEdges>);

    auto* count_per_rank_timer = cuda_start_gpu_timer(TimerRegion::CUDA_SPIKE_COUNT_PER_RANK_KERNEL, *default_stream);
    count_spikes_per_rank_kernel<<<blocks1, threads1>>>(d_fired, edges, network_displ.device_ptr(), spikes_per_rank.device_ptr(), my_rank);
    cuda_stop_gpu_timer(count_per_rank_timer, *default_stream);

    const auto rank_displ = prefixSum<int, int>(spikes_per_rank);
    const auto h_sizes = spikes_per_rank.get_device_data();

    int number_spikes;
    cudaMemcpy_to_host_bridge(&number_spikes, rank_displ.device_ptr() + edges.number_ranks, sizeof(int));
    cudaDeviceSynchronize_bridge();
    // Resolves count_network_size_kernel and count_spikes_per_rank_kernel's GPU timers -- free here
    // since the device is already caught up above.
    cuda_resolve_gpu_timers();

    // TODO Maybe reuse memory here
    auto fired_neurons = DeviceArray<neuron_id_type>(number_spikes);

    const auto& [blocks2, threads2] = get_grid_ands_block_size(number_spikes, fill_spikes_kernel<OutgoingDistantEdges>);
    auto* fill_spikes_timer = cuda_start_gpu_timer(TimerRegion::CUDA_SPIKE_FILL_KERNEL, *default_stream);
    fill_spikes_kernel<<<blocks2, threads2>>>(d_fired, edges, network_displ.device_ptr(), rank_displ.device_ptr(), fired_neurons.device_ptr(), my_rank);
    cuda_stop_gpu_timer(fill_spikes_timer, *default_stream);

    cudaDeviceSynchronize();
    cuda_resolve_gpu_timers();

    //
    // SEGMENTED SORT
    //
    if (sort_spikes) {
        // TODO Maybe reuse memory here
        DeviceArray<neuron_id_type> sorted(number_spikes, sort_stream);

        // rank_displ was used as write-cursor by fill_spikes_kernel (atomicAdd), so it is
        // now corrupted. Recompute before passing to the segmented sort.
        // TODO Reuse memory here definitely
        static DeviceArray<char> temp_storage(0, sort_stream);
        static DeviceArray<int> rank_displ_sort(0, sort_stream);
        static DeviceArray<char> temp_storage2(0, sort_stream);

        // Make sort_stream wait for fill_spikes_kernel on the default stream,
        // then proceed independently so the default stream can continue.
        static auto fill_done = EventWrapper(StreamWrapper::default_stream());
        record_event(fill_done);
        fill_done.wait_for_event(sort_stream);
        auto cuda_stream = get_cuda_stream_from_wrapper(*sort_stream);

        prefixSum<int, int>(spikes_per_rank.device_ptr(), spikes_per_rank.size(), rank_displ_sort, temp_storage, cuda_stream);

        size_t temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr,
            temp_storage_bytes,
            fired_neurons.device_ptr(),
            sorted.device_ptr(),
            number_spikes,
            edges.number_ranks,
            rank_displ_sort.device_ptr(),
            rank_displ_sort.device_ptr() + 1,
            0, sizeof(neuron_id_type) * 8, cuda_stream);

        if (temp_storage2.size() < temp_storage_bytes) {
            temp_storage2 = DeviceArray<char>(temp_storage_bytes, sort_stream);
        }

        cub::DeviceSegmentedRadixSort::SortKeys(
            temp_storage2.device_ptr(),
            temp_storage_bytes,
            fired_neurons.device_ptr(),
            sorted.device_ptr(),
            number_spikes,
            edges.number_ranks,
            rank_displ_sort.device_ptr(),
            rank_displ_sort.device_ptr() + 1,
            0, sizeof(neuron_id_type) * 8, cuda_stream);

        return PreparedSpikes{ std::move(h_sizes), std::move(sorted) };
    }

    return PreparedSpikes{ std::move(h_sizes), std::move(fired_neurons) };
}

PreparedSpikes prepare_spikes(const FiredStatusHandle fired_handle, NetworkHandle network_view, const NeuronsExtraInfoGPUHandleConst info_handle, bool sort_spikes, const std::shared_ptr<StreamWrapper>& sort_stream) {
    if (network_view.outgoing_local_handle.layout_type == LayoutType::MemoryPool) {
        if (network_view.outgoing_distant_handle.wide_ids) {
            return prepare_spikes_atomic_without_sorting_internal<MemoryPoolView<std::uint32_t>>(fired_handle.fired,
                                                                                                 *static_cast<MemoryPoolView<std::uint32_t>*>(network_view.outgoing_distant_handle.view_impl),
                                                                                                 info_handle.my_rank, sort_spikes, sort_stream);
        }
        return prepare_spikes_atomic_without_sorting_internal<MemoryPoolView<SmallNeuronIdType>>(fired_handle.fired,
                                                                                                 *static_cast<MemoryPoolView<SmallNeuronIdType>*>(network_view.outgoing_distant_handle.view_impl),
                                                                                                 info_handle.my_rank, sort_spikes, sort_stream);
    } else if (network_view.outgoing_local_handle.layout_type == LayoutType::Dummy) {
        RELEARN_CUDA_CHECK(false, "prepare_spikes: Dummy layout not supported for spike preparation");
    } else {
        RelearnException::fail("Invalid layout combinations3");
    }
}
