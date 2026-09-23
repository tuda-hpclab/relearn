/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "network_graph/NetworkGraphGPU.h"
#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/CudaTypes.cuh"
#include "cuda/input/Handle.h"
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/network_graph/Views.cuh"
#include "cuda/util/Util.cuh"
#include "cuda/wrapper/StreamWrapper.cuh"
#include "neurons/enums/FiredStatus.h"
#include "util/BinarySearch.cuh"
#include "util/RelearnException.h"
#include "util/Timers.h"

#include <cpp-utility/Cast.hpp>

#include <cmath>
#include <variant>

struct NeuronIDsFireInformation {
    const int* d_incoming_displ;
    const neuron_id_type* d_incoming_neuron_ids;
};

struct LocalVectorFireInformation {
    const FiredStatus* fired{};
};

struct NoAuxiliary { };

__global__ void build_keys(const int* incoming_displ,
                           const std::uint32_t* incoming_neuron_ids,
                           uint64_t* keys,
                           int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int rank = 0;
        while (idx >= incoming_displ[rank + 1])
            ++rank;
        keys[idx] = (static_cast<uint64_t>(rank) << 32) | static_cast<uint64_t>(incoming_neuron_ids[idx]);
    }
}

/**
 * Result of fire_to_partitioned_spikes(): the number of fired neurons, the per-rank displacement
 * array and flat neuron-ID array it was partitioned into, and (for variants that build fresh
 * arrays rather than reusing existing device storage) the optionals owning that storage --
 * rank_displ/neuron_ids are non-owning views into owned_rank_displ/owned_neuron_ids whenever the
 * latter are engaged.
 */
struct PartitionedSpikeInfo {
    int size;
    const int* rank_displ;
    const neuron_id_type* neuron_ids;
    std::optional<LazySyncedArray<int>> owned_rank_displ;
    std::optional<DeviceArray<neuron_id_type>> owned_neuron_ids;
};

template <typename Fire>
PartitionedSpikeInfo fire_to_partitioned_spikes(const Fire& fire, const mpi_rank_type my_rank, const mpi_rank_type number_ranks, const neuron_id_type number_neurons) {
    RelearnException::fail("Not suppoertd");
}

template <>
PartitionedSpikeInfo fire_to_partitioned_spikes(const NeuronIDsFireInformation& fire, const mpi_rank_type my_rank, const mpi_rank_type number_ranks, const neuron_id_type number_neurons) {
    if (fire.d_incoming_displ == nullptr) {
        return { 0, nullptr, nullptr, std::nullopt, std::nullopt };
    }
    const auto* size_ptr = fire.d_incoming_displ + number_ranks;
    int size;
    cudaMemcpy_to_host_bridge(&size, size_ptr, sizeof(int));
    return PartitionedSpikeInfo{ size, fire.d_incoming_displ, fire.d_incoming_neuron_ids, std::nullopt, std::nullopt };
}

template <>
PartitionedSpikeInfo fire_to_partitioned_spikes(const LocalVectorFireInformation& fire, const mpi_rank_type my_rank, const mpi_rank_type number_ranks, const neuron_id_type number_neurons) {
    DeviceArray<int> keep(number_neurons);
    DeviceArray<int> scan(number_neurons);
    const auto* keep_ptr = keep.device_ptr();

    thrust::transform(thrust::device_pointer_cast(fire.fired), thrust::device_pointer_cast(fire.fired + number_neurons), thrust::device_pointer_cast(keep.device_ptr()),
                      [] __device__(auto f) { return f == FiredStatus::Fired ? 1 : 0; });
    const auto number_spikes = thrust::reduce(thrust::device_pointer_cast(keep.device_ptr()), thrust::device_pointer_cast(keep.device_ptr() + number_neurons), 0);
    DeviceArray<neuron_id_type> indices(number_spikes);

    thrust::exclusive_scan(thrust::device_pointer_cast(keep.device_ptr()), thrust::device_pointer_cast(keep.device_ptr() + number_neurons), thrust::device_pointer_cast(scan.device_ptr()));

    auto* indices_ptr = indices.device_ptr();
    auto* scan_ptr = scan.device_ptr();
    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator((int)number_neurons),
        [=] __device__(int i) {
            if (keep_ptr[i] == 1) {
                indices_ptr[scan_ptr[i]] = i;
            }
        });

    // Displacement array needs number_ranks+1 entries; rank_displ[my_rank+1] marks the end of
    // the local rank's spike block. All other ranks have 0 spikes, so their entries stay 0.
    LazySyncedArray<int> rank_displ;
    rank_displ.resize(number_ranks + 1, 0);
    rank_displ[my_rank + 1] = number_spikes;

    return PartitionedSpikeInfo{ number_spikes, rank_displ.get_device_ptr_const(), indices_ptr, std::optional{ std::move(rank_displ) }, std::optional{ std::move(indices) } };
}

std::unique_ptr<set_type> current_set = nullptr;

// Backing storage for build_keys' output. Reused (grown, never shrunk) across calls: unlike
// current_set, keys is fully overwritten (indices [0, size)) by build_keys every call, so it
// needs no clearing before reuse -- only current_set's clear/init cost is unavoidable per call.
std::optional<DeviceArray<std::uint64_t>> current_keys = std::nullopt;

template <typename Fire>
std::optional<set_ref_type> create_set(
    const Fire& fire,
    const std::shared_ptr<StreamWrapper>& stream_wrapper, const mpi_rank_type my_rank, const mpi_rank_type number_ranks, const neuron_id_type number_neurons) {

    const auto& [size, rank_displ_ptr, neuron_ids_ptr, tmp0, tmp1] = fire_to_partitioned_spikes(fire, my_rank, number_ranks, number_neurons);

    if (size == 0) {
        return {};
    }

    const auto stream = get_cuda_stream_from_wrapper(*stream_wrapper);

    if (!current_keys.has_value() || current_keys->size() < static_cast<std::size_t>(size)) {
        current_keys.emplace(static_cast<std::size_t>(size), stream_wrapper);
    }
    auto* keys_ptr = current_keys->device_ptr();

    const auto& [blocks0, threads0] = get_grid_ands_block_size(size, build_keys);

    // Launched on stream_wrapper without a following sync -- time via CUDA events (resolved later,
    // at the outer CALC_ACTIVITY_INPUT sync point in NeuronModel.cpp) instead of a blocking
    // Timers::start/stop, which would only capture launch overhead.
    auto* gpu_timer = cuda_start_gpu_timer(TimerRegion::CREATE_SET, *stream_wrapper);
    build_keys<<<blocks0, threads0, 0, stream>>>(
        rank_displ_ptr, neuron_ids_ptr, keys_ptr, size);
    cuda_stop_gpu_timer(gpu_timer, *stream_wrapper);
    // Sizing the set's capacity to exactly `size` (a 100% load factor) is a worst case for
    // double_hashing's open addressing: expected probe length grows like 1/(1 - load_factor), so
    // insertion time blows up superlinearly (not just proportionally) as `size` grows. Give it
    // headroom by computing the padded capacity ourselves and using the raw-capacity constructor
    // -- NOT cuco's (n, desired_load_factor, ...) convenience overload, whose own
    // make_valid_extent() internally casts through `long double` (cuco/detail/extent/extent.inl),
    // which nvcc rejects as an error in device code under -Werror all-warnings.
    constexpr auto set_load_factor = 0.5;
    const auto capacity = static_cast<std::size_t>(std::ceil(static_cast<double>(size) / set_load_factor));
    current_set = std::make_unique<set_type>(
        capacity, cuco::empty_key<std::uint64_t>(std::numeric_limits<std::uint64_t>::max()));
    // Async, matching the "no sync" contract create_auxiliary()'s caller already documents --
    // correctness relies on every consumer of the returned set_ref (the synaptic_kernel launch)
    // running on this same stream, so it's ordered after this insert without a host-side sync.
    current_set->insert_async(keys_ptr, keys_ptr + size, stream);
    auto set_ref = current_set->ref(cuco::contains);
    return std::move(set_ref);
}

void release_synaptic_activity_set() {
    // Both are lazily-allocated statics that must be freed before the CUDA driver starts shutting
    // down -- otherwise their device-memory frees at static-destruction time throw and abort the
    // process (same reasoning that already applied to current_set alone).
    current_set.reset();
    current_keys.reset();
}

struct BinarySearchSpike {

    template <typename Fire, typename Net, typename Aux>
    __device__ static bool contains(const Fire& fire_information, Net&, const Aux& aux, const int mpi_rank,
                                    const CudaConfig::number_neurons_type neuron_id) {

        if constexpr (std::is_same_v<Fire, NeuronIDsFireInformation>) {
            const auto& d_incoming_displ = fire_information.d_incoming_displ;
            const auto& d_incoming_neuron_ids = fire_information.d_incoming_neuron_ids;

            const auto begin = d_incoming_displ[mpi_rank];
            const auto end = d_incoming_displ[mpi_rank + 1];
            RELEARN_DEVICE_CUDA_CHECK(end >= begin, "BinarySearchSpike::contains: end >= begin not fulfilled");
            const auto size = end - begin;
            if (size == 0) {
                return false;
            }

            return binary_search(begin, end, d_incoming_neuron_ids, neuron_id) != std::numeric_limits<std::size_t>::max();
        } else {
            RELEARN_DEVICE_CUDA_CHECK(false, "BinarySearchSpike::contains: Invalid fire type");
        }
    }

    template <typename Fire>
    static NoAuxiliary create_auxiliary(Fire& fire_information, const std::shared_ptr<StreamWrapper>& stream_wrapper, const mpi_rank_type my_rank, const mpi_rank_type number_ranks, const neuron_id_type number_neurons) {
        return NoAuxiliary{};
    }
};

struct LocalVectorLookupSpike {

    template <typename Fire, typename Net, typename Aux>
    __device__ bool contains(const Fire& info, const Net&, const Aux&, int, uint32_t id) const {
        if constexpr (std::is_same_v<Fire, LocalVectorFireInformation>) {
            return info.fired[id] == FiredStatus::Fired;

        } else {
            RELEARN_DEVICE_CUDA_CHECK(false, "LocalVectorLookupSpike::contains: Invalid fire type");
        }
    }

    template <typename Fire>
    static NoAuxiliary create_auxiliary(Fire& fire_information, const std::shared_ptr<StreamWrapper>& stream_wrapper, const mpi_rank_type my_rank, const mpi_rank_type number_ranks, const neuron_id_type number_neurons) {
        return NoAuxiliary{};
    }
};

struct SetAuxiliary {
    const std::optional<set_ref_type> set;
};

struct ContainsSetSpike {
    template <typename Fire, typename Net, typename Aux>
    __device__ static bool contains(const Fire& fire_information, const Net& net, const Aux& aux, const int mpi_rank,
                                    const std::uint32_t neuron_id) {

        if constexpr (std::is_same_v<Aux, SetAuxiliary>) {
            if (!aux.set.has_value()) {
                return false;
            }

            const auto key = (static_cast<uint64_t>(mpi_rank) << 32) | static_cast<uint64_t>(neuron_id);
            const auto x = aux.set.value().contains(key);
            return x;
        } else {
            RELEARN_DEVICE_CUDA_CHECK(false, "ContainsSetSpike::contains: Invalid fire type");
        }
    }

    template <typename Fire>
    static SetAuxiliary create_auxiliary(Fire& fire_information, const std::shared_ptr<StreamWrapper>& stream_wrapper, const mpi_rank_type my_rank, const mpi_rank_type number_ranks, const neuron_id_type number_neurons) {
        // No host Timers wrap here: create_set() only enqueues async work on stream_wrapper (no
        // sync), so a host Timers::start/stop would close long before the GPU actually finishes and
        // could never correctly bound the CUDA-event-timed build_keys kernel inside it.
        auto set = create_set(fire_information, stream_wrapper, my_rank, number_ranks, number_neurons);
        return SetAuxiliary{ std::move(set) };
    }
};

template <typename SpikePolicy, typename NetworkPolicy, typename FireInfo, typename Auxiliary>
__global__ void synaptic_kernel(int first, int last, CudaConfig::input_type* d_input, FireInfo fire_info, NetworkPolicy net,
                                SpikePolicy spikes, int my_rank, Auxiliary auxiliary, bool only_local, CudaConfig::input_type synapse_conductance) {
    int neuron_id = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= last)
        return;

    d_input[neuron_id] = 0;

    auto it = net.edge_begin(neuron_id);
    const auto end = net.edge_end(neuron_id);
    while (it != end) {
        const auto [src, rank, w] = it.next();
        if (rank != my_rank && only_local || rank == my_rank && !only_local || !spikes.contains(fire_info, net, auxiliary, rank, src))
            continue;
        d_input[neuron_id] += synapse_conductance * w;
    }
}

// Push-style: iterate over fired source neurons, push weight to outgoing local targets via atomicAdd.
// d_input must be zeroed before launch (done in launch() for IterationMode::Spikes).
template <typename SpikePolicy, typename EdgesLocal, typename FireInfo, typename Auxiliary>
__launch_bounds__(32)
    __global__ void synaptic_spikes_kernel(int first, int last, CudaConfig::input_type* d_input, FireInfo fire_info,
                                           EdgesLocal outgoing_local, SpikePolicy spikes, int my_rank, Auxiliary auxiliary, const CudaConfig::input_type synapse_conductance) {
    int neuron_id = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= last)
        return;

    if (!spikes.contains(fire_info, outgoing_local, auxiliary, my_rank, neuron_id))
        return;

    auto it = outgoing_local.edge_begin(neuron_id);
    const auto end = outgoing_local.edge_end(neuron_id);
    while (!(it == end)) {
        const auto [tgt, rank, w] = it.next();
        if (rank != my_rank)
            continue;
        atomicAdd(&d_input[tgt], static_cast<CudaConfig::input_type>(w) * synapse_conductance);
    }
}

std::variant<LocalVectorLookupSpike, ContainsSetSpike, BinarySearchSpike> get_spike(const LaunchConfig& config, const LaunchHandles& handles) {
    switch (config.spike) {
    case SpikeMode::LocalVector: {
        return LocalVectorLookupSpike{};
    }
    case SpikeMode::Set: {
        return ContainsSetSpike{};
    }
    case SpikeMode::BinarySearch: {
        return BinarySearchSpike{};
    }
    }
    RelearnException::fail("Invaild spike format");
}

std::variant<LocalVectorFireInformation, NeuronIDsFireInformation> get_fire(const LaunchConfig& config, const LaunchHandles& handles) {
    switch (config.fire_information) {
    case FireInformation::LocalVector: {
        const auto fire_handle = *static_cast<const FireStatusLocalVectorHandle*>(handles.fire_information);
        return LocalVectorFireInformation{ fire_handle.fired };
    }
    case FireInformation::NeuronIDs: {
        const auto fire_handle = *static_cast<const FireStatusCommunicatorUncompressedHandle*>(handles.fire_information);
        return NeuronIDsFireInformation{ fire_handle.incoming_displ, fire_handle.incoming_neuron_ids };
    }
    }
    RelearnException::fail("Invaild fire information");
}

template <typename Spike, typename Net, typename Fire>
std::optional<EventWrapper> start_spikes_kernel(const Spike& spike, const Fire& fire, const Net& edges, const CudaConfig::number_neurons_type first,
                                                CudaConfig::number_neurons_type last,
                                                CudaConfig::input_type* d_input,
                                                const std::shared_ptr<StreamWrapper>& stream_wrapper,
                                                const CudaConfig::mpi_rank_type my_rank, CudaConfig::input_type synapse_conductance, bool local) {

    auto aux = spike.create_auxiliary(fire, stream_wrapper, my_rank, edges.number_ranks, edges.number_neurons);
    using Aux = std::decay_t<decltype(aux)>;

    if (last - first == 0) {
        return {};
    }

    const auto& [blocks, threads] = get_grid_ands_block_size(last - first, synaptic_spikes_kernel<Spike, Net, Fire, Aux>);

    // Launched on stream_wrapper without a following CPU sync -- time via CUDA events (resolved
    // later, e.g. right after the next cudaDeviceSynchronize) rather than a blocking Timers::start/stop.
    const auto region = local ? TimerRegion::CUDA_UPDATE_SYNAPTIC_EQ_WEIGHTED_LOCAL_INPUT_KERNEL : TimerRegion::CUDA_UPDATE_SYNAPTIC_EQ_WEIGHTED_DISTANT_INPUT_KERNEL;
    auto* gpu_timer = cuda_start_gpu_timer(region, *stream_wrapper);
    synaptic_spikes_kernel<Spike, Net, Fire, Aux>
        <<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream_wrapper)>>>(first, last, d_input, fire, edges, spike, my_rank, aux, synapse_conductance);
    cuda_stop_gpu_timer(gpu_timer, *stream_wrapper);

    EventWrapper kernel_done_event{ stream_wrapper };
    record_event(kernel_done_event);
    return kernel_done_event;
}

template <typename Spike, typename Net, typename Fire>
std::optional<EventWrapper> start_kernel(const Spike& spike, const Fire& fire, const Net& edges, const CudaConfig::number_neurons_type first,
                                         CudaConfig::number_neurons_type last,
                                         CudaConfig::input_type* d_input,
                                         const std::shared_ptr<StreamWrapper>& stream_wrapper,
                                         const CudaConfig::mpi_rank_type my_rank, bool local, CudaConfig::input_type synapse_conductance) {

    auto aux = spike.create_auxiliary(fire, stream_wrapper, my_rank, edges.number_ranks, edges.number_neurons);
    using Aux = std::decay_t<decltype(aux)>;

    const auto& [blocks, threads] = get_grid_ands_block_size(last - first + 1, synaptic_kernel<Spike, Net, Fire, Aux>);

    // Launched on stream_wrapper without a following CPU sync -- time via CUDA events (resolved
    // later, e.g. right after the next cudaDeviceSynchronize) rather than a blocking Timers::start/stop.
    const auto region = local ? TimerRegion::CUDA_UPDATE_SYNAPTIC_EQ_WEIGHTED_LOCAL_INPUT_KERNEL : TimerRegion::CUDA_UPDATE_SYNAPTIC_EQ_WEIGHTED_DISTANT_INPUT_KERNEL;
    auto* gpu_timer = cuda_start_gpu_timer(region, *stream_wrapper);
    synaptic_kernel<Spike, Net, Fire, Aux>
        <<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream_wrapper)>>>(first, last, d_input, fire, edges, spike, my_rank, aux, local, synapse_conductance);
    cuda_stop_gpu_timer(gpu_timer, *stream_wrapper);

    EventWrapper kernel_done_event{ stream_wrapper };
    record_event(kernel_done_event);
    return kernel_done_event;
}

std::optional<EventWrapper> launch(const CudaConfig::number_neurons_type first,
                                   CudaConfig::number_neurons_type last,
                                   CudaConfig::input_type* d_input,
                                   const LaunchConfig& config,
                                   const LaunchHandles& handles,
                                   const std::shared_ptr<StreamWrapper>& stream_wrapper,
                                   const CudaConfig::mpi_rank_type my_rank, bool local, bool local_distant_helper, CudaConfig::input_type synapse_conductance) {

    // LocalVectorLookupSpike::contains and BinarySearchSpike::contains each only support one
    // FireInformation (see their `if constexpr` branches below) and abort the whole process via
    // RELEARN_DEVICE_CUDA_CHECK(false, ...) on the device otherwise -- catch the invalid pairing
    // here instead, before any GPU work is launched, so it fails as a normal, catchable exception.
    RelearnException::check(config.spike != SpikeMode::LocalVector || config.fire_information == FireInformation::LocalVector,
                            "launch: SpikeMode::LocalVector only supports FireInformation::LocalVector");
    RelearnException::check(config.spike != SpikeMode::BinarySearch || config.fire_information == FireInformation::NeuronIDs,
                            "launch: SpikeMode::BinarySearch only supports FireInformation::NeuronIDs");

    const auto _spike = get_spike(config, handles);
    const auto _fire = get_fire(config, handles);

    if (config.iteration == IterationMode::Spikes) {
        cudaMemsetAsync_bridge(d_input + first, 0, (last - first) * sizeof(CudaConfig::input_type), *stream_wrapper);

        return std::visit([&stream_wrapper, synapse_conductance, first, config, last, d_input, my_rank, &handles, local, local_distant_helper](auto& spike, auto& fire) {
            using Spike = std::decay_t<decltype(spike)>;
            using Fire = std::decay_t<decltype(fire)>;

            auto* impl = local && !local_distant_helper ? handles.network.outgoing_local_handle.view_impl : handles.network.outgoing_distant_handle.view_impl;
            auto layout = local && !local_distant_helper ? handles.network.outgoing_local_handle.layout_type : handles.network.outgoing_distant_handle.layout_type;
            auto wide_ids = local && !local_distant_helper ? handles.network.outgoing_local_handle.wide_ids : handles.network.outgoing_distant_handle.wide_ids;

            if (layout == LayoutType::MemoryPool && wide_ids) {
                auto* view = static_cast<MemoryPoolView<std::uint32_t>*>(impl);
                const bool hr = view->other_ranks != nullptr;
                const bool hw = view->weights != nullptr;
                if (!hr && hw)
                    return start_spikes_kernel<Spike, MemoryPoolViewTyped<false, true, std::uint32_t>, Fire>(
                        spike, fire, MemoryPoolViewTyped<false, true, std::uint32_t>{ *view }, first, last, d_input, stream_wrapper, my_rank, synapse_conductance, local && !local_distant_helper);
                else if (!hr && !hw)
                    return start_spikes_kernel<Spike, MemoryPoolViewTyped<false, false, std::uint32_t>, Fire>(
                        spike, fire, MemoryPoolViewTyped<false, false, std::uint32_t>{ *view }, first, last, d_input, stream_wrapper, my_rank, synapse_conductance, local && !local_distant_helper);
                else if (hr && hw)
                    return start_spikes_kernel<Spike, MemoryPoolViewTyped<true, true, std::uint32_t>, Fire>(
                        spike, fire, MemoryPoolViewTyped<true, true, std::uint32_t>{ *view }, first, last, d_input, stream_wrapper, my_rank, synapse_conductance, local && !local_distant_helper);
                else
                    return start_spikes_kernel<Spike, MemoryPoolViewTyped<true, false, std::uint32_t>, Fire>(
                        spike, fire, MemoryPoolViewTyped<true, false, std::uint32_t>{ *view }, first, last, d_input, stream_wrapper, my_rank, synapse_conductance, local && !local_distant_helper);
            } else if (layout == LayoutType::MemoryPool) {
                auto* view = static_cast<MemoryPoolView<SmallNeuronIdType>*>(impl);
                const bool hr = view->other_ranks != nullptr;
                const bool hw = view->weights != nullptr;
                if (!hr && hw)
                    return start_spikes_kernel<Spike, MemoryPoolViewTyped<false, true, SmallNeuronIdType>, Fire>(
                        spike, fire, MemoryPoolViewTyped<false, true, SmallNeuronIdType>{ *view }, first, last, d_input, stream_wrapper, my_rank, synapse_conductance, local && !local_distant_helper);
                else if (!hr && !hw)
                    return start_spikes_kernel<Spike, MemoryPoolViewTyped<false, false, SmallNeuronIdType>, Fire>(
                        spike, fire, MemoryPoolViewTyped<false, false, SmallNeuronIdType>{ *view }, first, last, d_input, stream_wrapper, my_rank, synapse_conductance, local && !local_distant_helper);
                else if (hr && hw)
                    return start_spikes_kernel<Spike, MemoryPoolViewTyped<true, true, SmallNeuronIdType>, Fire>(
                        spike, fire, MemoryPoolViewTyped<true, true, SmallNeuronIdType>{ *view }, first, last, d_input, stream_wrapper, my_rank, synapse_conductance, local && !local_distant_helper);
                else
                    return start_spikes_kernel<Spike, MemoryPoolViewTyped<true, false, SmallNeuronIdType>, Fire>(
                        spike, fire, MemoryPoolViewTyped<true, false, SmallNeuronIdType>{ *view }, first, last, d_input, stream_wrapper, my_rank, synapse_conductance, local && !local_distant_helper);
            } else if (layout == LayoutType::Dummy) {
                return launch(first, last, d_input, config, handles, stream_wrapper, my_rank, true, true, synapse_conductance);
            } else {
                RelearnException::fail("Invalid layout combinations3");
            }
        },
                          _spike, _fire);
    }

    return std::visit([&stream_wrapper, synapse_conductance, first, last, d_input, config, my_rank, &handles, local, local_distant_helper](auto& spike, auto& fire) {
        using Spike = std::decay_t<decltype(spike)>;
        using Fire = std::decay_t<decltype(fire)>;

        auto* impl = local && !local_distant_helper ? handles.network.incoming_local_handle.view_impl : handles.network.incoming_distant_handle.view_impl;
        auto layout = local && !local_distant_helper ? handles.network.incoming_local_handle.layout_type : handles.network.incoming_distant_handle.layout_type;
        auto wide_ids = local && !local_distant_helper ? handles.network.incoming_local_handle.wide_ids : handles.network.incoming_distant_handle.wide_ids;

        if (layout == LayoutType::MemoryPool && wide_ids) {
            return start_kernel<Spike, MemoryPoolView<std::uint32_t>, Fire>(spike, fire,
                                                                            *static_cast<MemoryPoolView<std::uint32_t>*>(impl),
                                                                            first, last, d_input, stream_wrapper, my_rank, local, synapse_conductance);
        } else if (layout == LayoutType::MemoryPool) {
            return start_kernel<Spike, MemoryPoolView<SmallNeuronIdType>, Fire>(spike, fire,
                                                                                *static_cast<MemoryPoolView<SmallNeuronIdType>*>(impl),
                                                                                first, last, d_input, stream_wrapper, my_rank, local, synapse_conductance);
        } else if (layout == LayoutType::Dummy) {
            return launch(first, last, d_input, config, handles, stream_wrapper, my_rank, true, true, synapse_conductance);
        } else {
            RelearnException::fail("Invalid layout combinations3");
        }
    },
                      _spike, _fire);
}