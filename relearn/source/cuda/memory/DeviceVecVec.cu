/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "DeviceVecVec.h"

#include "network_graph/GPUEdges.h"
#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceVecVec.cuh"
#include "thrust/binary_search.h"
#include "thrust/device_ptr.h"
#include "thrust/reduce.h"
#include "util/PrefixSum.cuh"
#include "util/Timers.h"
#include "util/Util.cuh"

#include <cpp-utility/Cast.hpp>

#include <thrust/detail/extrema.inl>
#include <thrust/detail/sort.inl>

#include <cuda/functional>

// Rounds a block size up to a multiple of `alignment`. Every block in a SharedBlockPool is
// reinterpreted as a Chunk<T> (which embeds T*/Chunk<T>* pointers requiring 8-byte alignment),
// so block_bytes must itself be a multiple of that alignment -- otherwise blocks at odd indices
// land at addresses like data + 1*block_bytes that aren't 8-byte aligned, and the pointer stores
// in get_new_chunk() fault with a CUDA "misaligned address" error. This bit when sizeof(T) doesn't
// divide evenly into the alignment (e.g. the 3-byte SmallNeuronIdType).
static std::size_t round_up_to_alignment(std::size_t size, std::size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

template <typename T>
__global__ void get_average_filled_main_chunk_kernel(const DynamicVecVecView<T>* view, const std::size_t number_main_chunks, float* out) {
    const auto neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_id >= number_main_chunks) {
        return;
    }

    const auto* chunk = view->chunks + neuron_id;

    out[neuron_id] = chunk->filled / static_cast<float>(chunk->size);
}

template <typename T>
__global__ void get_used_chunks_per_neuron_kernel(const DynamicVecVecView<T>* view, const std::size_t number_main_chunks, std::size_t* count) {
    const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= number_main_chunks) {
        return;
    }

    auto* chunk = view->chunks + thread_id;
    auto counter = 0U;
    while (chunk != nullptr) {
        counter++;
        chunk = chunk->next;
    }

    count[thread_id] = counter;
}

template <typename T>
__global__ void get_memory_pointers(const DynamicVecVecView<T>* view, const std::size_t number_main_chunks, const std::size_t* partition, T** chunk_begin_ptr, std::size_t* chunk_lengths) {
    const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id >= number_main_chunks) {
        return;
    }

    const auto begin = partition[thread_id];
    const auto end = partition[thread_id + 1];

    auto* chunk = view->chunks + thread_id;
    auto counter = 0U;
    while (chunk != nullptr) {
        RELEARN_DEVICE_CUDA_CHECK(counter <= end - begin, "get_memory_pointers: Partition not matching counter");
        chunk_begin_ptr[begin + counter] = chunk->begin;
        chunk_lengths[begin + counter] = chunk->filled;
        counter++;
        chunk = chunk->next;
    }
}

// ---- DynamicVecVecView<T> device methods ----

template <typename T>
__device__ Chunk<T>* DynamicVecVecView<T>::get_new_chunk(Chunk<T>* prev) {
    auto* raw = pool->acquire_block();
    if (raw == nullptr)
        return nullptr;
    // Descriptor lives at the start of the block; data follows immediately after.
    auto* new_chunk = reinterpret_cast<Chunk<T>*>(raw);
    new_chunk->begin = reinterpret_cast<T*>(raw + sizeof(Chunk<T>));
    new_chunk->chunk_id = pool->ptr_to_block_id(raw);
    new_chunk->filled = 0U;
    new_chunk->size = new_chunk_size;
    new_chunk->prev = prev;
    new_chunk->next = nullptr;
    prev->next = new_chunk;
    return new_chunk;
}

template <typename T>
__device__ bool DynamicVecVecView<T>::add(index_type neuron_id, T&& value) {
    // Only one thread is allowed to access neuron_id at the same time! There is no mutex

    auto* cur_chunk = chunks + neuron_id;
    auto siz = 0U;
    while (cur_chunk->next != nullptr) {
        siz += cur_chunk->filled;
        cur_chunk = cur_chunk->next;
    }

    if (cur_chunk->filled == cur_chunk->size) {
        Chunk<T>* new_chunk = get_new_chunk(cur_chunk);
        if (new_chunk == nullptr) {
            return false;
        }
        cur_chunk = new_chunk;
    }
    siz += cur_chunk->filled;

    auto* free_data = cur_chunk->begin + cur_chunk->filled;
    *free_data = std::move(value);
    ++(cur_chunk->filled);
    return true;
}

template <typename T>
__device__ T DynamicVecVecView<T>::get(const index_type neuron_id, const index_type element_idx) const {
    RELEARN_DEVICE_CUDA_CHECK(chunks != nullptr, "DynamicVecVecView::get: DynamicVecVecView is empty");
    const auto* cur_chunk = chunks + neuron_id;
    auto cur_offset = 0U;
    auto chunk_it = 0U;
    while (cur_chunk != nullptr) {
        if (element_idx < cur_offset + cur_chunk->filled) {
            return cur_chunk->begin[element_idx - cur_offset];
        }
        cur_offset += cur_chunk->filled;
        cur_chunk = cur_chunk->next;
        chunk_it++;
    }
    RELEARN_DEVICE_CUDA_CHECK(false, "DynamicVecVecView::get: OOB neuron=%u idx=%u size=%u chunks=%u", neuron_id, element_idx, cur_offset, chunk_it);
    return T{};
}

template <typename T>
__device__ DynamicVecVecView<T>::index_type DynamicVecVecView<T>::get_size(const index_type neuron_id) const {
    RELEARN_DEVICE_CUDA_CHECK(chunks != nullptr, "DynamicVecVecView::get_size: chunks is null");
    const auto* cur_chunk = chunks + neuron_id;
    auto size = 0U;
    while (cur_chunk != nullptr) {
        size += cur_chunk->filled;
        cur_chunk = cur_chunk->next;
    }
    return size;
}

template <typename T>
__device__ void DynamicVecVecView<T>::set(const index_type neuron_id, const index_type element_idx, T value) {
    auto* cur_chunk = chunks + neuron_id;
    auto cur_offset = 0U;
    while (cur_chunk != nullptr) {
        if (element_idx < cur_offset + cur_chunk->filled) {
            cur_chunk->begin[element_idx - cur_offset] = value;
            return;
        }
        cur_offset += cur_chunk->filled;
        cur_chunk = cur_chunk->next;
    }
    RELEARN_DEVICE_CUDA_CHECK(false, "DynamicVecVecView::set: index out of bounds");
}

template <typename T>
__device__ void DynamicVecVecView<T>::pop_back(const index_type neuron_id) {
    auto* cur_chunk = chunks + neuron_id;
    Chunk<T>* last_nonempty = nullptr;
    while (cur_chunk != nullptr) {
        if (cur_chunk->filled > 0)
            last_nonempty = cur_chunk;
        cur_chunk = cur_chunk->next;
    }
    RELEARN_DEVICE_CUDA_CHECK(last_nonempty != nullptr, "DynamicVecVecView::pop_back: pop_back on empty neuron");
    --last_nonempty->filled;

    // Release any trailing empty overflow chunk that followed last_nonempty.
    if (last_nonempty->next != nullptr) {
        auto* trailing = last_nonempty->next;
        pool->release_block(pool->ptr_to_block_id(reinterpret_cast<std::uint8_t*>(trailing)));
        last_nonempty->next = nullptr;
    }

    // If last_nonempty itself is now empty and is an overflow chunk (prev != nullptr),
    // unlink and release it. Main chunks (prev == nullptr) are never released.
    if (last_nonempty->filled == 0 && last_nonempty->prev != nullptr) {
        last_nonempty->prev->next = nullptr;
        pool->release_block(pool->ptr_to_block_id(reinterpret_cast<std::uint8_t*>(last_nonempty)));
    }
}

// ---- DynamicVecVec<T> ----

template <typename T>
void DynamicVecVec<T>::init_main(std::size_t number_elements, std::size_t init_size, SharedBlockPool* pool) {
    // Data elements per overflow block: subtract the embedded Chunk<T> header.
    const auto new_chunk_size = static_cast<std::uint32_t>((pool->get_block_bytes() - sizeof(Chunk<T>)) / sizeof(T));

    total_size = number_elements * init_size;
    main_chunks = number_elements;

    cudaMalloc_bridge(reinterpret_cast<void**>(&data), total_size * sizeof(T));
    cudaMalloc_bridge(reinterpret_cast<void**>(&chunks), number_elements * sizeof(Chunk<T>));

    std::vector<Chunk<T>> h_chunks(number_elements);
    for (std::size_t i = 0; i < number_elements; ++i) {
        h_chunks[i] = Chunk<T>{ data + i * init_size, static_cast<std::uint32_t>(i),
                                0U, static_cast<std::uint32_t>(init_size), nullptr, nullptr };
    }
    cudaMemcpy_to_device_bridge(chunks, h_chunks.data(), number_elements * sizeof(Chunk<T>));

    cudaMalloc_bridge(reinterpret_cast<void**>(&device_class), sizeof(DynamicVecVecView<T>));
    DynamicVecVecView<T> view{};
    view.chunks = chunks;
    view.pool = pool->get_device_pool();
    view.new_chunk_size = new_chunk_size;
    view.main_chunks = static_cast<std::uint32_t>(number_elements);
    cudaMemcpy_to_device_bridge(device_class, &view, sizeof(DynamicVecVecView<T>));
}

template <typename T>
DynamicVecVec<T>::DynamicVecVec(std::size_t number_elements) {
    const auto init_size = CudaConfig::expected_synapses_per_neuron;
    const auto data_elements = std::max(std::size_t{ 1 },
                                        static_cast<std::size_t>(CudaConfig::overflow_chunk_size_factor * init_size));
    const auto block_bytes = round_up_to_alignment(sizeof(Chunk<T>) + data_elements * sizeof(T), alignof(Chunk<T>));
    const auto overflow_blocks = static_cast<std::size_t>(number_elements * CudaConfig::number_overflow_chunks_factor);
    private_pool = std::make_unique<SharedBlockPool>(overflow_blocks, block_bytes);
    init_main(number_elements, init_size, private_pool.get());
}

template <typename T>
DynamicVecVec<T>::DynamicVecVec(std::size_t number_elements, std::size_t init_size, SharedBlockPool* shared_pool) {
    init_main(number_elements, init_size, shared_pool);
}

template <typename T>
__global__ void k_add_from_displ(DynamicVecVecView<T>* view, const T* values,
                                 const std::uint32_t* offsets, const std::uint32_t* neuron_ids,
                                 std::uint32_t n);

template <typename T>
void DynamicVecVec<T>::add_from_map(const std::unordered_map<std::uint32_t, std::vector<T>>& map) {
    if (map.empty())
        return;

    const auto n = static_cast<std::uint32_t>(map.size());

    std::vector<std::uint32_t> h_neuron_ids;
    std::vector<std::uint32_t> h_offsets;
    std::vector<T> h_values;
    h_neuron_ids.reserve(n);
    h_offsets.reserve(n + 1);
    h_offsets.push_back(0);

    for (const auto& [neuron_id, vals] : map) {
        h_neuron_ids.push_back(neuron_id);
        h_values.insert(h_values.end(), vals.begin(), vals.end());
        h_offsets.push_back(static_cast<std::uint32_t>(h_values.size()));
    }

    std::uint32_t* d_neuron_ids{};
    std::uint32_t* d_offsets{};
    T* d_values{};
    cudaMalloc_bridge(reinterpret_cast<void**>(&d_neuron_ids), n * sizeof(std::uint32_t));
    cudaMalloc_bridge(reinterpret_cast<void**>(&d_offsets), (n + 1) * sizeof(std::uint32_t));
    cudaMalloc_bridge(reinterpret_cast<void**>(&d_values), h_values.size() * sizeof(T));

    cudaMemcpy_to_device_bridge(d_neuron_ids, h_neuron_ids.data(), n * sizeof(std::uint32_t));
    cudaMemcpy_to_device_bridge(d_offsets, h_offsets.data(), (n + 1) * sizeof(std::uint32_t));
    cudaMemcpy_to_device_bridge(d_values, h_values.data(), h_values.size() * sizeof(T));

    const auto blocks = (n + 255u) / 256u;
    Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_ADD_FROM_DISPL_KERNEL);
    k_add_from_displ<<<blocks, 256>>>(device_class, d_values, d_offsets, d_neuron_ids, n);
    cudaDeviceSynchronize();
    Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_ADD_FROM_DISPL_KERNEL);

    cudaFree_bridge(d_neuron_ids);
    cudaFree_bridge(d_offsets);
    cudaFree_bridge(d_values);
}

template <typename T>
__global__ void k_add_from_displ(DynamicVecVecView<T>* view,
                                 const T* values,
                                 const std::uint32_t* offsets,
                                 const std::uint32_t* neuron_ids,
                                 std::uint32_t n) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    const auto neuron_id = neuron_ids[tid];
    for (auto i = offsets[tid]; i < offsets[tid + 1]; ++i) {
        T val = values[i];
        (void)view->add(neuron_id, std::move(val));
    }
}

template <typename T>
std::uint64_t DynamicVecVec<T>::get_gpu_memory_footprint() const {
    std::uint64_t fp = total_size * sizeof(T)
                       + main_chunks * sizeof(Chunk<T>)
                       + sizeof(DynamicVecVecView<T>);
    if (private_pool)
        fp += private_pool->get_gpu_memory_footprint();
    return fp;
}

template <typename T>
float DynamicVecVec<T>::get_average_filled_main_chunk() const {
    DeviceArray<float> filled_out(main_chunks);
    const auto [grid_size, block_size] = get_grid_ands_block_size(main_chunks, get_average_filled_main_chunk_kernel<T>);
    Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_AVERAGE_FILLED_KERNEL);
    get_average_filled_main_chunk_kernel<<<grid_size, block_size>>>(get_device_view(), main_chunks, filled_out.device_ptr());
    cudaDeviceSynchronize();
    Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_AVERAGE_FILLED_KERNEL);
    const auto sum = thrust::reduce(thrust::device_pointer_cast(filled_out.device_ptr()), thrust::device_pointer_cast(filled_out.device_ptr() + main_chunks));

    const auto r = sum / static_cast<float>(main_chunks);
    return r;
}

template <typename T>
ChunkUsageStats DynamicVecVec<T>::get_used_chunks_per_neuron() const {
    DeviceArray<std::size_t> counter(main_chunks);
    const auto [grid_size, block_size] = get_grid_ands_block_size(main_chunks, get_used_chunks_per_neuron_kernel<T>);
    Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_USED_CHUNKS_KERNEL);
    get_used_chunks_per_neuron_kernel<<<grid_size, block_size>>>(get_device_view(), main_chunks, counter.device_ptr());
    cudaDeviceSynchronize();
    Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_USED_CHUNKS_KERNEL);

    thrust::sort(thrust::device_pointer_cast(counter.device_ptr()), thrust::device_pointer_cast(counter.device_ptr() + main_chunks));
    std::size_t median;
    cudaMemcpy_to_host_bridge(&median, counter.device_ptr() + main_chunks / 2, sizeof(std::size_t));

    const auto d_begin = thrust::device_pointer_cast(counter.device_ptr());
    const auto d_end = thrust::device_pointer_cast(counter.device_ptr() + main_chunks);

    const std::size_t sum_counter = thrust::reduce(d_begin, d_end);
    const std::size_t max = thrust::reduce(d_begin, d_end, std::size_t{ 0 }, cuda::maximum<std::size_t>());
    const std::size_t count_gt_1 = static_cast<std::size_t>(d_end - thrust::upper_bound(d_begin, d_end, std::size_t{ 1 }));
    const auto ratio_more_than_one_chunk = main_chunks > 0 ? count_gt_1 / static_cast<float>(main_chunks) : 0;
    const float avg = sum_counter / static_cast<float>(main_chunks);
    return { sum_counter, avg, median, max, ratio_more_than_one_chunk };
}

// Counts filled elements (not chunks) for each neuron by following the linked list.
template <typename T>
__global__ void get_filled_elements_per_neuron_kernel(
    const DynamicVecVecView<T>* view,
    const std::size_t number_neurons,
    std::size_t* element_counts) {
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= number_neurons)
        return;
    std::size_t count = 0;
    const Chunk<T>* c = view->chunks + n;
    while (c != nullptr) {
        count += static_cast<std::size_t>(c->filled);
        c = c->next;
    }
    element_counts[n] = count;
}

// Copies each neuron's filled elements to a flat contiguous output buffer.
// offsets[n] gives the start index in flat_output for neuron n.
template <typename T>
__global__ void linearize_to_flat_kernel(
    const DynamicVecVecView<T>* view,
    const std::size_t number_neurons,
    const std::size_t* offsets,
    T* flat_output) {
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= number_neurons)
        return;
    auto out = offsets[n];
    const Chunk<T>* c = view->chunks + n;
    while (c != nullptr) {
        for (std::uint32_t i = 0U; i < c->filled; i++) {
            flat_output[out++] = c->begin[i];
        }
        c = c->next;
    }
}

template <typename T>
std::vector<std::vector<T>> DynamicVecVec<T>::copy_to_host() const {
    // Count filled elements per neuron (one kernel, one sync)
    DeviceArray<std::size_t> elem_counts(main_chunks);
    {
        const auto [gs, bs] = get_grid_ands_block_size(main_chunks, get_filled_elements_per_neuron_kernel<T>);
        Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_FILLED_ELEMENTS_KERNEL);
        get_filled_elements_per_neuron_kernel<<<gs, bs>>>(get_device_view(), main_chunks, elem_counts.device_ptr());
        cudaDeviceSynchronize_bridge();
        Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_FILLED_ELEMENTS_KERNEL);
    }

    // Exclusive prefix sum → offsets[0..N]: offsets[N] = total element count
    const auto offsets_gpu = prefixSum<std::size_t, std::size_t>(elem_counts.device_ptr(), main_chunks);

    std::size_t total_elements{};
    cudaMemcpy_to_host_bridge(&total_elements, offsets_gpu.device_ptr() + main_chunks, sizeof(std::size_t));

    if (total_elements == 0) {
        return std::vector<std::vector<T>>(main_chunks);
    }

    // Linearise all filled data on GPU then download with a single cudaMemcpy.
    // The GPU flat buffer is freed before building the result to avoid holding
    // both simultaneously.
    std::vector<T> h_flat(total_elements);
    {
        DeviceArray<T> flat_gpu(total_elements);
        const auto [gs, bs] = get_grid_ands_block_size(main_chunks, linearize_to_flat_kernel<T>);
        Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_LINEARIZE_KERNEL);
        linearize_to_flat_kernel<<<gs, bs>>>(get_device_view(), main_chunks, offsets_gpu.device_ptr(), flat_gpu.device_ptr());
        cudaDeviceSynchronize_bridge();
        Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_LINEARIZE_KERNEL);
        cudaMemcpy_to_host_bridge(h_flat.data(), flat_gpu.device_ptr(), sizeof(T) * total_elements);
    } // flat_gpu freed here

    const auto h_offsets = offsets_gpu.get_device_data(); // N+1 size_t values, small

    std::vector<std::vector<T>> all_host_data;
    all_host_data.reserve(main_chunks);
    for (std::size_t n = 0; n < main_chunks; n++) {
        all_host_data.emplace_back(h_flat.begin() + h_offsets[n], h_flat.begin() + h_offsets[n + 1]);
    }
    return all_host_data;
}

template <typename T>
void DynamicVecVec<T>::record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint,
                                              const std::string& prefix) const {
    footprint->emplace(prefix + ": filled main", static_cast<std::uint64_t>(get_average_filled_main_chunk() * 100.f));
    const auto [sum_counter, avg, median, max, ratio_gt_1] = get_used_chunks_per_neuron();
    footprint->emplace(prefix + ": chunks per neuron (avg)", static_cast<std::uint64_t>(avg * 100.f));
    footprint->emplace(prefix + ": chunks per neuron (median)", static_cast<std::uint64_t>(median * 100.f));
    footprint->emplace(prefix + ": chunks per neuron (sum)", static_cast<std::uint64_t>(sum_counter * 100.f));
    footprint->emplace(prefix + ": chunks per neuron (max)", static_cast<std::uint64_t>(max * 100.f));
    footprint->emplace(prefix + ": chunks per neuron (>1)", static_cast<std::uint64_t>(ratio_gt_1 * 100.0f));
}

template class DynamicVecVecView<std::uint32_t>;
template class DynamicVecVec<std::uint32_t>;

template class DynamicVecVecView<std::uint16_t>;
template class DynamicVecVec<std::uint16_t>;

template class DynamicVecVecView<std::int16_t>;
template class DynamicVecVec<std::int16_t>;

template class DynamicVecVecView<bool>;
template class DynamicVecVec<bool>;

template class DynamicVecVecView<SmallNeuronIdType>;
template class DynamicVecVec<SmallNeuronIdType>;

template class DynamicVecVecView<signed char>;
template class DynamicVecVec<signed char>;

// ---- DynamicVecVecView<bool> bit-packed specialisation ----
// Booleans are packed 32-per-word into uint32_t storage.
// Chunk::filled and Chunk::size count *bits*; chunk boundaries are always
// on 32-bit word boundaries (enforced by rounding in DynamicVecVec<bool>).

__device__ Chunk<std::uint32_t>* DynamicVecVecView<bool>::get_new_chunk(Chunk<std::uint32_t>* prev) {
    auto* raw = pool->acquire_block();
    if (raw == nullptr)
        return nullptr;
    auto* new_chunk = reinterpret_cast<Chunk<std::uint32_t>*>(raw);
    new_chunk->begin = reinterpret_cast<std::uint32_t*>(raw + sizeof(Chunk<std::uint32_t>));
    new_chunk->chunk_id = pool->ptr_to_block_id(raw);
    new_chunk->filled = 0U;
    new_chunk->size = new_chunk_size; // bits
    new_chunk->prev = prev;
    new_chunk->next = nullptr;
    prev->next = new_chunk;
    return new_chunk;
}

__device__ bool DynamicVecVecView<bool>::add(index_type neuron_id, bool value) {
    // Only one thread is allowed to access neuron_id at the same time! There is no mutex

    auto* cur_chunk = chunks + neuron_id;
    auto siz = 0U;
    while (cur_chunk->next != nullptr) {
        siz += cur_chunk->filled;
        cur_chunk = cur_chunk->next;
    }
    if (cur_chunk->filled == cur_chunk->size) {
        Chunk<std::uint32_t>* new_chunk = get_new_chunk(cur_chunk);
        // Pool exhausted: unlike the earlier silent `return;` here, this MUST be reported to the
        // caller -- add_synapse() appends to other_neuron_ids/weights/excitatory/deleted in
        // lockstep, and a silently-dropped bit here desyncs this array's length from its
        // siblings' with no trap anywhere, corrupting state that only surfaces much later (e.g.
        // as a bogus chunk read in copy_to_host()).
        if (new_chunk == nullptr)
            return false;
        cur_chunk = new_chunk;
    }
    siz += cur_chunk->filled;

    const index_type bit_idx = cur_chunk->filled;
    const index_type word_idx = bit_idx >> 5U;
    const index_type bit_pos = bit_idx & 31U;
    if (value)
        atomicOr(cur_chunk->begin + word_idx, 1U << bit_pos);
    else
        atomicAnd(cur_chunk->begin + word_idx, ~(1U << bit_pos));
    ++cur_chunk->filled;
    return true;
}

__device__ bool DynamicVecVecView<bool>::get(index_type neuron_id, index_type bit_idx) const {
    RELEARN_DEVICE_CUDA_CHECK(chunks != nullptr, "DynamicVecVecView::get: DynamicVecVecView is empty");
    const auto* cur_chunk = chunks + neuron_id;
    auto cur_offset = 0U;
    auto chunk_it = 0U;
    while (cur_chunk != nullptr) {
        if (bit_idx < cur_offset + cur_chunk->filled) {
            const index_type local = bit_idx - cur_offset;
            return (cur_chunk->begin[local >> 5U] >> (local & 31U)) & 1U;
        }
        cur_offset += cur_chunk->filled;
        cur_chunk = cur_chunk->next;
        chunk_it++;
    }
    RELEARN_DEVICE_CUDA_CHECK(false, "DynamicVecVecView::get: OOB neuron=%u idx=%u size=%u chunks=%u", neuron_id, bit_idx, cur_offset, chunk_it);
    return false;
}

__device__ DynamicVecVecView<bool>::index_type DynamicVecVecView<bool>::get_size(index_type neuron_id) const {
    RELEARN_DEVICE_CUDA_CHECK(chunks != nullptr, "DynamicVecVecView::get_size: chunks is null");
    const auto* cur_chunk = chunks + neuron_id;
    auto size = 0U;
    while (cur_chunk != nullptr) {
        size += cur_chunk->filled;
        cur_chunk = cur_chunk->next;
    }
    return size;
}

__device__ void DynamicVecVecView<bool>::set(index_type neuron_id, index_type bit_idx, bool value) {
    auto* cur_chunk = chunks + neuron_id;
    auto cur_offset = 0U;
    while (cur_chunk != nullptr) {
        if (bit_idx < cur_offset + cur_chunk->filled) {
            const index_type local = bit_idx - cur_offset;
            const index_type word_idx = local >> 5U;
            const index_type bit_pos = local & 31U;
            if (value)
                atomicOr(cur_chunk->begin + word_idx, 1U << bit_pos);
            else
                atomicAnd(cur_chunk->begin + word_idx, ~(1U << bit_pos));
            return;
        }
        cur_offset += cur_chunk->filled;
        cur_chunk = cur_chunk->next;
    }
    RELEARN_DEVICE_CUDA_CHECK(false, "DynamicVecVecView::set: index out of bounds");
}

__device__ void DynamicVecVecView<bool>::pop_back(index_type neuron_id) {
    auto* cur_chunk = chunks + neuron_id;
    Chunk<std::uint32_t>* last_nonempty = nullptr;
    while (cur_chunk != nullptr) {
        if (cur_chunk->filled > 0)
            last_nonempty = cur_chunk;
        cur_chunk = cur_chunk->next;
    }
    RELEARN_DEVICE_CUDA_CHECK(last_nonempty != nullptr, "DynamicVecVecView::pop_back: pop_back on empty neuron");
    --last_nonempty->filled;

    if (last_nonempty->next != nullptr) {
        auto* trailing = last_nonempty->next;
        pool->release_block(pool->ptr_to_block_id(reinterpret_cast<std::uint8_t*>(trailing)));
        last_nonempty->next = nullptr;
    }

    if (last_nonempty->filled == 0 && last_nonempty->prev != nullptr) {
        last_nonempty->prev->next = nullptr;
        pool->release_block(pool->ptr_to_block_id(reinterpret_cast<std::uint8_t*>(last_nonempty)));
    }
}

// ---- DynamicVecVec<bool> bit-packed specialisation ----

void DynamicVecVec<bool>::init_main(std::size_t number_elements, std::size_t init_size, SharedBlockPool* pool) {
    const auto init_bits = ((init_size + 31U) / 32U) * 32U;
    // Data bits per overflow block: subtract the embedded Chunk header, convert remainder to bits.
    const auto chunk_bits = (pool->get_block_bytes() - sizeof(Chunk<std::uint32_t>)) * 8U;
    const auto init_words = init_bits / 32U;

    total_words = number_elements * init_words;
    main_chunks = number_elements;

    cudaMalloc_bridge(reinterpret_cast<void**>(&data), total_words * sizeof(std::uint32_t));
    cudaMalloc_bridge(reinterpret_cast<void**>(&chunks), number_elements * sizeof(Chunk<std::uint32_t>));
    cudaMemset_bridge(data, 0, total_words * sizeof(std::uint32_t));

    std::vector<Chunk<std::uint32_t>> h_chunks(number_elements);
    for (std::size_t i = 0; i < number_elements; ++i) {
        h_chunks[i] = Chunk<std::uint32_t>{
            data + i * init_words, static_cast<std::uint32_t>(i),
            0U, static_cast<std::uint32_t>(init_bits), nullptr, nullptr
        };
    }
    cudaMemcpy_to_device_bridge(chunks, h_chunks.data(), number_elements * sizeof(Chunk<std::uint32_t>));

    cudaMalloc_bridge(reinterpret_cast<void**>(&device_class), sizeof(DynamicVecVecView<bool>));
    DynamicVecVecView<bool> view{};
    view.chunks = chunks;
    view.pool = pool->get_device_pool();
    view.new_chunk_size = static_cast<std::uint32_t>(chunk_bits); // bits
    view.main_chunks = static_cast<std::uint32_t>(number_elements);
    cudaMemcpy_to_device_bridge(device_class, &view, sizeof(DynamicVecVecView<bool>));
}

DynamicVecVec<bool>::DynamicVecVec(std::size_t number_elements) {
    const auto init_size = CudaConfig::expected_synapses_per_neuron;
    const auto data_bits = std::max(std::size_t{ 32 },
                                    static_cast<std::size_t>(CudaConfig::overflow_chunk_size_factor * init_size));
    const auto data_bytes = ((data_bits + 31U) / 32U) * 4U; // round up to whole uint32_t words
    const auto block_bytes = round_up_to_alignment(sizeof(Chunk<std::uint32_t>) + data_bytes, alignof(Chunk<std::uint32_t>));
    const auto overflow_blocks = static_cast<std::size_t>(number_elements * CudaConfig::number_overflow_chunks_factor);
    private_pool = std::make_unique<SharedBlockPool>(overflow_blocks, block_bytes);
    init_main(number_elements, init_size, private_pool.get());
}

DynamicVecVec<bool>::DynamicVecVec(std::size_t number_elements, std::size_t init_size, SharedBlockPool* shared_pool) {
    init_main(number_elements, init_size, shared_pool);
}

static __global__ void k_add_bool_from_displ(
    DynamicVecVecView<bool>* view,
    const std::uint8_t* values,
    const std::uint32_t* offsets,
    const std::uint32_t* neuron_ids,
    std::uint32_t n) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    const auto neuron_id = neuron_ids[tid];
    for (auto i = offsets[tid]; i < offsets[tid + 1]; ++i) {
        const auto successful = view->add(neuron_id, static_cast<bool>(values[i]));
        RELEARN_DEVICE_CUDA_CHECK(successful, "k_add_bool_from_displ: pool exhausted for neuron=%u", neuron_id);
    }
}

void DynamicVecVec<bool>::add_from_map(const std::unordered_map<std::uint32_t, std::vector<bool>>& map) {
    if (map.empty())
        return;

    const auto n = static_cast<std::uint32_t>(map.size());

    std::vector<std::uint32_t> h_neuron_ids;
    std::vector<std::uint32_t> h_offsets;
    std::vector<std::uint8_t> h_values;
    h_neuron_ids.reserve(n);
    h_offsets.reserve(n + 1);
    h_offsets.push_back(0U);

    for (const auto& [neuron_id, bools] : map) {
        h_neuron_ids.push_back(neuron_id);
        for (const bool b : bools)
            h_values.push_back(static_cast<std::uint8_t>(b));
        h_offsets.push_back(static_cast<std::uint32_t>(h_values.size()));
    }

    std::uint32_t* d_neuron_ids{};
    std::uint32_t* d_offsets{};
    std::uint8_t* d_values{};
    cudaMalloc_bridge(reinterpret_cast<void**>(&d_neuron_ids), n * sizeof(std::uint32_t));
    cudaMalloc_bridge(reinterpret_cast<void**>(&d_offsets), (n + 1) * sizeof(std::uint32_t));
    cudaMalloc_bridge(reinterpret_cast<void**>(&d_values), h_values.size() * sizeof(std::uint8_t));

    cudaMemcpy_to_device_bridge(d_neuron_ids, h_neuron_ids.data(), n * sizeof(std::uint32_t));
    cudaMemcpy_to_device_bridge(d_offsets, h_offsets.data(), (n + 1) * sizeof(std::uint32_t));
    cudaMemcpy_to_device_bridge(d_values, h_values.data(), h_values.size() * sizeof(std::uint8_t));

    const auto blocks = (n + 255U) / 256U;
    Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_ADD_FROM_DISPL_KERNEL);
    k_add_bool_from_displ<<<blocks, 256>>>(device_class, d_values, d_offsets, d_neuron_ids, n);
    cudaDeviceSynchronize();
    Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_ADD_FROM_DISPL_KERNEL);

    cudaFree_bridge(d_neuron_ids);
    cudaFree_bridge(d_offsets);
    cudaFree_bridge(d_values);
}

std::uint64_t DynamicVecVec<bool>::get_gpu_memory_footprint() const {
    std::uint64_t fp = total_words * sizeof(std::uint32_t)
                       + main_chunks * sizeof(Chunk<std::uint32_t>)
                       + sizeof(DynamicVecVecView<bool>);
    if (private_pool)
        fp += private_pool->get_gpu_memory_footprint();
    return fp;
}

float DynamicVecVec<bool>::get_average_filled_main_chunk() const {
    DeviceArray<float> filled_out(main_chunks);
    const auto [grid_size, block_size] = get_grid_ands_block_size(main_chunks, get_average_filled_main_chunk_kernel<bool>);
    Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_AVERAGE_FILLED_KERNEL);
    get_average_filled_main_chunk_kernel<bool><<<grid_size, block_size>>>(get_device_view(), main_chunks, filled_out.device_ptr());
    cudaDeviceSynchronize();
    Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_AVERAGE_FILLED_KERNEL);
    const auto sum = thrust::reduce(thrust::device_pointer_cast(filled_out.device_ptr()), thrust::device_pointer_cast(filled_out.device_ptr() + main_chunks));

    const auto r = sum / static_cast<float>(main_chunks);
    return r;
}

BoolChunkUsageStats DynamicVecVec<bool>::get_used_chunks_per_neuron() const {
    DeviceArray<std::size_t> counter(main_chunks);
    const auto [grid_size, block_size] = get_grid_ands_block_size(main_chunks, get_used_chunks_per_neuron_kernel<bool>);
    Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_USED_CHUNKS_KERNEL);
    get_used_chunks_per_neuron_kernel<bool><<<grid_size, block_size>>>(get_device_view(), main_chunks, counter.device_ptr());
    cudaDeviceSynchronize();
    Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_USED_CHUNKS_KERNEL);

    thrust::sort(thrust::device_pointer_cast(counter.device_ptr()), thrust::device_pointer_cast(counter.device_ptr() + main_chunks));
    std::size_t median;
    cudaMemcpy_to_host_bridge(&median, counter.device_ptr() + main_chunks / 2, sizeof(std::size_t));

    const auto d_begin = thrust::device_pointer_cast(counter.device_ptr());
    const auto d_end = thrust::device_pointer_cast(counter.device_ptr() + main_chunks);

    const std::size_t sum_counter = thrust::reduce(d_begin, d_end);
    const std::size_t max = thrust::reduce(d_begin, d_end, std::size_t{ 0 }, cuda::maximum<std::size_t>());
    const std::size_t count_gt_1 = static_cast<std::size_t>(d_end - thrust::upper_bound(d_begin, d_end, std::size_t{ 1 }));
    const float avg = sum_counter / static_cast<float>(main_chunks);
    return { sum_counter, avg, median, max, count_gt_1 };
}

static __global__ void get_bool_memory_pointers(
    const DynamicVecVecView<bool>* view,
    std::size_t number_main_chunks,
    const std::size_t* partition,
    std::uint32_t** chunk_begin_ptr,
    std::size_t* chunk_bit_lengths) {
    const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= number_main_chunks)
        return;

    const auto begin = partition[thread_id];
    const auto end = partition[thread_id + 1];
    auto* chunk = view->chunks + thread_id;
    auto counter = 0U;
    while (chunk != nullptr) {
        RELEARN_DEVICE_CUDA_CHECK(counter <= end - begin, "get_bool_memory_pointers: Partition not matching counter (bool)");
        chunk_begin_ptr[begin + counter] = chunk->begin;
        chunk_bit_lengths[begin + counter] = chunk->filled;
        counter++;
        chunk = chunk->next;
    }
}

std::vector<std::vector<bool>> DynamicVecVec<bool>::copy_to_host() const {
    DeviceArray<std::size_t> counter(main_chunks);
    const auto [grid_size, block_size] = get_grid_ands_block_size(main_chunks, get_used_chunks_per_neuron_kernel<bool>);
    Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_USED_CHUNKS_KERNEL);
    get_used_chunks_per_neuron_kernel<bool><<<grid_size, block_size>>>(get_device_view(), main_chunks, counter.device_ptr());
    cudaDeviceSynchronize_bridge();
    Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_USED_CHUNKS_KERNEL);

    const auto sum_chunks = thrust::reduce(thrust::device_pointer_cast(counter.device_ptr()),
                                           thrust::device_pointer_cast(counter.device_ptr() + main_chunks));
    const auto part = prefixSum(counter.device_ptr(), main_chunks);

    DeviceArray<std::uint32_t*> chunk_ptrs(sum_chunks);
    DeviceArray<std::size_t> chunk_bit_lengths_dev(sum_chunks);

    const auto [grid_size2, block_size2] = get_grid_ands_block_size(main_chunks, get_bool_memory_pointers);
    Timers::start(TimerRegion::CUDA_DEVICE_VEC_VEC_BOOL_MEMORY_POINTERS_KERNEL);
    get_bool_memory_pointers<<<grid_size2, block_size2>>>(
        get_device_view(), main_chunks,
        part.device_ptr(),
        chunk_ptrs.device_ptr(),
        chunk_bit_lengths_dev.device_ptr());
    cudaDeviceSynchronize_bridge();
    Timers::stop_and_add(TimerRegion::CUDA_DEVICE_VEC_VEC_BOOL_MEMORY_POINTERS_KERNEL);

    const auto h_counter = counter.get_device_data();
    const auto h_chunk_ptrs = chunk_ptrs.get_device_data();
    const auto h_chunk_bits = chunk_bit_lengths_dev.get_device_data();

    std::vector<std::vector<bool>> all_host_data;
    all_host_data.reserve(main_chunks);

    auto offset = 0UL;
    for (auto neuron_id = 0UL; neuron_id < main_chunks; ++neuron_id) {
        const auto num_neuron_chunks = h_counter[neuron_id];

        auto total_bits = 0UL;
        for (auto i = 0UL; i < num_neuron_chunks; ++i) {
            total_bits += h_chunk_bits[offset + i];
        }

        std::vector<bool> neuron_data;
        neuron_data.reserve(total_bits);

        for (auto i = 0UL; i < num_neuron_chunks; ++i) {
            // offset (not offset + i) -- offset already advances once per iteration below,
            // matching h_chunk_ptrs[offset]; adding i on top double-counts the position for
            // every i >= 1.
            const auto bits = h_chunk_bits[offset];
            if (bits > 0) {
                const auto num_words = (bits + 31U) / 32U;
                std::vector<std::uint32_t> words(num_words);
                cudaMemcpy_to_host_bridge(words.data(), h_chunk_ptrs[offset], num_words * sizeof(std::uint32_t));
                for (auto bit_idx = 0UL; bit_idx < bits; ++bit_idx) {
                    neuron_data.push_back((words[bit_idx >> 5U] >> (bit_idx & 31U)) & 1U);
                }
            }
            offset++;
        }

        all_host_data.push_back(std::move(neuron_data));
    }

    return all_host_data;
}

void DynamicVecVec<bool>::record_usage_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint, const std::string& prefix) const {
    footprint->emplace(prefix + " filled main", static_cast<std::uint64_t>(get_average_filled_main_chunk() * 100.f));
    const auto [sum_counter, avg, median, max, count_gt_1] = get_used_chunks_per_neuron();
    footprint->emplace(prefix + " chunks per neuron (avg)", static_cast<std::uint64_t>(avg * 100.f));
    footprint->emplace(prefix + " chunks per neuron (median)", static_cast<std::uint64_t>(median * 100.f));
    footprint->emplace(prefix + " chunks per neuron (sum)", static_cast<std::uint64_t>(sum_counter * 100.f));
    footprint->emplace(prefix + " chunks per neuron (max)", static_cast<std::uint64_t>(max * 100.f));
    footprint->emplace(prefix + " chunks per neuron (>1)", static_cast<std::uint64_t>(count_gt_1));
}
