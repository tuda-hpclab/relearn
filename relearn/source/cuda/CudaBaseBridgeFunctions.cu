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

#include "util/Timers.h"
#include "util/Util.cuh"
#include "wrapper/StreamWrapper.cuh"

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>

#include <vector>

namespace {
struct PendingGpuTimer {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    timer_stack stack;
};

// Only ever touched from the single host thread driving the CUDA calls -- not synchronized.
std::vector<PendingGpuTimer> pending_gpu_timers{};
} // namespace

void* cuda_start_gpu_timer(const TimerRegion region, const StreamWrapper& stream) {
    auto* handle = new PendingGpuTimer{};
    CUDA_CHECK(cudaEventCreate(&handle->start));
    CUDA_CHECK(cudaEventCreate(&handle->stop));
    handle->stack = Timers::capture_stack(region);
    CUDA_CHECK(cudaEventRecord(handle->start, get_cuda_stream_from_wrapper(stream)));
    return handle;
}

void cuda_stop_gpu_timer(void* handle, const StreamWrapper& stream) {
    auto* pending = static_cast<PendingGpuTimer*>(handle);
    CUDA_CHECK(cudaEventRecord(pending->stop, get_cuda_stream_from_wrapper(stream)));
    pending_gpu_timers.push_back(std::move(*pending));
    delete pending;
}

void cuda_resolve_gpu_timers() {
    for (auto& timer : pending_gpu_timers) {
        CUDA_CHECK(cudaEventSynchronize(timer.stop));

        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer.start, timer.stop));

        const auto elapsed_ns = static_cast<std::uint64_t>(static_cast<double>(elapsed_ms) * 1e6);
        Timers::add_measurement(timer.stack, elapsed_ns);

        CUDA_CHECK(cudaEventDestroy(timer.start));
        CUDA_CHECK(cudaEventDestroy(timer.stop));
    }
    pending_gpu_timers.clear();
}

template <typename T>
__global__ void fill_kernel(T* const device_ptr, const T value, const std::size_t size) {
    const auto thread_id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    if (thread_id >= size) {
        return;
    }

    device_ptr[thread_id] = value;
}

void cudaMemsetAsync_bridge(void* d_ptr, const int value, const size_t bytes, const StreamWrapper& stream_wrapper) {
    CUDA_CHECK(cudaMemsetAsync(d_ptr, value, bytes, get_cuda_stream_from_wrapper(stream_wrapper)));
}

void cudaMemset_bridge(void* d_ptr, const int value, const size_t bytes) {
    CUDA_CHECK(cudaMemset(d_ptr, value, bytes));
}

void cudaSetDevice_bridge(const int current_mpi_rank) {
    cudaSetDevice(current_mpi_rank);
}

int cudaGetDeviceCount_bridge() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

void cudaMalloc_bridge(void** d_ptr, const size_t bytes) {
    // Timers::start(TimerRegion::CUDA_MALLOC);
    CUDA_CHECK(cudaMalloc(d_ptr, bytes));
    // Timers::stop_and_add(TimerRegion::CUDA_MALLOC);
}

void cudaMallocAsync_bridge(void** d_ptr, const size_t bytes, const StreamWrapper& stream_wrapper) {
    Timers::start(TimerRegion::CUDA_MALLOC);
    const auto stream = get_cuda_stream_from_wrapper(stream_wrapper);
    CUDA_CHECK(cudaMallocAsync(d_ptr, bytes, stream));
    Timers::stop_and_add(TimerRegion::CUDA_MALLOC);
}

void cudaStreamSynchronize_bride(const StreamWrapper& stream_wrapper) {
    CUDA_CHECK(cudaStreamSynchronize(get_cuda_stream_from_wrapper(stream_wrapper)));
}

void cudaDeviceSynchronize_bridge() {
    CUDA_CHECK(cudaDeviceSynchronize());
    kernelErrCheck();
}

void cudaResetLastError_bridge() {
    cudaGetLastError();
}

std::size_t getCurandStateSize() {
    return sizeof(curandState);
}

void cudaMallocHost_bridge(void** d_ptr, size_t bytes) {
    CUDA_CHECK(cudaMallocHost(d_ptr, bytes));
}

void cudaMemcpy_to_device_bridge(void* d_dst, const void* h_src, const size_t bytes) {
    // Timers::start(TimerRegion::CUDA_MEMCPY_DEVICE);
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice));
    // Timers::stop_and_add(TimerRegion::CUDA_MEMCPY_DEVICE);
}

void cudaMemcpy_on_device_bridge(void* d_dst, const void* d_src, const size_t bytes) {
    Timers::start(TimerRegion::CUDA_MEMCPY_ON_DEVICE);
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
    Timers::stop_and_add(TimerRegion::CUDA_MEMCPY_ON_DEVICE);
}

void cudaMemcpyAsync_on_device_bridge(void* d_dst, const void* d_src, const size_t bytes) {
    Timers::start(TimerRegion::CUDA_MEMCPY_ON_DEVICE);
    CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
    Timers::stop_and_add(TimerRegion::CUDA_MEMCPY_ON_DEVICE);
}

void cudaMemcpyAsync_on_device_bridge(void* d_dst, const void* d_src, const size_t bytes, const StreamWrapper& stream_wrapper) {
    Timers::start(TimerRegion::CUDA_MEMCPY_ON_DEVICE);
    CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, get_cuda_stream_from_wrapper(stream_wrapper)));
    Timers::stop_and_add(TimerRegion::CUDA_MEMCPY_ON_DEVICE);
}

void cudaMemcpyAsync_to_device_bridge(void* d_dst, const void* h_src, size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice));
}

void cudaMemcpyAsync_to_device_bridge(void* d_dst, const void* h_src, size_t bytes, const StreamWrapper& stream_wrapper) {
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, get_cuda_stream_from_wrapper(stream_wrapper)));
}

void cudaProfilerStart_bridge() {
    cudaProfilerStart();
}

void cudaProfilerStop_bridge() {
    cudaProfilerStop();
}

void cudaMemcpy_to_host_bridge(void* h_dst, const void* d_src, const size_t bytes) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, bytes, cudaMemcpyDeviceToHost));
}

void cudaMemcpyAsync_to_host_bridge(void* h_dst, const void* d_src, const size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, bytes, cudaMemcpyDeviceToHost));
}

void cudaMemcpyAsync_to_host_bridge(void* h_dst, const void* d_src, const size_t bytes, const StreamWrapper& stream_wrapper) {
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, bytes, cudaMemcpyDeviceToHost, get_cuda_stream_from_wrapper(stream_wrapper)));
}

void cudaFree_bridge(void* d_ptr) {
    // Timers::start(TimerRegion::CUDA_FREE);
    if (d_ptr != nullptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
    // Timers::stop_and_add(TimerRegion::CUDA_FREE);
}

void cudaFreeAsync_bridge(void* d_ptr, const StreamWrapper& stream_wrapper) {
    if (d_ptr != nullptr) {
        CUDA_CHECK(cudaFreeAsync(d_ptr, get_cuda_stream_from_wrapper(stream_wrapper)));
    }
}

void cudaFreeHost_bridge(void* d_ptr) {
    if (d_ptr != nullptr) {
        CUDA_CHECK(cudaFreeHost(d_ptr));
    }
}

template <typename T>
void fill_entry(T* const device_ptr, const T value, const std::size_t size) {
    Timers::start(TimerRegion::CUDA_FILL_KERNEL);
    const auto& [blocks, threads] = get_grid_ands_block_size(size, fill_kernel<T>);
    fill_kernel<T><<<blocks, threads>>>(device_ptr, value, size);
    cudaDeviceSynchronize();
    kernelErrCheck();
    Timers::stop_and_add(TimerRegion::CUDA_FILL_KERNEL);
}

template <typename T>
void fill_entry(T* const device_ptr, const T value, const std::size_t size, const std::shared_ptr<StreamWrapper>& stream) {
    const auto& [blocks, threads] = get_grid_ands_block_size(size, fill_kernel<T>);

    // Launched on `stream` without a following sync -- LazySyncedArray/DeviceArray callers manage
    // their own synchronization on their own schedule, so time via CUDA events instead of a blocking
    // Timers::start/stop; whichever cuda_resolve_gpu_timers() call happens to run next (there are
    // several throughout the simulation loop) will pick this measurement up correctly regardless of
    // which region is active at that point, since cudaEventSynchronize always waits for the right event.
    auto* gpu_timer = cuda_start_gpu_timer(TimerRegion::CUDA_FILL_KERNEL, *stream);
    fill_kernel<<<blocks, threads, 0, get_cuda_stream_from_wrapper(*stream)>>>(device_ptr, value, size);
    cuda_stop_gpu_timer(gpu_timer, *stream);
}

template void fill_entry<int>(int* const device_ptr, const int value, const std::size_t size);
template void fill_entry<unsigned long>(unsigned long* const device_ptr, const unsigned long value, const std::size_t size);
template void fill_entry<float>(float* const device_ptr, const float value, const std::size_t size);
template void fill_entry<unsigned int>(unsigned int* const device_ptr, const unsigned int value, const std::size_t size);
template void fill_entry<unsigned char>(unsigned char* const device_ptr, const unsigned char value, const std::size_t size);
template void fill_entry<unsigned short>(unsigned short* const device_ptr, const unsigned short value, const std::size_t size);

template void fill_entry<int>(int* const device_ptr, const int value, const std::size_t size, const std::shared_ptr<StreamWrapper>& stream);
template void fill_entry<unsigned long>(unsigned long* const device_ptr, const unsigned long value, const std::size_t size, const std::shared_ptr<StreamWrapper>& stream);
template void fill_entry<float>(float* const device_ptr, const float value, const std::size_t size, const std::shared_ptr<StreamWrapper>& stream);
template void fill_entry<unsigned int>(unsigned int* const device_ptr, const unsigned int value, const std::size_t size, const std::shared_ptr<StreamWrapper>& stream);
template void fill_entry<unsigned char>(unsigned char* const device_ptr, const unsigned char value, const std::size_t size, const std::shared_ptr<StreamWrapper>& stream);
template void fill_entry<unsigned short>(unsigned short* const device_ptr, const unsigned short value, const std::size_t size, const std::shared_ptr<StreamWrapper>& stream);