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

#include "util/Timers.h"

#include <cstddef>
#include <memory>

class StreamWrapper;

/**
 * @brief Synchronously fills @p bytes of device memory starting at @p d_ptr with the given byte value.
 * @param d_ptr   Device pointer to the start of the region to fill.
 * @param value   Byte value to write (interpreted as unsigned char).
 * @param bytes   Number of bytes to fill.
 */
void cudaMemset_bridge(void* d_ptr, int value, std::size_t bytes);

/**
 * @brief Asynchronously fills @p bytes of device memory with @p value on the given stream.
 * @param d_ptr          Device pointer to fill.
 * @param value          Byte value to write.
 * @param bytes          Number of bytes to fill.
 * @param stream_wrapper Stream to enqueue the operation on.
 */
void cudaMemsetAsync_bridge(void* d_ptr, int value, std::size_t bytes, const StreamWrapper& stream_wrapper);

/**
 * @brief Sets the active CUDA device based on the MPI rank (assumes one GPU per rank).
 * @param current_mpi_rank MPI rank used to select the GPU via cudaSetDevice.
 */
void cudaSetDevice_bridge(int current_mpi_rank);

/**
 * @brief Synchronously allocates @p bytes of device memory and stores the pointer in @p d_ptr.
 * @param d_ptr  Output: pointer to the allocated device memory.
 * @param bytes  Number of bytes to allocate.
 */
void cudaMalloc_bridge(void** d_ptr, std::size_t bytes);

/**
 * @brief Asynchronously allocates @p bytes of device memory on the given stream.
 * @param d_ptr         Output: pointer to the allocated device memory.
 * @param bytes         Number of bytes to allocate.
 * @param streamWrapper Stream to enqueue the allocation on.
 */
void cudaMallocAsync_bridge(void** d_ptr, std::size_t bytes, const StreamWrapper& streamWrapper);

/**
 * @brief Returns the size of a single cuRAND state object.
 * Used to compute the total byte size of a cuRAND state array.
 */
[[nodiscard]] std::size_t getCurandStateSize();

/**
 * @brief Allocates @p bytes of pinned (page-locked) host memory for fast DMA transfers.
 * @param d_ptr  Output: pointer to the pinned host memory.
 * @param bytes  Number of bytes to allocate.
 */
void cudaMallocHost_bridge(void** d_ptr, std::size_t bytes);

/** @brief Starts the CUDA profiler (nvprof / Nsight). */
void cudaProfilerStart_bridge();

/** @brief Stops the CUDA profiler. */
void cudaProfilerStop_bridge();

/**
 * @brief Synchronously copies @p bytes from the host buffer @p h_src to the device buffer @p d_dst.
 * @param d_dst  Destination device pointer.
 * @param h_src  Source host pointer.
 * @param bytes  Number of bytes to copy.
 */
void cudaMemcpy_to_device_bridge(void* d_dst, const void* h_src, std::size_t bytes);

/**
 * @brief Asynchronously copies @p bytes from @p h_src (host) to @p d_dst (device) on the given stream.
 * @param d_dst          Destination device pointer.
 * @param h_src          Source host pointer.
 * @param bytes          Number of bytes to copy.
 * @param streamWrapper  Stream to enqueue the copy on.
 */
void cudaMemcpyAsync_to_device_bridge(void* d_dst, const void* h_src, std::size_t bytes, const StreamWrapper& streamWrapper);

/**
 * @brief Synchronously copies @p bytes between two device buffers.
 * @param d_dst  Destination device pointer.
 * @param d_src  Source device pointer.
 * @param bytes  Number of bytes to copy.
 */
void cudaMemcpy_on_device_bridge(void* d_dst, const void* d_src, std::size_t bytes);

/**
 * @brief Asynchronously copies @p bytes between two device buffers using a default implicit stream.
 * @param d_dst  Destination device pointer.
 * @param d_src  Source device pointer.
 * @param bytes  Number of bytes to copy.
 */
void cudaMemcpyAsync_on_device_bridge(void* d_dst, const void* d_src, std::size_t bytes);

/**
 * @brief Asynchronously copies @p bytes between two device buffers on the given stream.
 * @param d_dst          Destination device pointer.
 * @param d_src          Source device pointer.
 * @param bytes          Number of bytes to copy.
 * @param streamWrapper  Stream to enqueue the copy on.
 */
void cudaMemcpyAsync_on_device_bridge(void* d_dst, const void* d_src, std::size_t bytes, const StreamWrapper& streamWrapper);

/**
 * @brief Asynchronously copies @p bytes from @p h_src (host) to @p d_dst (device) without an explicit stream.
 * @param d_dst  Destination device pointer.
 * @param h_src  Source host pointer.
 * @param bytes  Number of bytes to copy.
 */
void cudaMemcpyAsync_to_device_bridge(void* d_dst, const void* h_src, std::size_t bytes);

/**
 * @brief Synchronously copies @p bytes from the device buffer @p d_src to the host buffer @p h_dst.
 * @param h_dst  Destination host pointer.
 * @param d_src  Source device pointer.
 * @param bytes  Number of bytes to copy.
 */
void cudaMemcpy_to_host_bridge(void* h_dst, const void* d_src, std::size_t bytes);

/**
 * @brief Asynchronously copies @p bytes from @p d_src (device) to @p h_dst (host) on the given stream.
 * @param h_dst          Destination host pointer.
 * @param d_src          Source device pointer.
 * @param bytes          Number of bytes to copy.
 * @param streamWrapper  Stream to enqueue the copy on.
 */
void cudaMemcpyAsync_to_host_bridge(void* h_dst, const void* d_src, std::size_t bytes, const StreamWrapper& streamWrapper);

/**
 * @brief Asynchronously copies @p bytes from @p d_src (device) to @p h_dst (host) without an explicit stream.
 * @param h_dst  Destination host pointer.
 * @param d_src  Source device pointer.
 * @param bytes  Number of bytes to copy.
 */
void cudaMemcpyAsync_to_host_bridge(void* h_dst, const void* d_src, std::size_t bytes);

/**
 * @brief Blocks the host until all operations on the given stream have completed.
 * @param stream_wrapper Stream to synchronize.
 */
void cudaStreamSynchronize_bride(const StreamWrapper& stream_wrapper);

/**
 * @brief Synchronously frees device memory allocated with cudaMalloc_bridge.
 * @param d_ptr Device pointer to free (may be nullptr).
 */
void cudaFree_bridge(void* d_ptr);

/**
 * @brief Frees pinned host memory allocated with cudaMallocHost_bridge.
 * @param d_ptr Host pointer to free.
 */
void cudaFreeHost_bridge(void* d_ptr);

/**
 * @brief Asynchronously frees device memory on the given stream.
 * The memory remains valid until all previously enqueued work on the stream completes.
 * @param d_ptr          Device pointer to free.
 * @param stream_wrapper Stream to enqueue the free on.
 */
void cudaFreeAsync_bridge(void* d_ptr, const StreamWrapper& stream_wrapper);

/** @brief Blocks the host until all CUDA operations on all streams of the current device are complete. */
void cudaDeviceSynchronize_bridge();

/** @brief Clears the last CUDA error state so that subsequent error checks start fresh. */
void cudaResetLastError_bridge();

/**
 * @brief Records the start event of a GPU-timed region for `region` on `stream`, for kernels
 *      that are launched asynchronously on a stream without a following CPU-side
 *      synchronization (so Timers::start/stop around the launch would only measure launch
 *      overhead, not actual device execution time). Does not block the host.
 * @param region The timer region the measurement (once resolved) will be added to.
 * @param stream The stream the timed kernel is launched on.
 * @return An opaque handle to pass to cuda_stop_gpu_timer().
 */
[[nodiscard]] void* cuda_start_gpu_timer(TimerRegion region, const StreamWrapper& stream);

/**
 * @brief Records the stop event on `stream` for a timer started with cuda_start_gpu_timer() and
 *      queues it for later resolution via cuda_resolve_gpu_timers(). Does not block the host.
 * @param handle The handle returned by the matching cuda_start_gpu_timer() call.
 * @param stream The stream the timed kernel was launched on (same stream as at start).
 */
void cuda_stop_gpu_timer(void* handle, const StreamWrapper& stream);

/**
 * @brief Waits for every GPU timer queued since the last call and adds its measured elapsed
 *      time to Timers. Only call this where the host is already guaranteed to have caught up
 *      with the GPU (e.g. right after cudaDeviceSynchronize_bridge()), so it never introduces a
 *      new synchronization point of its own.
 */
void cuda_resolve_gpu_timers();

/**
 * @brief Returns the number of available CUDA devices on this host.
 */
[[nodiscard]] int cudaGetDeviceCount_bridge();

/**
 * @brief Fills a device array of @p size elements with @p value (synchronous).
 * @param device_ptr  Device pointer to the target array.
 * @param value       Value to write into every element.
 * @param size        Number of elements to fill.
 */
template <typename T>
void fill_entry(T* device_ptr, T value, std::size_t size);

/**
 * @brief Fills a device array of @p size elements with @p value asynchronously on the given stream.
 * @param device_ptr  Device pointer to the target array.
 * @param value       Value to write into every element.
 * @param size        Number of elements to fill.
 * @param stream      Stream to enqueue the fill on.
 */
template <typename T>
void fill_entry(T* device_ptr, T value, std::size_t size, const std::shared_ptr<StreamWrapper>& stream);
