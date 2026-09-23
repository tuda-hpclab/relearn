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

#ifdef RELEARN_CUDA_ENABLED

#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "util/RelearnException.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

/**
 * @brief Dual host+device array with lazy host↔device synchronization.
 *
 * LazySyncedArray<T> maintains a host std::vector and a device allocation that are kept in sync
 * on demand.  Only the last-modified side is considered valid; attempting to use both sides
 * after modifying both without a sync in between is an error.
 *
 * Modifications must be declared explicitly:
 *  - After writing to the host copy (via operator[], begin()/end(), or host()), call nothing —
 *    LazySyncedArray tracks modifications automatically through those accessors.
 *  - After a CUDA kernel writes to the device copy, call device_was_modified().
 *
 * Constructors that accept a stream perform all CUDA operations asynchronously on that stream.
 * Constructors without a stream use the synchronous CUDA API.
 *
 * The array may optionally be a non-owning view into a contiguous device region
 * (see the device_base_ptr constructor); in that case the destructor does not free the device memory.
 */
template <typename T>
class LazySyncedArray {
public:
    LazySyncedArray() = default;

    /**
     * @brief Allocates host and device storage for @p n elements (synchronous).
     * @param n Number of elements to allocate.
     */
    explicit LazySyncedArray(std::size_t n)
        : number_elements(n)
        , size_in_bytes(sizeof(T) * n)
        , sync_size_in_bytes(size_in_bytes)
        , host_data(n)
        , owns_memory(true) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&(device_data)), size_in_bytes);
    }

    /**
     * @brief Allocates host and device storage for @p n elements asynchronously on @p stream_wrapper.
     * @param n              Number of elements to allocate.
     * @param stream_wrapper Stream to use for asynchronous allocation.
     */
    explicit LazySyncedArray(std::size_t n, const std::shared_ptr<StreamWrapper> stream_wrapper)
        : number_elements(n)
        , size_in_bytes(sizeof(T) * n)
        , sync_size_in_bytes(size_in_bytes)
        , host_data(n)
        , owns_memory(true)
        , stream(stream_wrapper) {
        RelearnException::check(stream != nullptr, "LazySyncedArray::LazySyncedArray: stream is empty");
        cudaMallocAsync_bridge(reinterpret_cast<void**>(&(device_data)), size_in_bytes, *stream_wrapper);
    }

    /**
     * @brief Constructs from a moved host vector; device memory is allocated and data is uploaded immediately.
     * @param data Host vector (moved into the array).
     */
    LazySyncedArray(std::vector<T>&& data)
        : number_elements(data.size())
        , size_in_bytes(sizeof(T) * number_elements)
        , sync_size_in_bytes(size_in_bytes)
        , host_data(number_elements)
        , owns_memory(true) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&(device_data)), size_in_bytes);
        host_data = std::move(data);
        host_was_modified();
    }

    /**
     * @brief Constructs from a moved host vector with async upload on @p stream_wrapper.
     * @param data           Host vector (moved into the array).
     * @param stream_wrapper Stream to use for asynchronous upload.
     */
    LazySyncedArray(std::vector<T>&& data, const std::shared_ptr<StreamWrapper> stream_wrapper)
        : number_elements(data.size())
        , size_in_bytes(sizeof(T) * number_elements)
        , sync_size_in_bytes(size_in_bytes)
        , host_data(number_elements)
        , owns_memory(true)
        , stream(stream_wrapper) {
        RelearnException::check(stream != nullptr, "LazySyncedArray::LazySyncedArray: stream is empty");
        cudaMallocAsync_bridge(reinterpret_cast<void**>(&(device_data)), size_in_bytes, *stream_wrapper);
        host_data = std::move(data);
        host_was_modified();
    }

    /**
     * @brief Non-owning view into a sub-region of an existing device allocation.
     * @param device_base_ptr Pointer into the existing device allocation.
     * @param n               Number of elements in the view.
     */
    LazySyncedArray(T* device_base_ptr, std::size_t n)
        : number_elements(n)
        , size_in_bytes(sizeof(T) * n)
        , sync_size_in_bytes(size_in_bytes)
        , host_data(n)
        , device_data(device_base_ptr)
        , owns_memory(false) {
        // update_host();
    }

    LazySyncedArray(LazySyncedArray& other) = delete;
    LazySyncedArray operator=(LazySyncedArray& other) = delete;

    LazySyncedArray(LazySyncedArray&& other) noexcept
        : number_elements(other.number_elements)
        , size_in_bytes(other.size_in_bytes)
        , sync_size_in_bytes((other.sync_size_in_bytes))
        , host_data(std::move(other.host_data))
        , device_data(other.device_data)
        , owns_memory(other.owns_memory)
        , host_modified_(other.host_modified_)
        , device_modified_(other.device_modified_)
        , stream(other.stream) {
        // Reset other
        other.number_elements = 0;
        other.size_in_bytes = 0;
        other.sync_size_in_bytes = 0;
        other.device_data = nullptr;
        other.host_modified_ = false;
        other.device_modified_ = false;
        other.stream.reset();
    }

    LazySyncedArray& operator=(LazySyncedArray&& other) noexcept {
        if (this != &other) {
            if (device_data != nullptr && owns_memory) {
                cudaFree_bridge(device_data);
            }

            number_elements = other.number_elements;
            size_in_bytes = other.size_in_bytes;
            sync_size_in_bytes = other.sync_size_in_bytes;
            host_data = std::move(other.host_data);
            device_data = other.device_data;
            host_modified_ = other.host_modified_;
            device_modified_ = other.device_modified_;
            owns_memory = other.owns_memory;
            stream = other.stream;

            // Reset other
            other.number_elements = 0;
            other.size_in_bytes = 0;
            other.sync_size_in_bytes = 0;
            other.device_data = nullptr;
            other.host_modified_ = false;
            other.device_modified_ = false;
            other.owns_memory = false;
            other.stream.reset();
        }
        return *this;
    }

    ~LazySyncedArray() {
#ifdef RELEARN_CUDA_ENABLED
        if (owns_memory) {
            if (stream != nullptr) {
                cudaFreeAsync_bridge(device_data, *stream);
            } else {
                cudaFree_bridge(device_data);
            }
        }
#endif
    }

    /**
     * @brief Returns the device pointer without triggering a sync (returns nullptr if no device allocation).
     */
    [[nodiscard]] const T* get_device_ptr_const() const {
        if (device_data == nullptr) {
            return nullptr;
        }
        ensure_device_uptodate();
        return device_data;
    }

    [[nodiscard]] T* get_device_ptr() {
        if (device_data == nullptr) {
            return nullptr;
        }
        ensure_device_uptodate();
        device_was_modified();
        return device_data;
    }

    /**
     * @brief Forces both host and device copies to be up to date (uploads or downloads as needed).
     */
    void force_update() const {
        ensure_host_uptodate();
        ensure_device_uptodate();
    }

private:
    void update_device() const {
#ifdef RELEARN_CUDA_ENABLED
        if (stream != nullptr) {
            cudaMemcpyAsync_to_device_bridge(reinterpret_cast<void*>(device_data), reinterpret_cast<void*>(host_data.data()), sync_size_in_bytes, *stream);
        } else {
            cudaMemcpy_to_device_bridge(reinterpret_cast<void*>(device_data), reinterpret_cast<void*>(host_data.data()), sync_size_in_bytes);
        }
#endif
        host_modified_ = false;
    }

    void update_host() const {
#ifdef RELEARN_CUDA_ENABLED
        if (stream != nullptr) {
            cudaMemcpyAsync_to_host_bridge(reinterpret_cast<void*>(host_data.data()), reinterpret_cast<void*>(device_data), sync_size_in_bytes, *stream);
        } else {
            cudaMemcpy_to_host_bridge(reinterpret_cast<void*>(host_data.data()), reinterpret_cast<void*>(device_data), sync_size_in_bytes);
        }
#endif
        device_modified_ = false;
    }

    /**
     * @brief Declares that a CUDA kernel has written to the device copy.
     * Errors if the host copy is also marked modified (would create a conflict).
     */
    void device_was_modified() const {
        RelearnException::check(number_elements != 0, "LazySyncedArray:device_was_modified: Array is empty");
        RelearnException::check(!host_modified_, "LazySyncedArray:device_was_modified: Host and device were both modified");
        device_modified_ = true;
    }

    void ensure_device_uptodate() const {
        RelearnException::check(number_elements != 0, "LazySyncedArray:device(): Array is empty");
        if (host_modified_ && device_modified_) {
            RelearnException::check(false, "LazySyncedArray: both host and device modified — cannot synchronize");
        }
        if (host_modified_) {
            update_device();
            host_modified_ = false;
        }
    }

    void ensure_host_uptodate() const {
        // RelearnException::check(number_elements != 0, "LazySyncedArray:device(): Array is empty");
        if (host_modified_ && device_modified_) {
            RelearnException::check(false, "LazySyncedArray: both host and device modified — cannot synchronize");
        }
        if (device_modified_) {
            update_host();
            device_modified_ = false;
        }
    }

    std::size_t number_elements{};
    std::size_t size_in_bytes{};
    mutable std::size_t sync_size_in_bytes{};
    mutable std::vector<T> host_data{};
    T* device_data{};
    bool owns_memory{ true };

    mutable bool host_modified_ = false;
    mutable bool device_modified_ = false;

public:
    // --- std::vector-compatible interface (operates on host copy) ---
    /**
     * @brief Returns the number of elements (same on host and device).
     */
    [[nodiscard]] std::size_t size() const noexcept { return number_elements; }

    /**
     * @brief Returns the allocated capacity (equals size; no separate over-allocation).
     */
    [[nodiscard]] std::size_t capacity() const noexcept { return number_elements; }

    /**
     * @brief Returns true if the array contains no elements.
     */
    [[nodiscard]] bool empty() const noexcept { return number_elements == 0; }

    /**
     * @brief Resizes the array, reallocating device memory and re-uploading the host data.
     * @param new_size New number of elements.
     */
    void resize(std::size_t new_size) {
        RelearnException::check(owns_memory, "LazySyncedArray::resize: Cannot resize if I dont own the memory");
        ensure_host_uptodate();
        host_data.resize(new_size);

        size_in_bytes = sizeof(T) * new_size;
        sync_size_in_bytes = size_in_bytes;
        number_elements = new_size;
        host_was_modified();
#ifdef RELEARN_CUDA_ENABLED
        if (stream != nullptr) {
            cudaFreeAsync_bridge(device_data, *stream);
            cudaMallocAsync_bridge(reinterpret_cast<void**>(&device_data), size_in_bytes, *stream);
        } else {
            cudaFree_bridge(device_data);
            cudaMalloc_bridge(reinterpret_cast<void**>(&device_data), size_in_bytes);
        }
        update_device();
#endif
    }

    /**
     * @brief Resizes the array initialising new elements to @p init_value.
     * @param new_size   New number of elements.
     * @param init_value Value to assign to newly created elements.
     */
    void resize(std::size_t new_size, T init_value) {
        RelearnException::check(owns_memory, "LazySyncedArray::resize: Cannot resize if I dont own the memory");
        ensure_host_uptodate();
        host_data.resize(new_size, init_value);

        size_in_bytes = sizeof(T) * new_size;
        sync_size_in_bytes = size_in_bytes;
        number_elements = new_size;
        host_was_modified();
#ifdef RELEARN_CUDA_ENABLED
        if (stream != nullptr) {
            cudaFreeAsync_bridge(device_data, *stream);
            cudaMallocAsync_bridge(reinterpret_cast<void**>(&device_data), size_in_bytes, *stream);
        } else {
            cudaFree_bridge(device_data);
            cudaMalloc_bridge(reinterpret_cast<void**>(&device_data), size_in_bytes);
        }
        update_device();
#endif
    }

    /**
     * @brief Fills the device array with @p value without touching the host copy.
     * Uses cudaMemset for zero fills; a kernel otherwise.
     * @param value Value to fill every element with.
     */
    void fill(T value) {
        if (value == T{ 0 }) {
            if (stream != nullptr) {
                cudaMemsetAsync_bridge(device_data, 0, sizeof(T) * number_elements, *stream);
            } else {
                cudaMemset_bridge(device_data, 0, sizeof(T) * number_elements);
            }
        } else {
            if (stream != nullptr) {
                fill_entry<T>(device_data, value, number_elements, stream);
            } else {
                fill_entry<T>(device_data, value, number_elements);
            }
        }
        host_modified_ = false;
        device_modified_ = true;
    }

    /**
     * @brief Returns a reference to element @p i (downloads from device if needed, marks host modified).
     * @param i Element index.
     */
    T& operator[](std::size_t i) noexcept {
        ensure_host_uptodate();
        host_was_modified();
        return host_data[i];
    }

    const T& operator[](std::size_t i) const noexcept {
        ensure_host_uptodate();
        return host_data[i];
    }

    auto begin() noexcept {
        ensure_host_uptodate();
        host_was_modified();
        return host_data.begin();
    }
    auto end() noexcept {
        ensure_host_uptodate();
        host_was_modified();
        return host_data.end();
    }
    auto begin() const noexcept {
        ensure_host_uptodate();
        return host_data.begin();
    }
    auto end() const noexcept {
        ensure_host_uptodate();
        return host_data.end();
    }

    /**
     * @brief Returns a span over the host data, downloading from device if needed; marks host modified.
     */
    [[nodiscard]] std::span<T> host() {
        ensure_host_uptodate();
        host_was_modified();
        return std::span<T>(host_data);
    }

    /**
     * @brief Returns a read-only span over the host data, downloading from device if needed.
     */
    [[nodiscard]] std::span<const T> host() const {
        ensure_host_uptodate();
        return std::span<const T>(host_data);
    }

    /**
     * @brief Returns total device memory (bytes) occupied by this array.
     */
    [[nodiscard]] std::size_t get_memory_footprint() const {
        return sizeof(T) * number_elements;
    }

private:
    void host_was_modified() const {
        host_modified_ = true;
    }

    std::shared_ptr<StreamWrapper> stream;
};

#else
#define LazySyncedArray std::vector
#endif
