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

#include "cuda/CudaBaseBridgeFunctions.h"

#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

/**
 * Owns a typed device-memory buffer.  Non-copyable; move-only.
 *
 * All constructors allocate device memory; some accept a stream for async allocation.
 * The destructor frees memory asynchronously if a stream was provided, synchronously otherwise.
 */
template <typename T>
class DeviceArray {
public:
    /**
     * @brief Allocates device memory and copies data from the host span synchronously.
     * @param data Host span to copy from; size determines the allocation.
     */
    explicit DeviceArray(const std::span<const T>& data)
        : size_(data.size()) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&device_ptr_), sizeof(T) * size_);                      // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T** to void** for cudaMalloc_bridge's C-style out-param
        cudaMemcpy_to_device_bridge(reinterpret_cast<void*>(device_ptr_), data.data(), sizeof(T) * size_); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T* to void* for the CUDA memcpy API
    }

    /**
     * @brief Allocates device memory and copies data from the host span asynchronously on @p stream_wrapper.
     * @param data           Host span to copy from.
     * @param stream_wrapper Stream to use for the asynchronous allocation and copy.
     */
    explicit DeviceArray(const std::span<const T>& data, const std::shared_ptr<StreamWrapper>& stream_wrapper)
        : size_(data.size())
        , stream(stream_wrapper) {
        cudaMallocAsync_bridge(reinterpret_cast<void**>(&device_ptr_), sizeof(T) * size_, *stream_wrapper);                      // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T** to void** for cudaMallocAsync_bridge's C-style out-param
        cudaMemcpyAsync_to_device_bridge(reinterpret_cast<void*>(device_ptr_), data.data(), sizeof(T) * size_, *stream_wrapper); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T* to void* for the CUDA memcpy API
    }

    /**
     * @brief Allocates uninitialised device memory for @p _size elements.
     * @param _size Number of elements to allocate.
     */
    explicit DeviceArray(std::size_t _size)
        : size_(_size) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&device_ptr_), sizeof(T) * size_); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T** to void** for cudaMalloc_bridge's C-style out-param
    }

    /**
     * @brief Allocates device memory and fills it with @p value (arithmetic types, zero-path uses cudaMemset).
     * @param _size Number of elements to allocate.
     * @param value Fill value.
     */
    explicit DeviceArray(std::size_t _size, T value)
        requires std::is_arithmetic_v<T>
        : size_(_size) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&device_ptr_), sizeof(T) * size_); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T** to void** for cudaMalloc_bridge's C-style out-param

        if (value == T{ 0 }) {
            cudaMemset_bridge(device_ptr_, 0, sizeof(T) * size_);
        } else {
            fill(value);
        }
    }

    /**
     * @brief Allocates device memory and fills it with @p value (non-arithmetic types).
     * @param _size Number of elements to allocate.
     * @param value Fill value.
     */
    explicit DeviceArray(std::size_t _size, T value)
        requires(!std::is_arithmetic_v<T>)
        : size_(_size) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&device_ptr_), sizeof(T) * size_); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T** to void** for cudaMalloc_bridge's C-style out-param

        fill(value);
    }

    /**
     * @brief Allocates device memory asynchronously on @p stream_wrapper (uninitialised).
     * @param _size          Number of elements to allocate.
     * @param stream_wrapper Stream to use for the asynchronous allocation.
     */
    explicit DeviceArray(std::size_t _size, const std::shared_ptr<StreamWrapper>& stream_wrapper)
        : size_(_size)
        , stream(stream_wrapper) {
        cudaMallocAsync_bridge(reinterpret_cast<void**>(&device_ptr_), sizeof(T) * size_, *stream_wrapper); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T** to void** for cudaMallocAsync_bridge's C-style out-param
    }

    DeviceArray() = delete;

    // Rule of 5
    DeviceArray(const DeviceArray&) = delete;

    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& other) noexcept
        : device_ptr_(other.device_ptr_)
        , size_(other.size_) {
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            cudaFree_bridge(device_ptr_);
            device_ptr_ = other.device_ptr_;
            size_ = other.size_;

            other.device_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Copies device data to a host vector synchronously.
     */
    [[nodiscard]] std::vector<T> get_device_data() const {
        // std::vector<bool> is a bitset specialization with no contiguous .data(), so it can't be
        // memcpy'd into directly; stage through a byte buffer instead (sizeof(bool) == 1).
        if constexpr (std::is_same_v<T, bool>) {
            std::vector<std::uint8_t> raw(size_);
            cudaMemcpy_to_host_bridge(raw.data(), reinterpret_cast<void*>(device_ptr_), sizeof(T) * size_); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T* to void* for the CUDA memcpy API
            return std::vector<T>(raw.begin(), raw.end());
        } else {
            std::vector<T> data(size_);
            cudaMemcpy_to_host_bridge(data.data(), reinterpret_cast<void*>(device_ptr_), sizeof(T) * size_); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T* to void* for the CUDA memcpy API
            return data;
        }
    }

    /**
     * @brief Copies device data to a host vector asynchronously on @p stream_wrapper.
     * @param stream_wrapper Stream to enqueue the copy on.
     */
    [[nodiscard]] std::vector<T> get_device_data(const StreamWrapper& stream_wrapper) const {
        if constexpr (std::is_same_v<T, bool>) {
            std::vector<std::uint8_t> raw(size_);
            cudaMemcpyAsync_to_host_bridge(raw.data(), reinterpret_cast<void*>(device_ptr_), sizeof(T) * size_, stream_wrapper); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T* to void* for the CUDA memcpy API
            return std::vector<T>(raw.begin(), raw.end());
        } else {
            std::vector<T> data(size_);
            cudaMemcpyAsync_to_host_bridge(data.data(), reinterpret_cast<void*>(device_ptr_), sizeof(T) * size_, stream_wrapper); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) - T* to void* for the CUDA memcpy API
            return data;
        }
    }

    /**
     * @brief Fills all elements with @p value using the associated stream if set, otherwise synchronously.
     * @param value Value to write into every element.
     */
    void fill(T value) {
        if (stream != nullptr) {
            fill_entry<T>(device_ptr_, value, size_, stream);
        } else {
            fill_entry<T>(device_ptr_, value, size_);
        }
    }

    /**
     * @brief Returns total device memory (bytes) occupied by this array.
     */
    [[nodiscard]] std::size_t get_memory_footprint() const {
        return sizeof(T) * size_;
    }

    /**
     * @brief Returns the device pointer cast to void*.
     */
    [[nodiscard]] void* void_device_ptr() const {
        return device_ptr_;
    }

    /**
     * @brief Returns the number of elements in the array.
     */
    [[nodiscard]] std::size_t size() const {
        return size_;
    }

    /**
     * @brief Returns the raw typed device pointer.
     */
    [[nodiscard]] T* device_ptr() const {
        return device_ptr_;
    }

    ~DeviceArray() {
        if (device_ptr_ != nullptr) {
            if (stream != nullptr) {
                cudaFreeAsync_bridge(device_ptr_, *stream);
            } else {
                cudaFree_bridge(device_ptr_);
            }
        }
    }

private:
    T* device_ptr_{};
    std::size_t size_{};
    std::shared_ptr<StreamWrapper> stream;
};
