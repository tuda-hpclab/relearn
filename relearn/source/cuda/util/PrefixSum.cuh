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

#include <cub/device/device_scan.cuh>

#include <cuda.h>

#include <cstddef>

template <typename T, typename U = std::size_t>
void prefixSum(const T* d_in, std::size_t d_in_size, DeviceArray<U>& d_out, DeviceArray<char>& preallocated_temp) {
    size_t temp_size = 0;

    if (d_out.size() < d_in_size + 1) {
        d_out = DeviceArray<U>(d_in_size + 1);
        const U c0 = 0;
        cudaMemset(d_out.device_ptr(), 0, sizeof(U));
    }

    cub::DeviceScan::InclusiveSum(
        nullptr, temp_size,
        d_in, d_out.device_ptr() + 1,
        d_in_size);

    if (temp_size > preallocated_temp.size()) {
        preallocated_temp = DeviceArray<char>(temp_size);
    }

    cub::DeviceScan::InclusiveSum(
        preallocated_temp.device_ptr(), temp_size,
        d_in, d_out.device_ptr() + 1,
        d_in_size);
}

template <typename T, typename U = std::size_t>
DeviceArray<U> prefixSum(const T* d_in, std::size_t d_in_size) {
    DeviceArray<char> temp(0);
    DeviceArray<U> d_out(0);
    prefixSum<T, U>(d_in, d_in_size, d_out, temp);
    return d_out;
}

template <typename T, typename U = std::size_t>
DeviceArray<U> prefixSum(const DeviceArray<T>& d_in) {
    DeviceArray<U> d_out(0);
    DeviceArray<char> temp(0);
    prefixSum<T, U>(d_in.device_ptr(), d_in.size(), d_out, temp);
    return d_out;
}

template <typename T, typename U = std::size_t>
void prefixSum(const T* d_in, std::size_t d_in_size, DeviceArray<U>& d_out, DeviceArray<char>& preallocated_temp, cudaStream_t stream) {
    size_t temp_size = 0;

    if (d_out.size() < d_in_size + 1) {
        d_out = DeviceArray<U>(d_in_size + 1);
        cudaMemset(d_out.device_ptr(), 0, sizeof(U));
    }

    cub::DeviceScan::InclusiveSum(nullptr, temp_size, d_in, d_out.device_ptr() + 1, d_in_size, stream);

    if (temp_size > preallocated_temp.size()) {
        preallocated_temp = DeviceArray<char>(temp_size);
    }

    cub::DeviceScan::InclusiveSum(preallocated_temp.device_ptr(), temp_size, d_in, d_out.device_ptr() + 1, d_in_size, stream);
}