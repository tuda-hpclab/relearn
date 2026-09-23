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

#include "Util.cuh"

#include <cstddef>

template <typename T>
__device__ inline std::size_t upper_bound(
    const std::size_t begin,
    const std::size_t end,
    const T* data,
    const T element) {
    RELEARN_DEVICE_CUDA_CHECK(end >= begin, "upper_bound: end >= begin not fulfilled");

    std::size_t low = begin;
    std::size_t high = end;

    while (low < high) {
        std::size_t mid = low + (high - low) / 2;

        if (data[mid] <= element) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return low - 1;
}
