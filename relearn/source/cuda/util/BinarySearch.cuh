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

#include <cstddef>
#include <limits>

template <typename T>
__device__ inline std::size_t binary_search(const std::size_t begin, const std::size_t end,
                                            const T* data,
                                            const T element) {
    RELEARN_DEVICE_CUDA_CHECK(end >= begin, "binary_search: end >= begin not fulfilled");
    const auto size = end - begin;

    if (size == 0) {
        return std::numeric_limits<std::size_t>::max();
    }

    int low = 0;
    int high = size - 1;
    while (low <= high) {
        const auto mid = low + (high - low) / 2;
        const auto idx = begin + mid;

        if (data[idx] == element) {
            return idx;
        }

        if (data[idx] < element) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return std::numeric_limits<std::size_t>::max();
}