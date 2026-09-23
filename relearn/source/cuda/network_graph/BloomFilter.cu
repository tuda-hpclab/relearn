/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BloomFilter.cuh"

BloomFilterView BloomFilter::get_gpu_view() {
    return BloomFilterView{ bit_per_edge, number_hash_functions, words_per_filter, bits.get_device_ptr() };
}
