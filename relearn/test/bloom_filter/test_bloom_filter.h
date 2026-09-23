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

#include <cstdint>
#include <vector>

class BloomFilter;

// Inserts each key in `keys_to_insert` into neuron_id's filter (device-side), mirroring the
// atomicOr insertion pattern used in production by NetworkGraph's rebuild_bloom_kernel and
// OnlyOutgoingView::add_synapse.
void device_bloom_insert(BloomFilter& filter, std::uint32_t neuron_id, const std::vector<std::uint32_t>& keys_to_insert);

// Queries bloom_query(neuron_id, key) for each key in `keys_to_query`, returning one flag per key
// (1 = "might be present", 0 = "definitely absent").
std::vector<std::uint8_t> device_bloom_query(BloomFilter& filter, std::uint32_t neuron_id, const std::vector<std::uint32_t>& keys_to_query);

#endif
