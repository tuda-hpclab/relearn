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

template <typename T>
class DynamicVecVecView;

struct DeviceSharedBlockPool;

struct StressPoolResult {
    // Number of times a thread found a block it just acquired already marked as owned by
    // someone else (must be 0 for acquire_block/release_block to be correct).
    std::uint32_t violations;
    // Number of acquire attempts that gave up after the per-attempt timeout without ever
    // getting a block (must be 0). A nonzero count means either a genuine liveness bug in
    // acquire_block/release_block, or a GPU too slow/contended to complete this trivially small
    // amount of work within the (generous) timeout -- either way, worth failing loudly on rather
    // than spinning forever.
    std::uint32_t timed_out_threads;
};

// Launches num_threads threads (single block, so they're guaranteed co-resident), each
// repeatedly acquiring a block, claiming sole ownership of it via an `owner` marker array,
// holding it briefly, releasing ownership, then returning it to the pool -- iterations times.
// Each acquire attempt is bounded by a generous wall-clock timeout (see k_stress_pool's
// timeout_cycles) rather than spinning unconditionally, so a genuine hang fails fast with a
// clear count instead of blocking the whole test run (and CI) indefinitely.
StressPoolResult device_stress_pool(
    DeviceSharedBlockPool* d_pool,
    std::uint32_t num_blocks,
    std::uint32_t num_threads,
    std::uint32_t iterations);

// Single-threaded: acquires blocks from d_pool until it reports empty (or max_to_drain is
// reached), returning the block ids obtained, in acquisition order.
std::vector<std::uint32_t> device_drain_pool(
    DeviceSharedBlockPool* d_pool,
    std::uint32_t max_to_drain);

// Single-threaded: releases one specific block id back to d_pool's free list.
void device_release_block(
    DeviceSharedBlockPool* d_pool,
    std::uint32_t block_id);

// Launches one thread per neuron; thread i adds value (i * multiplier) to neuron i.
void device_add_one_per_neuron(
    DynamicVecVecView<std::uint32_t>* d_view,
    std::uint32_t n_neurons,
    std::uint32_t multiplier);

// Single-threaded: sequentially adds each value in `values` to `neuron_id`.
void device_add_multiple_to_neuron(
    DynamicVecVecView<std::uint32_t>* d_view,
    std::uint32_t neuron_id,
    const std::vector<std::uint32_t>& values);

// Returns element_idx-th element of neuron neuron_id for each (neuron_id, element_idx) pair.
std::vector<std::uint32_t> device_get_at(
    DynamicVecVecView<std::uint32_t>* d_view,
    const std::vector<std::uint32_t>& neuron_ids,
    const std::vector<std::uint32_t>& element_idxs);

// Uses DeviceVecVecCursor to drain all elements of neuron_id into a host vector.
// max_elems is an upper bound on the number of elements to collect.
std::vector<std::uint32_t> device_cursor_collect(
    DynamicVecVecView<std::uint32_t>* d_view,
    std::uint32_t neuron_id,
    std::uint32_t max_elems);

#endif // RELEARN_CUDA_ENABLED
