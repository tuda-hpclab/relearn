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

#include "cuda/memory/SharedBlockPool.h"

enum class LayoutType;

struct EdgeHandle {
    LayoutType layout_type{};
    void* view_impl{};

    // Set from the owning GPUEdgesBase::wide_ids. Tells every consumer that reinterprets
    // view_impl which MemoryPoolView<IdT>/OnlyOutgoingView<IdT> template instantiation it actually points
    // to: std::uint32_t when true, SmallNeuronIdType when false. See CudaConfig::use_wide_neuron_ids.
    bool wide_ids{ false };
};

struct NetworkHandle {
    EdgeHandle incoming_local_handle{};
    EdgeHandle incoming_distant_handle{};
    EdgeHandle outgoing_local_handle{};
    EdgeHandle outgoing_distant_handle{};

    // Host-side pool backing all four edge handles above. Lets callers that are about to launch a
    // kernel with a known worst-case block demand (e.g. ExchangeAlgorithm) call
    // shared_overflow_pool->ensure_capacity(...) beforehand, without threading a separate
    // parameter through every layer that passes a NetworkHandle around.
    SharedBlockPool* shared_overflow_pool{};
};