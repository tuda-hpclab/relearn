/** @file SpikePreparation.h
 * @brief CUDA entry points for compressing fired-neuron information before MPI communication.
 *
 * After the neuron-activity update, each rank must broadcast which of its neurons fired.
 * These functions collect the fire-status array and the outgoing edge lists into a compact
 * format (neuron-ID list or bitvector) that can be efficiently sent to remote ranks.
 */
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

#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/util/NeuronsExtraInfoHandle.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/enums/FiredStatus.h"

#include <memory>
#include <vector>

struct NetworkHandle;

/** Per-neuron fire status for the local rank, as read by the spike-preparation kernels. */
struct FiredStatusHandle {
    const FiredStatus* fired;
};

/** Per-rank send counts and a flat device array of fired-neuron IDs. */
struct PreparedSpikes {
    std::vector<int> send_counts;
    DeviceArray<CudaConfig::number_neurons_type> fired_neuron_ids;
};

/**
 * @brief Collects fired-neuron IDs using atomic counters; optionally sorts the result.
 *
 * This variant iterates over the network graph instead of the full neuron array, which is
 * more efficient when the fraction of firing neurons is small.
 *
 * @param fired_handle  Per-neuron fire status for this rank's neurons.
 * @param network       GPU view of the local network graph (provides outgoing edge info).
 * @param info_handle   Per-neuron metadata; only .my_rank is read.
 * @param sort_spikes   If true, sort the collected IDs within each rank's segment.
 * @param sort_stream   Optional stream to use for sorting (uses default stream if nullptr).
 * @return Per-rank send counts and a flat device array of fired-neuron IDs.
 */
PreparedSpikes prepare_spikes(FiredStatusHandle fired_handle, NetworkHandle network, NeuronsExtraInfoGPUHandleConst info_handle, bool sort_spikes, const std::shared_ptr<StreamWrapper>& sort_stream);
