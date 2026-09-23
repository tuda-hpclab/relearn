/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticEquallyWeightedActivityInputGPU.h"

#include "cuda/input/Handle.h"
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/input/ActivityInput.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <span>

std::optional<EventWrapper> SynapticEquallyWeightedActivityInputGPU::update_local_input(
    const number_neurons_type first, const number_neurons_type last, activity_type* d_input,
    const FiredStatus* d_fired, const activity_type* /*d_scales*/,
    const std::shared_ptr<StreamWrapper>& stream_wrapper) {
    // return update_synaptic_equally_weighted_local_input_entry(first, last, d_input, d_fired,
    //                                                  incoming_edges_handle, stream_wrapper, mpiPP::MPIInfo::get_my_rank().get_rank());
    const std::unique_ptr<FireStatusCommunicatorHandle> ff = std::make_unique<FireStatusLocalVectorHandle>(d_fired);
    // IterationMode::Spikes is a "push" kernel that iterates fired SOURCE neurons restricted to
    // [first, last) and writes to whatever targets their edges reach, which can land outside
    // [first, last) -- so it only matches CPU semantics when [first, last) spans all neurons (see
    // the GTEST_SKIP notes on testUpdateInputRangeFullNetworkGraph/testUpdateInputMultipleRangesFullNetworkGraph
    // for why IterationMode::Neurons ("pull", scoped by TARGET) can't be used instead here).
    const LaunchConfig config{
        WeightMode::Weighted,
        SpikeMode::LocalVector,
        NetworkMode::Default,
        FireInformation::LocalVector,
        IterationMode::Spikes,
    };
    const LaunchHandles handle{ get_network_graph()->get_gpu_handle_const(), ff.get() };
    return launch(static_cast<CudaConfig::number_neurons_type>(first), static_cast<CudaConfig::number_neurons_type>(last), d_input, config, handle, stream_wrapper, mpiPP::MPIInfo::get_my_rank().get_rank(), true, false, synapse_conductance);
}

std::optional<EventWrapper> SynapticEquallyWeightedActivityInputGPU::update_distant_input(
    const number_neurons_type first, const number_neurons_type last, activity_type* d_input,
    const activity_type* /*d_scales*/,
    std::unique_ptr<FireStatusCommunicatorHandle>& fire_status_handle,
    const std::shared_ptr<StreamWrapper>& stream_wrapper) {
    if (get_number_ranks() == 1) {
        return { std::nullopt };
    }

    auto do_binary_search = Config::do_binary_search;
    const auto config = LaunchConfig{ WeightMode::Weighted, do_binary_search ? SpikeMode::BinarySearch : SpikeMode::Set, do_binary_search ? NetworkMode::Sorted : NetworkMode::Default, FireInformation::NeuronIDs, IterationMode::Neurons };
    const LaunchHandles handle{ get_network_graph()->get_gpu_handle_const(), fire_status_handle.get() };
    return launch(static_cast<CudaConfig::number_neurons_type>(first), static_cast<CudaConfig::number_neurons_type>(last), d_input, config, handle, stream_wrapper, mpiPP::MPIInfo::get_my_rank().get_rank(), false, false, synapse_conductance);
}
