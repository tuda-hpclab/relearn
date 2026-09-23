#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/helper/SynapseDeletionFinderBase.h"
#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceArray.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "types/CommunicationTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <optional>
#include <span>
#include <utility>

/**
 * GPU implementation of the synapse deletion finder: synapses to delete are found and committed
 * via CUDA kernels. Unlike the CPU version, this does not dispatch through find_synapses_on_neuron()
 * -- there is currently only a GPU kernel for random deletion, so all strategies share this same
 * implementation on the GPU (InverseLengthSynapseDeletionFinder does not override it, see
 * SynapseDeletionFinder.h).
 */
class SynapseDeletionFinderGPU : public SynapseDeletionFinderBase {
public:
    /**
     * @brief Initializes the deletion finder to hold the given number of neurons, additionally
     *      registering the device RNG stream that find_synapses_to_delete draws from (random_key).
     * @param number_neurons The number of neurons, must be > 0
     */
    void init(RelearnTypes::number_neurons_type number_neurons) override;

    /**
     * @brief Commits the updates for the synaptic elements, deletes synapses in the network graph,
     *      exchanges the deletions between MPI ranks, and commits the deletions from other ranks as well
     * @return The number of deleted synapses that are initiated by (1) the local axons and (2) the local dendrites
     */
    [[nodiscard]] std::pair<number_synapse_type, number_synapse_type> delete_synapses();

protected:
    RelearnTypes::comm_map_deletion<SynapseDeletionRequest> find_synapses_to_delete(ElementType element_type, std::optional<SignalType> single_signal_type, std::span<const SignalType> signal_types, const DeviceArray<CudaConfig::synaptic_count_type>& number_deletions);

    [[nodiscard]] number_synapse_type commit_deletions(const RelearnTypes::comm_map_deletion<SynapseDeletionRequest>& deletions, mpiPP::MPIRank my_rank);
};
