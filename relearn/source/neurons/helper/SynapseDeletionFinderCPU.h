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

#include "SynapseDeletionFinderBase.h"

#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "types/CommunicationTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <span>
#include <utility>

/**
 * CPU implementation of the synapse deletion finder: synapses to delete are found and committed
 * via host loops driven by the (virtual) find_synapses_on_neuron() hook.
 */
class SynapseDeletionFinderCPU : public SynapseDeletionFinderBase {
public:
    /**
     * @brief Commits the updates for the synaptic elements, deletes synapses in the network graph,
     *      exchanges the deletions between MPI ranks, and commits the deletions from other ranks as well
     * @return The number of deleted synapses that are initiated by (1) the local axons and (2) the local dendrites
     */
    [[nodiscard]] std::pair<number_synapse_type, number_synapse_type> delete_synapses();

protected:
    [[nodiscard]] virtual RelearnTypes::comm_map_deletion<SynapseDeletionRequest> find_synapses_to_delete(ElementType element_type, std::span<const SignalType> signal_types,
                                                                                                          std::span<const counter_type> number_deletions);

    [[nodiscard]] virtual RelearnTypes::comm_map_deletion<SynapseDeletionRequest> find_synapses_to_delete(ElementType element_type, SignalType signal_types,
                                                                                                          std::span<const counter_type> number_deletions);

    [[nodiscard]] number_synapse_type commit_deletions(const RelearnTypes::comm_map_deletion<SynapseDeletionRequest>& deletions, mpiPP::MPIRank my_rank);
};
