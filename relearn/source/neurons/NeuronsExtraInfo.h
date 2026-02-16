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

#include "Config.h"
#include "Types.h"
#include "Types3.h"

#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"

#include <memory>
#include <span>
#include <vector>

namespace utility {
class MemoryFootprint;
}

/**
 * An object of type NeuronsExtraInfo additional information of neurons.
 * For a single neuron, these additional information are: its x-, y-, and z- positionand
 * which neurons update their electrical activity and their plasticity.
 */
class NeuronsExtraInfo {
public:
    using position_type = RelearnTypes::position_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    NeuronsExtraInfo() = default;

    /**
     * @brief Initializes a NeuronsExtraInfo that holds at most the given number of neurons.
     *      Must only be called once. Sets up all neurons so that they update, but does not initialize the positions.
     * @param number_neurons The number of neurons, greater than 0
     * @exception Throws an RelearnException if number_neurons is 0 or if called multiple times.
     */
    void init(number_neurons_type number_neurons);

    /**
     * @brief Inserts additional neurons with x-, y-, z- positions randomly picked from already existing ones.
     *      Sets all neurons to update. Only works with one MPI rank.
     * @param creation_count The number of new neurons, greater than 0
     * @exception Throws an RelearnException if creation_count is 0, if the positions are empty, or if more than one MPI rank is active
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Marks the specified neurons as enabled
     * @param enabled_neurons The neuron ids from the now enabled neurons
     * @exception Throws a RelearnException if one of the specified ids exceeds the number of stored neurons
     */
    void set_enabled_neurons(std::span<const NeuronID> enabled_neurons);

    /**
     * @brief Marks the specified neurons as disabled
     * @param disabled_neurons The neuron ids from the now disabled neurons
     * @exception Throws a RelearnException if one of the specified ids exceeds the number of stored neurons
     */
    void set_disabled_neurons(std::span<const NeuronID> disabled_neurons);

    /**
     * @brief Marks the specified neurons as static
     * @param static_neurons The neuron ids from the now static neurons
     * @exception Throws a RelearnException if one of the specified ids exceeds the number of stored neurons
     */
    void set_static_neurons(std::span<const NeuronID> static_neurons);

    /**
     * @brief Overwrites the current positions with the supplied ones
     * @param pos The new positions, must have the same size as neurons are stored
     * @exception Throws an RelearnException if pos.empty() or if the number of supplied elements does not match the number of stored neurons
     */
    void set_positions(std::vector<position_type> pos) {
        RelearnException::check(!pos.empty(), "NeuronsExtraInformation::set_positions: New positions are empty");
        RelearnException::check(size == pos.size(), "NeuronsExtraInformation::set_positions: Size does not match group names count");
        positions = std::move(pos);
    }

    /**
     * @brief Returns the currently stored positions
     * @return The currently stored positions
     */
    [[nodiscard]] std::span<const position_type> get_positions() const noexcept {
        return positions;
    }

    /**
     * @brief Returns a position_type with the x-, y-, and z- positions for a specified neuron.
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     */
    [[nodiscard]] position_type get_position(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::get_position: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < positions.size(), "NeuronsExtraInfo::get_position: neuron_id must be smaller than positions.size() but was {}", neuron_id);
        return positions[local_neuron_id];
    }

    /**
     * @brief Checks for a neuron if it updates its electrical activity
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     * @return True iff the neuron updates its electrical activity
     */
    [[nodiscard]] bool does_update_electrical_actvity(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::does_update_electrical_actvity: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < update_status.size(), "NeuronsExtraInfo::does_update_electrical_actvity: neuron_id must be smaller than update_status.size() but was {}", neuron_id);

        return update_status[local_neuron_id] != UpdateStatus::Disabled;
    }

    /**
     * @brief Checks for a neuron if it updates its plasticity
     * @param neuron_id The local id of the neuron, i.e., from [0, num_local_neurons)
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     * @return True iff the neuron updates its plasticity
     */
    [[nodiscard]] bool does_update_plasticity(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::does_update_plasticity: neuron_id must be smaller than size but was {}", neuron_id);
        RelearnException::check(local_neuron_id < update_status.size(), "NeuronsExtraInfo::does_update_plasticity: neuron_id must be smaller than update_status.size() but was {}", neuron_id);

        return update_status[local_neuron_id] == UpdateStatus::Enabled;
    }

    /**
     * @brief Returns the disable flags for the neurons
     * @return The disable flags
     */
    [[nodiscard]] std::span<const UpdateStatus> get_disable_flags() const noexcept {
        return update_status;
    }

    /**
     * @brief Returns the number of stored neurons
     * @return The number of neurons
     */
    [[nodiscard]] number_neurons_type get_size() const noexcept {
        return size;
    }

    /**
     * @brief Translates a collection of local neuron ids to their positions
     * @param local_neurons The local neuron ids
     * @return The position of the local neurons, has the same size as the argument
     */
    [[nodiscard]] RelearnTypes::comm_map_position<RelearnTypes::position_type> get_positions_for(const RelearnTypes::comm_map_position<NeuronID>& local_neurons) const;

    /**
     * @brief Clears the deletion log
     */
    void reset_deletion_log() {
        deletions_log.clear();
        deletions_log.resize(size, {});
    }

    /**
     * @brief Mark a deletion for the specified neuron
     * @param neuron_id The neuron to mark
     * @param source_neuron The other neuron
     * @param weight The weight of the deletion
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     */
    void mark_deletion(const NeuronID neuron_id, const RankNeuronId& source_neuron, const RelearnTypes::plastic_synapse_weight weight) {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::mark_deletion: neuron_id must be smaller than size but was {}", neuron_id);

        deletions_log[local_neuron_id].emplace_back(source_neuron, weight);
    }

    /**
     * @brief Returns the deletion log for a specified neuron
     * @param neuron_id The local neuron's id
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     * @return The deletion log of the neuron
     */
    [[nodiscard]] std::span<const std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>> get_deletions_log(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInfo::get_deletions_log: neuron_id must be smaller than size but was {}", neuron_id);

        return deletions_log[local_neuron_id];
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint);

private:
    number_neurons_type size{ 0 };

    std::vector<position_type> positions{};
    std::vector<UpdateStatus> update_status{};

    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> deletions_log{};
};
