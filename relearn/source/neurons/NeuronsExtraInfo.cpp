/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronsExtraInfo.h"

#include "Types.h"
#include "Types3.h"

#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronID.h"
#include "util/Random.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIInfo.h"

#include <range/v3/algorithm/fill.hpp>
#include <range/v3/view/transform.hpp>

#include <memory>
#include <span>

void NeuronsExtraInfo::init(const number_neurons_type number_neurons) {
    RelearnException::check(number_neurons > 0, "NeuronsExtraInfo::init: number_neurons must be larger than 0.");
    RelearnException::check(size == 0, "NeuronsExtraInfo::init: NeuronsExtraInfo initialized two times, its size is already {}", size);

    size = number_neurons;
    update_status.resize(number_neurons, UpdateStatus::Enabled);
    deletions_log.resize(number_neurons, {});
}

void NeuronsExtraInfo::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(!positions.empty(), "NeuronsExtraInfo::create_neurons: Was not initialized");
    RelearnException::check(creation_count != 0, "Cannot add 0 neurons");

    const auto num_ranks = mpiPP::MPIInfo::get_number_ranks();

    RelearnException::check(num_ranks == 1, "NeuronsExtraInfo::create_neurons: Cannot create neurons if more than 1 MPI rank is computing");

    const auto current_size = size;
    const auto new_size = current_size + creation_count;

    update_status.resize(new_size, UpdateStatus::Enabled);
    deletions_log.resize(new_size, {});
    positions.resize(new_size);

    for (auto i = current_size; i < new_size; i++) {
        const auto x_it = RandomHolder::get_random_uniform_double(RandomHolderKey::NeuronsExtraInformation, 0.0, 1.0);
        const auto y_it = RandomHolder::get_random_uniform_double(RandomHolderKey::NeuronsExtraInformation, 0.0, 1.0);
        const auto z_it = RandomHolder::get_random_uniform_double(RandomHolderKey::NeuronsExtraInformation, 0.0, 1.0);

        const auto random_pos_x = static_cast<number_neurons_type>(x_it * static_cast<double>(current_size));
        const auto random_pos_y = static_cast<number_neurons_type>(y_it * static_cast<double>(current_size));
        const auto random_pos_z = static_cast<number_neurons_type>(z_it * static_cast<double>(current_size));

        const auto x_pos = positions[random_pos_x].get_x();
        const auto y_pos = positions[random_pos_y].get_y();
        const auto z_pos = positions[random_pos_z].get_z();

        positions[i] = { x_pos, y_pos, z_pos };
    }

    size = new_size;
}

void NeuronsExtraInfo::set_enabled_neurons(const std::span<const NeuronID> enabled_neurons) {
    const auto get_update_status = [this](const auto& neuron_id) -> UpdateStatus& {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < size, "NeuronsExtraInformation::set_enabled_neurons: NeuronID {} is too large: {}", neuron_id);
        RelearnException::check(update_status[local_neuron_id] == UpdateStatus::Disabled, "NeuronsExtraInformation::set_enabled_neurons: Cannot enable a not disabled neuron");

        return update_status[local_neuron_id];
    };
    RelearnException::check(!enabled_neurons.empty(), "NeuronsExtraInformation::set_enabled_neurons: enabled_neurons is empty");

    ranges::fill(enabled_neurons | ranges::views::transform(get_update_status), UpdateStatus::Enabled);
}

void NeuronsExtraInfo::set_disabled_neurons(const std::span<const NeuronID> disabled_neurons) {
    const auto get_update_status = [this](const auto& neuron_id) -> UpdateStatus& {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < size, "NeuronsExtraInformation::set_disabled_neurons: NeuronID {} is too large: {}", neuron_id, size);

        auto& status = update_status[local_neuron_id];

        RelearnException::check(status != UpdateStatus::Static,
                                "NeuronsExtraInformation::set_disabled_neurons: Cannot disable a static neuron");
        RelearnException::check(status != UpdateStatus::Disabled,
                                "NeuronsExtraInformation::set_disabled_neurons: Cannot disable a disabled neuron");

        return status;
    };

    ranges::fill(disabled_neurons | ranges::views::transform(get_update_status), UpdateStatus::Disabled);
}

void NeuronsExtraInfo::set_static_neurons(const std::span<const NeuronID> static_neurons) {
    const auto get_update_status = [this](const auto& neuron_id) -> UpdateStatus& {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < this->size, "NeuronsExtraInformation::set_static_neurons: NeuronID {} is too large", neuron_id);

        return update_status[local_neuron_id];
    };

    ranges::fill(static_neurons | ranges::views::transform(get_update_status), UpdateStatus::Static);
}

RelearnTypes::comm_map_position<RelearnTypes::position_type> NeuronsExtraInfo::get_positions_for(const RelearnTypes::comm_map_position<NeuronID>& local_neurons) const {
    const auto number_ranks = local_neurons.get_number_ranks();
    const auto size_hint = local_neurons.size();

    auto cm_positions = RelearnTypes::comm_map_position<RelearnTypes::position_type>(number_ranks, size_hint);

    if (local_neurons.empty()) {
        return cm_positions;
    }

    cm_positions.resize(local_neurons.get_request_sizes());

    for (const auto& [source_rank, requests] : local_neurons) {
        for (auto i = 0U; i < requests.size(); i++) {
            const auto neuron_id = requests[i];
            const auto& position = get_position(neuron_id);
            cm_positions.set_request(source_rank, i, position);
        }
    }

    return cm_positions;
}

void NeuronsExtraInfo::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this)
                              + (positions.capacity() * sizeof(position_type))
                              + (update_status.capacity() * sizeof(UpdateStatus));
    footprint->emplace("ExtraInfos", my_footprint);
}
