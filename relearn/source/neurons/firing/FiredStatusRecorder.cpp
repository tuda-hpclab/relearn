/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FiredStatusRecorder.h"

#include "neurons/enums/FiredStatus.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIInfo.h"

#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#include <range/v3/algorithm/fill.hpp>

#include <cstddef>
#include <memory>
#include <span>

void FiredStatusRecorder::init(const number_neurons_type number_neurons) {
    RelearnException::check(number_neurons > 0, "FiredStatusRecorder::init: Number of neurons must be greater than 0.");
    RelearnException::check(number_local_neurons == 0, "FiredStatusRecorder::init: FiredStatusRecorder already initialized.");

    for (auto& recorder : fired_recorder) {
        recorder.resize(number_neurons, 0U);
    }

    fire_history.resize(number_neurons, boost::dynamic_bitset<>(fire_history_length));
    fired.resize(number_neurons, FiredStatus::Inactive);
    number_local_neurons = number_neurons;
}

void FiredStatusRecorder::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(creation_count > 0, "FiredStatusRecorder::create_neurons: Creation count must be greater than 0.");
    RelearnException::check(number_local_neurons > 0, "FiredStatusRecorder::create_neurons: FiredStatusRecorder not initialized.");

    const auto current_size = number_local_neurons;
    const auto new_size = current_size + creation_count;

    for (auto& recorder : fired_recorder) {
        recorder.resize(new_size, 0U);
    }

    fire_history.resize(new_size, boost::dynamic_bitset<>(fire_history_length));
    fired.resize(new_size, FiredStatus::Inactive);
    number_local_neurons = new_size;
}

void FiredStatusRecorder::disable_neurons(const std::span<const NeuronID> neuron_ids) {
    for (const auto neuron_id : neuron_ids) {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < number_local_neurons, "FiredStatusRecorder::disable_neurons: Neuron ID out of bounds.");

        for (auto& recorder : fired_recorder) {
            recorder[local_neuron_id] = 0U;
        }

        fired[local_neuron_id] = FiredStatus::Inactive;
    }
}

void FiredStatusRecorder::set_fired(const NeuronID neuron_id, const FiredStatus new_value) {
    const auto local_neuron_id = neuron_id.get_neuron_id();

    RelearnException::check(local_neuron_id < number_local_neurons, "FiredStatusRecorder::set_fired: Neuron ID out of bounds.");

    fire_history[local_neuron_id] <<= 1;
    fire_history[local_neuron_id][0] = static_cast<bool>(new_value);

    fired[local_neuron_id] = new_value;

    if (new_value != FiredStatus::Fired) {
        return;
    }

    for (auto& recorder : fired_recorder) {
        recorder[local_neuron_id]++;
    }
}

void FiredStatusRecorder::set_fired(const NeuronID::value_type neuron_id, const FiredStatus new_value) {
    RelearnException::check(neuron_id < number_local_neurons, "FiredStatusRecorder::set_fired: Neuron ID out of bounds.");

    fire_history[neuron_id] <<= 1;
    fire_history[neuron_id][0] = static_cast<bool>(new_value);

    fired[neuron_id] = new_value;

    if (new_value != FiredStatus::Fired) {
        return;
    }

    for (auto& recorder : fired_recorder) {
        recorder[neuron_id]++;
    }
}

void FiredStatusRecorder::register_neuron_monitor(NeuronMonitor& monitor) {
    const auto inv = 1.0f / static_cast<float>(Config::fire_history_reset_step);

    monitor.register_paramter("Hz", [this, inv](const RelearnTypes::number_neurons_type neuron_id) {
        const auto& fire_hist = fire_history[neuron_id];
        const auto hz = fire_hist.count();
        return static_cast<float>(hz) * 1000.0F * inv;
    });
}

std::span<const FiredStatusRecorder::counter_type> FiredStatusRecorder::get_fired_recorder(const FireRecorderPeriod fire_recorder_period) const noexcept {
    return fired_recorder[static_cast<std::size_t>(fire_recorder_period)];
}

void FiredStatusRecorder::reset(const FireRecorderPeriod period) {
    ranges::fill(fired_recorder[static_cast<std::size_t>(period)], 0U);
}

const boost::dynamic_bitset<>& FiredStatusRecorder::get_fire_history(const NeuronID neuron_id) const {
    const auto actual_id = neuron_id.get_neuron_id();
    RelearnException::check(actual_id < number_local_neurons, "NeuronsExtraInfo::get_fire_history: neuron_id must be smaller than number_local_neurons but was {}", neuron_id);

    return fire_history[actual_id];
}

boost::dynamic_bitset<> FiredStatusRecorder::get_fire_history(const RankNeuronId& rank_neuron_id) const {
    const auto rank = rank_neuron_id.get_rank();
    const auto neuron_id = rank_neuron_id.get_neuron_id();

    if (rank == mpiPP::MPIInfo::get_my_rank()) {
        return get_fire_history(neuron_id);
    }
    RelearnException::fail("Fix this!");
    // const auto data = MPIWrapper::get_from_window<boost::dynamic_bitset<>>(MPIWindow::FireHistory, rank.get_rank(), neuron_id.get_neuron_id(), 1);
    return {};
}

void FiredStatusRecorder::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this);

    auto recorder_footprint = 0ULL;
    for (const auto& recorder : fired_recorder) {
        recorder_footprint += recorder.capacity() * sizeof(unsigned int);
    }

    footprint->emplace("FiredStatusRecorder", my_footprint + recorder_footprint);
}
