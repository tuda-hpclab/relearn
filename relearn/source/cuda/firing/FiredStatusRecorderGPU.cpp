/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FiredStatusRecorderGPU.h"

#include "Config.h"

#include "cuda/util/Util.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/helper/NeuronMonitor.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <range/v3/algorithm/fill.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

FiredStatusRecorderGPU::~FiredStatusRecorderGPU() = default;

void FiredStatusRecorderGPU::init(const number_neurons_type number_neurons) {
    RelearnException::check(number_neurons > 0, "FiredStatusRecorderGPU::init: Number of neurons must be greater than 0.");
    RelearnException::check(number_local_neurons == 0,
                            "FiredStatusRecorderGPU::init: FiredStatusRecorder already initialized.");

    for (auto& recorder : fired_recorder) {
        recorder.resize(number_neurons, 0U);
    }

    d_fired_recorder_ptrs.resize(3);
    all_fired_recorders.resize(number_fire_recorders * number_neurons);
    auto* dev_ptr = all_fired_recorders.get_device_ptr();
    for (auto i = 0U; i < fired_recorder.size(); i++) {
        // number_local_neurons (the member) isn't assigned until the end of this function, so it
        // reads 0 here -- using it made every period's offset collapse to dev_ptr+0, aliasing all
        // three periods onto the same memory (every fire incremented NeuronMonitor's slot 3 times
        // over, and AreaMonitor/Plasticity never had their own independent counts at all). Use the
        // number_neurons parameter instead, which holds the real value already.
        auto* offset = dev_ptr + number_neurons * i;
        d_fired_recorder_ptrs[i] = offset;
    }

    fired.resize(number_neurons, FiredStatus::Inactive);

    number_local_neurons = number_neurons;
}

void FiredStatusRecorderGPU::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(creation_count > 0,
                            "FiredStatusRecorderGPU::create_neurons: Creation count must be greater than 0.");
    RelearnException::check(number_local_neurons > 0,
                            "FiredStatusRecorderGPU::create_neurons: FiredStatusRecorder not initialized.");

    const auto current_size = number_local_neurons;
    const auto new_size = current_size + creation_count;

    // all_fired_recorders is laid out period-major ([period0: current_size neurons][period1:
    // current_size neurons]...); a plain resize() only appends at the very end, which would
    // leave period 1's/2's existing data sitting at their OLD offsets instead of the new,
    // wider ones. It was also resized to number_fire_recorders * creation_count instead of
    // * new_size -- undersized for the full neuron count, an out-of-bounds write as soon as the
    // kernel touched any neuron beyond creation_count. Rebuild explicitly: read back the old
    // (period-major, current_size-wide) data, copy each period into its new, wider offset, and
    // zero-fill the newly added neurons' slots.
    const auto old_data = std::as_const(all_fired_recorders).host();
    auto new_data = std::vector<counter_type>(number_fire_recorders * new_size, 0U);
    for (auto i = std::size_t{ 0 }; i < number_fire_recorders; ++i) {
        std::copy_n(old_data.begin() + static_cast<std::ptrdiff_t>(i * current_size), current_size,
                    new_data.begin() + static_cast<std::ptrdiff_t>(i * new_size));
    }
    all_fired_recorders = LazySyncedArray<counter_type>(std::move(new_data));

    auto* dev_ptr = all_fired_recorders.get_device_ptr();
    for (auto i = 0U; i < fired_recorder.size(); i++) {
        d_fired_recorder_ptrs[i] = dev_ptr + new_size * i;
    }

    for (auto& recorder : fired_recorder) {
        recorder.resize(new_size, 0U);
    }

    fired.resize(new_size, FiredStatus::Inactive);
    number_local_neurons = new_size;
}

void FiredStatusRecorderGPU::set_fired(const NeuronID neuron_id, const FiredStatus new_value) {
    const auto local_neuron_id = neuron_id.get_neuron_id();

    RelearnException::check(local_neuron_id < number_local_neurons,
                            "FiredStatusRecorderGPU::set_fired: Neuron ID out of bounds.");

    fired[local_neuron_id] = new_value;

    if (new_value != FiredStatus::Fired) {
        return;
    }

    for (auto& recorder : fired_recorder) {
        recorder[local_neuron_id]++;
    }

    // Required for tests
    for (auto i = 0U; i < fired_recorder.size(); i++) {
        const auto offset = number_local_neurons * i;
        all_fired_recorders[offset + local_neuron_id]++;
    }
}

std::span<const FiredStatusRecorderGPU::counter_type>
FiredStatusRecorderGPU::get_fired_recorder(const FireRecorderPeriod fire_recorder_period) const noexcept {
    // fired_recorder is only kept current by the host-side set_fired() path; every GPU neuron
    // model's update_activity() increments all_fired_recorders directly on the device instead
    // (see e.g. AEIFModel.cpp) and never touches fired_recorder, so reading fired_recorder here
    // would silently return stale (usually all-zero) counts in a GPU-driven simulation. Read
    // through all_fired_recorders instead -- .host() lazily syncs from the device if needed and
    // it holds the same per-period layout (offset = number_local_neurons * period_index).
    const auto period_index = static_cast<std::size_t>(fire_recorder_period);
    const auto offset = number_local_neurons * period_index;
    return all_fired_recorders.host().subspan(offset, number_local_neurons);
}

void FiredStatusRecorderGPU::reset(const FireRecorderPeriod period) {
    ranges::fill(fired_recorder[static_cast<std::size_t>(period)], 0U);

    // Only this period's slice -- NeuronMonitor/AreaMonitor/Plasticity are reset on independent
    // schedules (see Simulation.cpp), so zeroing the whole array here would wipe the other two
    // periods' still-accumulating counts.
    const auto period_index = static_cast<std::size_t>(period);
    const auto offset = number_local_neurons * period_index;
    set_memory_entry(all_fired_recorders.get_device_ptr() + offset, number_local_neurons, 0U);
    // set_memory_entry writes through the raw pointer, bypassing LazySyncedArray's own
    // modification tracking -- without this, the array wouldn't know its device copy just
    // changed, and a later host-side operator[] (in set_fired()) would keep incrementing the
    // stale pre-reset host cache instead of downloading the zeroed values first.
}

void FiredStatusRecorderGPU::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Hz", [this](const RelearnTypes::number_neurons_type neuron_id) {
        const auto& fire_hist = get_fired_recorder(FireRecorderPeriod::NeuronMonitor);
        const auto hz = static_cast<double>(fire_hist[neuron_id]) / Config::neuron_monitor_log_step;
        return utility::cast<float>(hz) * 1000.0F; }, []() { }, []() { });
}

void FiredStatusRecorderGPU::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this);

    auto recorder_footprint = 0ULL;
    for (const auto& recorder : fired_recorder) {
        recorder_footprint += recorder.capacity() * sizeof(unsigned int);
    }

    footprint->emplace("FiredStatusRecorder", my_footprint + recorder_footprint);
    footprint->emplace("FiredStatusRecorder GPU", d_fired_recorder_ptrs.get_memory_footprint() + all_fired_recorders.get_memory_footprint());
}
