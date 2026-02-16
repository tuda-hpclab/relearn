/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElements.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/growthrate/GrowthrateCalculator.h"
#include "neurons/helper/NeuronMonitor.h"
#include "util/CalculatorUtil.h"
#include "util/RelearnException.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <span>
#include <vector>

void SynapticElements::init(const number_neurons_type number_neurons) {
    RelearnException::check(size == 0, "SynapticElements::init: init() was called previously");
    RelearnException::check(number_neurons > 0, "SynapticElements::init: number_neurons must be > 0");

    size = number_neurons;

    axons->init(number_neurons);
    dendrites->init(number_neurons);
}

void SynapticElements::create_neurons(const number_neurons_type creation_count) {
    RelearnException::check(size > 0, "SynapticElements::create_neurons: init() was not called before");
    RelearnException::check(creation_count > 0, "SynapticElements::create_neurons: creation_count must be > 0");

    const auto old_size = size;
    const auto new_size = old_size + creation_count;
    size = new_size;

    axons->create_neurons(creation_count);
    dendrites->create_neurons(creation_count);
}

void SynapticElements::disable_neurons(const std::span<const number_neurons_type> disabled_neuron_ids) {
    RelearnException::check(size > 0, "SynapticElements::disable_neurons: init() was not called before");
    if (disabled_neuron_ids.empty()) {
        return;
    }

    axons->disable_neurons(disabled_neuron_ids);
    dendrites->disable_neurons(disabled_neuron_ids);
}

void SynapticElements::register_neuron_monitor(NeuronMonitor& monitor) {
    axons->register_neuron_monitor(monitor);
    dendrites->register_neuron_monitor(monitor);
}

static void update_number_elements_openmp(const RelearnTypes::number_neurons_type size, const std::span<const UpdateStatus> disable_flags,
    const std::span<double> deltas, const std::span<const double> target_calcium, const std::span<const double> calcium,
    const std::span<const double> minimum_calcium, const std::shared_ptr<GrowthrateCalculator> growthrate_calculator) {

#pragma omp parallel for shared(size, disable_flags, deltas, target_calcium, calcium, minimum_calcium, growthrate_calculator) default(none)
    for (auto neuron_id = 0UL; neuron_id < size; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            deltas[neuron_id] = 0.0;
            continue;
        }

        const auto target_calcium_value = target_calcium[neuron_id];
        const auto current_calcium_value = calcium[neuron_id];

        const auto clamped_target = std::max(target_calcium_value, minimum_calcium[neuron_id]);

        const auto delta = gaussian_growth_curve(current_calcium_value, minimum_calcium[neuron_id], clamped_target);
        const auto growth_rate = growthrate_calculator->get_growth_rate(neuron_id);
        const auto inc = delta * growth_rate;

        deltas[neuron_id] = inc;

        growthrate_calculator->set_last_change(neuron_id, inc);
    }
}

void SynapticElements::update_number_elements(const step_type current_step, const std::span<const double> calcium, const std::span<const double> target_calcium) {
    RelearnException::check(extra_infos != nullptr, "SynapticElements::update_number_elements: extra_infos are null");
    const auto& disable_flags = extra_infos->get_disable_flags();

    RelearnException::check(calcium.size() == size, "SynapticElements::update_number_elements: calcium was not of the right size");
    RelearnException::check(target_calcium.size() == size, "SynapticElements::update_number_elements: target_calcium was not of the right size");
    RelearnException::check(disable_flags.size() == size, "SynapticElements::update_number_elements: disable_flags was not of the right size");

    const auto update_number_elements = [this, &calcium, &target_calcium, &disable_flags](const auto synaptic_element_type, const auto& growthrate_calculator) {
        auto minimum_calcium = get_minimum_calcium(synaptic_element_type);
        static auto deltas = std::vector<double>(size, 0.0);
        if (deltas.size() != size) {
            // This is here in case of an creation event
            // Having the vector static saves quite some time
            deltas.resize(size, 0.0);
        }

        update_number_elements_openmp(size, disable_flags, deltas, target_calcium, calcium, minimum_calcium, growthrate_calculator);

        add_to_delta(deltas, synaptic_element_type);
    };

    update_number_elements(SynapticElementType::Axon, growthrate_calculator_axon);
    update_number_elements(SynapticElementType::DendriteExcitatory, growthrate_calculator_excitatory_dendrites);
    update_number_elements(SynapticElementType::DendriteInhibitory, growthrate_calculator_inhibitory_dendrites);

    growthrate_calculator_axon->update_growth_rate(current_step);
    growthrate_calculator_excitatory_dendrites->update_growth_rate(current_step);
    growthrate_calculator_inhibitory_dendrites->update_growth_rate(current_step);
}
