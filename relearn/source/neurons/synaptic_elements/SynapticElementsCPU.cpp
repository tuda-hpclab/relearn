/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElementsCPU.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/growthrate/GrowthrateCalculator.h"
#include "types/BasicTypes.h"
#include "util/RelearnException.h"

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

static void update_number_elements_openmp(const RelearnTypes::number_neurons_type size, const std::span<const UpdateStatus> disable_flags,
                                          const std::span<RelearnTypes::grown_type> deltas, const std::span<const RelearnTypes::calcium_type> target_calcium, const std::span<const RelearnTypes::calcium_type> calcium,
                                          const std::span<const RelearnTypes::calcium_type> minimum_calcium, const std::shared_ptr<GrowthrateCalculator> growthrate_calculator) {

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

void SynapticElementsCPU::update_number_elements(const step_type current_step, const std::span<const calcium_type> calcium, const std::span<const calcium_type> target_calcium) {
    RelearnException::check(extra_infos != nullptr, "SynapticElements::update_number_elements: extra_infos are null");
    const auto& disable_flags = extra_infos->get_disable_flags();

    RelearnException::check(calcium.size() == size, "SynapticElements::update_number_elements: calcium was not of the right size");
    RelearnException::check(target_calcium.size() == size, "SynapticElements::update_number_elements: target_calcium was not of the right size");
    RelearnException::check(disable_flags.size() == size, "SynapticElements::update_number_elements: disable_flags was not of the right size");

    const auto update_number_elements = [this, &calcium, &target_calcium, &disable_flags](const auto synaptic_element_type, const auto& growthrate_calculator) {
        auto minimum_calcium = get_minimum_calcium(synaptic_element_type);
        static auto deltas = std::vector<grown_type>(size, 0.0);
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
