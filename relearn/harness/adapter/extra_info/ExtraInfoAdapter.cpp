/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "ExtraInfoAdapter.h"

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/UpdateStatus.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"

#include <cpp-utility/ranges/Functional.hpp>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>

#include <memory>

void NeuronsExtraInfoAdapter::enable_all(const std::shared_ptr<NeuronsExtraInfo>& extra_info) {
    const auto disable_flags = extra_info->get_disable_flags();
    const auto ids_to_enable = NeuronIDRange::range(disable_flags.size()) | ranges::views::filter(utility::equal_to(UpdateStatus::Disabled), utility::lookup(disable_flags, &NeuronID::get_neuron_id)) | ranges::to_vector;

    if (!ids_to_enable.empty()) {
        extra_info->set_enabled_neurons(ids_to_enable);
    }
}

void NeuronsExtraInfoAdapter::disable_all(const std::shared_ptr<NeuronsExtraInfo>& extra_info) {
    const auto disable_flags = extra_info->get_disable_flags();

    const auto ids_to_disable = NeuronIDRange::range(disable_flags.size()) | ranges::views::filter(utility::equal_to(UpdateStatus::Enabled), utility::lookup(disable_flags, &NeuronID::get_neuron_id)) | ranges::to_vector;

    if (!ids_to_disable.empty()) {
        extra_info->set_disabled_neurons(ids_to_disable);
    }
}
