/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapseLoader.h"

#include "sim/Essentials.h"
#include "types/BasicTypes.h"
#include "util/Timers.h"

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/traits.hpp>
#include <range/v3/view/transform.hpp>

#include <memory>

SynapseLoader::synapses_pair_type SynapseLoader::load_synapses(const std::unique_ptr<Essentials>& essentials) {
    const auto& synapses_pair = internal_load_synapses();
    const auto& [synapses_static, synapses_plastic] = synapses_pair;
    const auto& [local_synapses, in_synapses, out_synapses] = synapses_plastic;

    const auto sum_weights = []<typename SynapsesType>(const SynapsesType& synapses) {
        return ranges::accumulate(synapses | ranges::views::transform(&ranges::range_value_t<SynapsesType>::get_weight), RelearnTypes::plastic_synapse_weight{ 0U });
    };

    const auto total_local_weight = sum_weights(local_synapses);
    const auto total_in_weight = sum_weights(in_synapses);
    const auto total_out_weight = sum_weights(out_synapses);

    essentials->insert("Loaded-Local-Synapses", local_synapses.size());
    essentials->insert("Loaded-Local-Synapses-Weight", total_local_weight);
    essentials->insert("Loaded-In-Synapses", in_synapses.size());
    essentials->insert("Loaded-In-Synapses-Weight", total_in_weight);
    essentials->insert("Loaded-Out-Synapses", out_synapses.size());
    essentials->insert("Loaded-Out-Synapses-Weight", total_out_weight);

    return synapses_pair;
}
