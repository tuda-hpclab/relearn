/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "local_group_translator_factory.h"

#include "neurons/LocalGroupTranslator.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"

#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neurons/neurons_factory.h"

#include <memory>
#include <random>
#include <vector>

std::shared_ptr<LocalGroupTranslator> LocalGroupTranslatorFactory::get_randomized_group_translator(std::mt19937& mt) {
    const auto num_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto group_id_to_group_name = RelearnTypes::group_names{};
    auto neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{};

    NeuronsFactory::generate_random_neuron_groups(neuron_id_to_group_ids, group_id_to_group_name, num_neurons, mt);

    return std::make_shared<LocalGroupTranslator>(group_id_to_group_name, neuron_id_to_group_ids);
}

std::shared_ptr<LocalGroupTranslator> LocalGroupTranslatorFactory::get_randomized_group_translator(const NeuronID::value_type num_neurons, std::mt19937& mt) {
    auto group_id_to_group_name = RelearnTypes::group_names{};
    auto neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{};

    NeuronsFactory::generate_random_neuron_groups(neuron_id_to_group_ids, group_id_to_group_name, num_neurons, mt);

    return std::make_shared<LocalGroupTranslator>(group_id_to_group_name, neuron_id_to_group_ids);
}
