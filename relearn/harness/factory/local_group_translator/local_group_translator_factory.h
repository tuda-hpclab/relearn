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

#include "util/NeuronID.h"

#include <memory>
#include <random>

class LocalGroupTranslator;

class LocalGroupTranslatorFactory {
public:
    static std::shared_ptr<LocalGroupTranslator> get_randomized_group_translator(std::mt19937& mt);

    static std::shared_ptr<LocalGroupTranslator> get_randomized_group_translator(NeuronID::value_type num_neurons, std::mt19937& mt);
};
