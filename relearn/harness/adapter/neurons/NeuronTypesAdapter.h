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

#include <memory>
#include <random>

class NeuronsExtraInfo;

class NeuronTypesAdapter {
public:
    static void disable_neurons(size_t number_neurons, std::shared_ptr<NeuronsExtraInfo> extra_infos, std::mt19937& mt);

    static void disable_neurons(size_t number_neurons, size_t number_disabled, const std::shared_ptr<NeuronsExtraInfo>& extra_infos, std::mt19937& mt);
};
