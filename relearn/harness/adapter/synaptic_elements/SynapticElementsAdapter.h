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

#include "neurons/enums/SynapticElementType.h"
#include "neurons/synaptic_elements/SynapticElements.h"

#include <memory>
#include <random>
#include <span>

class SynapticElementsAdapter {
public:
    static void increase_grown_axons(const std::shared_ptr<SynapticElements>& synaptic_elements, std::span<const double> grown_elements);

    static void increase_grown_dendrites(const std::shared_ptr<SynapticElements>& synaptic_elements, std::span<const double> grown_elements, SignalType signal_type);

    static void increase_connected_axons(const std::shared_ptr<SynapticElements>& synaptic_elements, std::span<const unsigned int> connected_elements);

    static void increase_connected_dendrites(const std::shared_ptr<SynapticElements>& synaptic_elements, std::span<const unsigned int> connected_elements, SignalType signal_type);

    static void grow_and_connect(const std::shared_ptr<SynapticElements>& synaptic_elements, std::mt19937& mt);

    static void set_growthrate_calculators(const std::shared_ptr<SynapticElements>& synaptic_elements);
};
