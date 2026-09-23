#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <cpp-utility/ranges/Functional.hpp>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <vector>

class SynapsesAdapter {
public:
    template <typename SynapseType>
    static std::vector<SynapseType> invert_synapses(const std::vector<SynapseType>& synapses) {
        static constexpr auto to_weight = utility::element<2>;
        return synapses | ranges::views::transform([](SynapseType synapse) {
                   to_weight(synapse) = -to_weight(synapse);
                   return synapse;
               })
               | ranges::to_vector;
    }
};
