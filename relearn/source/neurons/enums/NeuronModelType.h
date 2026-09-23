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

#include "Macros.h"

#ifdef HOST_COMPILER
#include <fmt/ostream.h>

#include <ostream>
#include <string>
#endif

/**
 * This enums lists all types of neuron models
 */
enum class NeuronModelType : char {
    Poisson,
    Izhikevich,
    AEIF,
    FitzHughNagumo
};

#ifdef HOST_COMPILER
/**
 * @brief Returns the name of the chosen neuron model
 * @param neuron_model_type The neuron model
 * @return The name as a std::string
 */
[[nodiscard]] inline std::string stringify(const NeuronModelType neuron_model_type) {
    if (neuron_model_type == NeuronModelType::Poisson) {
        return "Poisson";
    }

    if (neuron_model_type == NeuronModelType::Izhikevich) {
        return "Izhikevich";
    }

    if (neuron_model_type == NeuronModelType::AEIF) {
        return "AEIF";
    }

    if (neuron_model_type == NeuronModelType::FitzHughNagumo) {
        return "FitzHughNagumo";
    }

    return "";
}


/**
 * @brief Pretty-prints the neuron model to the chosen stream
 * @param out The stream to which to print the neuron model
 * @param neuron_model_type The neuron model to print
 * @return The argument out, now altered with the neuron model
 */
inline std::ostream& operator<<(std::ostream& out, const NeuronModelType& neuron_model_type) {
    return out << stringify(neuron_model_type);
}

template <>
struct fmt::formatter<NeuronModelType> : ostream_formatter { };
#endif

