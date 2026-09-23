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

#include <fmt/ostream.h>

#include <ostream>
/**
 * This enums lists all types of synaptic input calculators
 */
enum class SynapticInputCalculatorType : char {
    Linear,
    Logarithmic,
    HyperbolicTangent,
};

/**
 * @brief Pretty-prints the synaptic input calculator type to the chosen stream
 * @param out The stream to which to print the synaptic input calculator
 * @param calculator_type The synaptic input calculator to print
 * @return The argument out, now altered with the synaptic input calculator
 */
inline std::ostream& operator<<(std::ostream& out, const SynapticInputCalculatorType& calculator_type) {
    if (calculator_type == SynapticInputCalculatorType::Linear) {
        return out << "Linear";
    }

    if (calculator_type == SynapticInputCalculatorType::Logarithmic) {
        return out << "Logarithmic";
    }

    if (calculator_type == SynapticInputCalculatorType::HyperbolicTangent) {
        return out << "Hyperbolic-Tangent";
    }

    return out;
}

template <>
struct fmt::formatter<SynapticInputCalculatorType> : ostream_formatter { };