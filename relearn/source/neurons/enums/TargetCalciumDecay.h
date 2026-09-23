#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <fmt/ostream.h>

#include <ostream>

/**
 * An instance of this enum symbolizes if the calcium target decays absolute, relative, or not at all
 */
enum class TargetCalciumDecay : char {
    None = 0,
    Absolute = 1,
    Relative = 2,
};

/**
 * @brief Pretty-prints the target calcium decay to the chosen stream
 * @param out The stream to which to print the target calcium decay
 * @param decay_type The target calcium decay to print
 * @return The argument out, now altered with the target calcium decay
 */
inline std::ostream& operator<<(std::ostream& out, const TargetCalciumDecay decay_type) {
    if (decay_type == TargetCalciumDecay::None) {
        return out << "None";
    }

    if (decay_type == TargetCalciumDecay::Absolute) {
        return out << "Absolute";
    }

    if (decay_type == TargetCalciumDecay::Relative) {
        return out << "Relative";
    }

    return out;
}

template <>
struct fmt::formatter<TargetCalciumDecay> : ostream_formatter { };
