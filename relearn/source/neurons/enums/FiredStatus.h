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
 * An instance of this enum symbolizes if a neuron fired in the current simulation step.
 * That is, the model of electrical activity determined that the activity spiked and thus
 * the neuron gives input to its neighbors.
 */
enum class FiredStatus : char {
    Inactive = 0,
    Fired = 1,
};
/**
 * @brief Pretty-prints the fired status to the chosen stream
 * @param out The stream to which to print the fired status
 * @param fired_status The fired status to print
 * @return The argument out, now altered with the fired status
 */
inline std::ostream& operator<<(std::ostream& out, const FiredStatus fired_status) {
    if (fired_status == FiredStatus::Fired) {
        return out << "Fired";
    }

    if (fired_status == FiredStatus::Inactive) {
        return out << "Inactive";
    }

    return out;
}

template <>
struct fmt::formatter<FiredStatus> : ostream_formatter { };
