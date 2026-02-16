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

#include <fmt/ostream.h>

#include <ostream>

/**
 * An instance of this enum signals if a neuron should be updated or not.
 */
enum class UpdateStatus : char {
    Disabled = 0,
    Enabled = 1,
    Static = 2
};

/**
 * @brief Pretty-prints the update status to the chosen stream
 * @param out The stream to which to print the signal type
 * @param update_status The update status to print
 * @return The argument out, now altered with the update status
 */
inline std::ostream& operator<<(std::ostream& out, const UpdateStatus update_status) {
    if (update_status == UpdateStatus::Disabled) {
        return out << "Disabled";
    }

    if (update_status == UpdateStatus::Enabled) {
        return out << "Enabled";
    }

    if (update_status == UpdateStatus::Static) {
        return out << "Static";
    }

    return out;
}

template <>
struct fmt::formatter<UpdateStatus> : ostream_formatter { };