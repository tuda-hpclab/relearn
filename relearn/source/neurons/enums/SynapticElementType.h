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

#include "util/RelearnException.h"

/**
 * An instance of this enum classifies the synaptic elements of a neuron.
 * In this simulation, there exists exactly two different ones: axonal elements and dendritic elements.
 * The distinction excitatory / inhibitory is made by the type SignalType.
 */
enum class ElementType : char {
    Axon,
    Dendrite
};

/**
 * @brief Returns the other element type, i.e., Axon for Dendrite and vice versa
 * @param element_type The current element type
 * @return The other element type
 */
[[nodiscard]] constexpr ElementType get_other_element_type(const ElementType element_type) noexcept {
    if (element_type == ElementType::Axon) {
        return ElementType::Dendrite;
    }

    return ElementType::Axon;
}

/**
 * An instance of this enum classifies a synaptic elements as either excitatory or inhibitory.
 * An axon of a specific type pairs only with dendrites of that type.
 * A spiked transfered via an excitatory axon to an excitatory dendrite increases the electrical activity,
 * while the same for an inhibitory decreases the electrical activity.
 */
enum class SignalType : char {
    Excitatory,
    Inhibitory
};

/**
 * @brief Returns the other signal type, i.e., Excitatory for Inhibitory and vice versa
 * @param signal_type The current signal type
 * @return The other signal type
 */
[[nodiscard]] constexpr SignalType get_other_signal_type(const SignalType signal_type) noexcept {
    if (signal_type == SignalType::Excitatory) {
        return SignalType::Inhibitory;
    }

    return SignalType::Excitatory;
}

/**
 * An instance of this enum classifies a synaptic element.
 * A neuron grows an axon, as well as an excitatory and an inhibitory dendrite.
 */
enum class SynapticElementType : char {
    Axon,
    DendriteExcitatory,
    DendriteInhibitory
};

/**
 * @brief Returns the element type of a synaptic element
 * @param synaptic_element_type The synaptic element type
 * @return The element type of the synaptic element
 */
[[nodiscard]] constexpr ElementType get_element_type(const SynapticElementType synaptic_element_type) noexcept {
    if (synaptic_element_type == SynapticElementType::Axon) {
        return ElementType::Axon;
    }
    return ElementType::Dendrite;
}

/**
 * @brief Returns the signal type of a synaptic element. Works only for dendritic elements.
 * @param synaptic_element_type The synaptic element type
 * @exception Throws a RelearnException if the synaptic element type is SynapticElementType::Axon
 * @return The signal type of the synaptic element
 */
[[nodiscard]] constexpr SignalType get_signal_type(const SynapticElementType synaptic_element_type) {
    if (synaptic_element_type == SynapticElementType::DendriteExcitatory) {
        return SignalType::Excitatory;
    }
    if (synaptic_element_type == SynapticElementType::DendriteInhibitory) {
        return SignalType::Inhibitory;
    }

    RelearnException::fail("Invalid synaptic element type");
}

/**
 * @brief Constructs a SynapticElementType from an ElementType and a SignalType.
 *      If the element type is Axon, the signal type is ignored.
 * @param element_type The element type of the synaptic element
 * @param signal_type The signal type of the synaptic element
 * @return The SynapticElementType constructed from the element type and the signal type
 */
[[nodiscard]] constexpr SynapticElementType get_synaptic_element_type(const ElementType element_type, const SignalType signal_type) noexcept {
    if (element_type == ElementType::Axon) {
        return SynapticElementType::Axon;
    }
    if (signal_type == SignalType::Excitatory) {
        return SynapticElementType::DendriteExcitatory;
    }
    return SynapticElementType::DendriteInhibitory;
}

/**
 * @brief Pretty-prints the element type to the chosen stream
 * @param out The stream to which to print the element type
 * @param element_type The element type to print
 * @return The argument out, now altered with the element type
 */
inline std::ostream& operator<<(std::ostream& out, const ElementType element_type) {
    if (element_type == ElementType::Axon) {
        return out << "Axon";
    }

    if (element_type == ElementType::Dendrite) {
        return out << "Dendrite";
    }

    return out;
}

template <>
struct fmt::formatter<ElementType> : ostream_formatter { };

/**
 * @brief Pretty-prints the signal type to the chosen stream
 * @param out The stream to which to print the signal type
 * @param signal_type The signal type to print
 * @return The argument out, now altered with the signal type
 */
inline std::ostream& operator<<(std::ostream& out, const SignalType signal_type) {
    if (signal_type == SignalType::Excitatory) {
        return out << "Excitatory";
    }

    if (signal_type == SignalType::Inhibitory) {
        return out << "Inhibitory";
    }

    return out;
}

template <>
struct fmt::formatter<SignalType> : ostream_formatter { };

/**
 * @brief Pretty-prints the synaptic element type to the chosen stream
 * @param out The stream to which to print the synaptic element type
 * @param synaptic_element_type The synaptic element type to print
 * @return The argument out, now altered with the synaptic element type
 */
inline std::ostream& operator<<(std::ostream& out, const SynapticElementType synaptic_element_type) {
    if (synaptic_element_type == SynapticElementType::DendriteExcitatory) {
        return out << "Excitatory Dendrite";
    }

    if (synaptic_element_type == SynapticElementType::DendriteInhibitory) {
        return out << "Inhibitory Dendrite";
    }

    if (synaptic_element_type == SynapticElementType::Axon) {
        return out << "Axon";
    }

    return out;
}

template <>
struct fmt::formatter<SynapticElementType> : ostream_formatter { };