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

#include "neurons/enums/SynapticElementType.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cstddef>
#include <utility>

/**
 * This type represents a synapse deletion request.
 * It is exchanged between MPI ranks but does not perform MPI communication on its own.
 * A synapse deletion is represented as
 *      (initiator_neuron_id, initiator_element_type, signal_type) ---> (affected_neuron_id, !initiator_element_type, signal_type)
 */
class SynapseDeletionRequest {
public:
    /**
     * Creates a new object with default-constructed members.
     *      This constructor is present for resizing vectors, etc.
     */
    constexpr SynapseDeletionRequest() = default;

    /**
     * @brief Creates a new deletion request with the passed arguments
     * @param initiator_neuron The neuron that initiated the request, must be an actual neuron id
     * @param affected_neuron The neuron that is affected by the request (the other end of the synapse), must be an actual neuron id
     * @param element_type The element type that initiated the request
     * @param _signal_type The signal type of the synapse
     * @exception Throws a RelearnException if any neuron id is invalid or virtual
     */
    constexpr SynapseDeletionRequest(const NeuronID initiator_neuron, const NeuronID affected_neuron,
                                     const ElementType element_type, const SignalType _signal_type)
        : initiator_neuron_id(initiator_neuron)
        , affected_neuron_id(affected_neuron)
        , initiator_element_type(element_type)
        , signal_type(_signal_type) {
        RelearnException::check(initiator_neuron.is_actual_id(), "SynapseDeletionRequest::SynapseDeletionRequest(): initiator_neuron is not local: {}", initiator_neuron);
        RelearnException::check(affected_neuron.is_actual_id(), "SynapseDeletionRequest::SynapseDeletionRequest(): affected_neuron is not local: {}", affected_neuron);
    }

    /**
     * @brief Returns the initiator neuron id, i.e., the neuron which started the deletion
     * @return The source neuron id
     */
    [[nodiscard]] constexpr NeuronID get_initiator_neuron_id() const noexcept {
        return initiator_neuron_id;
    }

    /**
     * @brief Returns the affected neuron id, i.e., the neuron that must be notified
     * @return The target neuron id
     */
    [[nodiscard]] constexpr NeuronID get_affected_neuron_id() const noexcept {
        return affected_neuron_id;
    }

    /**
     * @brief Returns the initiator element type
     * @return The affected element type
     */
    [[nodiscard]] constexpr ElementType get_initiator_element_type() const noexcept {
        return initiator_element_type;
    }

    /**
     * @brief Returns the synapses signal type
     * @return The signal type
     */
    [[nodiscard]] constexpr SignalType get_signal_type() const noexcept {
        return signal_type;
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto& get() & {
        if constexpr (Index == 0) {
            return initiator_neuron_id;
        }
        if constexpr (Index == 1) {
            return affected_neuron_id;
        }
        if constexpr (Index == 2) {
            return initiator_element_type;
        }
        if constexpr (Index == 3) {
            return signal_type;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr const auto& get() const& {
        if constexpr (Index == 0) {
            return initiator_neuron_id;
        }
        if constexpr (Index == 1) {
            return affected_neuron_id;
        }
        if constexpr (Index == 2) {
            return initiator_element_type;
        }
        if constexpr (Index == 3) {
            return signal_type;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto&& get() && {
        if constexpr (Index == 0) {
            return initiator_neuron_id;
        }
        if constexpr (Index == 1) {
            return affected_neuron_id;
        }
        if constexpr (Index == 2) {
            return initiator_element_type;
        }
        if constexpr (Index == 3) {
            return signal_type;
        }
    }

private:
    NeuronID initiator_neuron_id;
    NeuronID affected_neuron_id;
    ElementType initiator_element_type{ ElementType::Axon };
    SignalType signal_type{ SignalType::Excitatory };
};

namespace std {
template <>
struct tuple_size<typename ::SynapseDeletionRequest> {
    static constexpr size_t value = 4;
};

template <>
struct tuple_element<0, typename ::SynapseDeletionRequest> {
    using type = NeuronID;
};

template <>
struct tuple_element<1, typename ::SynapseDeletionRequest> {
    using type = NeuronID;
};

template <>
struct tuple_element<2, typename ::SynapseDeletionRequest> {
    using type = ElementType;
};

template <>
struct tuple_element<3, typename ::SynapseDeletionRequest> {
    using type = SignalType;
};

} // namespace std
