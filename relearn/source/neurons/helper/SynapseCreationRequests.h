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
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cstddef>
#include <utility>

/**
 * One SynapseCreationRequest always consists of a target neuron, a source neuron, and a signal type.
 * It is exchanged between MPI ranks but does not perform MPI communication on its own.
 * A synapse creation is requested as
 *      (source, element_type, signal_type) ---> (target, !element_type, signal_type)
 * with element_type being known from context (i.e., only dendrites send creation requests)
 */
class SynapseCreationRequest {
public:
    /**
     * @brief Constructs an object with default-constructed members.
     *      This constructor is present for resizing vectors, etc.
     */
    constexpr SynapseCreationRequest() = default;

    /**
     * @brief Constructs a new request with the arguments
     * @param _target The neuron target id of the request
     * @param _source The neuron source id of the request
     * @param _signal_type The signal type
     */
    constexpr SynapseCreationRequest(const NeuronID _target, const NeuronID _source, const SignalType _signal_type)
        : target(_target)
        , source(_source)
        , signal_type(_signal_type) {
        RelearnException::check(target.is_actual_id(), "SynapseCreationRequest::SynapseCreationRequest: Can only serve non-virtual ids (target): {}", target);
        RelearnException::check(source.is_actual_id(), "SynapseCreationRequest::SynapseCreationRequest: Can only serve non-virtual ids (source): {}", source);
    }

    /**
     * @brief Returns the target of the request
     * @return The target
     */
    [[nodiscard]] constexpr NeuronID get_target() const noexcept {
        return target;
    }

    /**
     * @brief Returns the source of the request
     * @return The source
     */
    [[nodiscard]] constexpr NeuronID get_source() const noexcept {
        return source;
    }

    /**
     * @brief Returns the neuron type of the request
     * @return The neuron type
     */
    [[nodiscard]] constexpr SignalType get_signal_type() const noexcept {
        return signal_type;
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto& get() & {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return signal_type;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr const auto& get() const& {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return signal_type;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto&& get() && {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return signal_type;
        }
    }

private:
    NeuronID target{};
    NeuronID source{};
    SignalType signal_type{};
};

namespace std {
template <>
struct tuple_size<typename ::SynapseCreationRequest> {
    static constexpr size_t value = 3;
};

template <>
struct tuple_element<0, typename ::SynapseCreationRequest> {
    using type = NeuronID;
};

template <>
struct tuple_element<1, typename ::SynapseCreationRequest> {
    using type = NeuronID;
};

template <>
struct tuple_element<2, typename ::SynapseCreationRequest> {
    using type = SignalType;
};

} // namespace std

/**
 * The response for a SynapseCreationRequest can be that it failed or succeeded
 */
enum class SynapseCreationResponse : char {
    Failed = 0,
    Succeeded = 1,
};
