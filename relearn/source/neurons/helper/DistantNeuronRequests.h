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
#include "neurons/helper/SynapseCreationRequests.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

/**
 * One DistantNeuronRequest always consists of a source neuron and its position, which initiates the request,
 * the signal type which it looks for, and an identifier for the target neuron.
 * This identifier changes based on the height of the target neuron in the octree.
 * There are 3 distinct cases (with descending priority):
 * (a) The target is a leaf node
 * (b) The target is on the branch node level
 * (c) The target is a virtual node
 */
class DistantNeuronRequest {
public:
    /**
     * A target neuron can be one of the following:
     * (a) The target is a leaf node
     * (b) The target is a virtual node
     */
    enum class TargetNeuronType : char {
        Leaf,
        VirtualNode
    };

    /**
     * @brief Constructs an object with default-constructed members.
     *      This constructor is present for resizing vectors, etc.
     */
    constexpr DistantNeuronRequest() = default;

    /**
     * @brief Constructs a new request with the arguments. A request can be built for three different use cases:
     *      (a) The target node is a branch node -- target_neuron_type should be TargetNeuronType::BranchNode and target_neuron_identifier the index of it when considering all branch nodes
     *      (b) The target node is a leaf node -- target_neuron_type should be TargetNeuronType::Leaf and target_neuron_identifier the index of it in the local neurons
     *      (c) The target node is a virtual node -- target_neuron_type should be TargetNeuronType::VirtualNode and target_neuron_identifier the RMA offset
     * @param _source_id The RankNeuronId of the source, must be an actual neuron id
     * @param _source_position The position of the source
     * @param _target_neuron_identifier The identifier of the target node
     * @param _target_neuron_type The type of the target node
     * @param _signal_type The signal type
     * @exception Throws a RelearnException if source_id is virtual or uninitialized
     */
    constexpr DistantNeuronRequest(const NeuronID _source_id, const RelearnTypes::position_type& _source_position,
                                   const NeuronID::value_type _target_neuron_identifier, const TargetNeuronType _target_neuron_type, const SignalType _signal_type)
        : source_id(_source_id)
        , source_position(_source_position)
        , target_neuron_identifier(_target_neuron_identifier)
        , target_neuron_type(_target_neuron_type)
        , signal_type(_signal_type) {
        RelearnException::check(source_id.is_actual_id(), "DistantNeuronRequest::DistantNeuronRequest: The source neuron must be initialized and non-virtual.");
    }

    /**
     * @brief Returns the source of the request
     * @return The source
     */
    [[nodiscard]] constexpr NeuronID get_source_id() const noexcept {
        return source_id;
    }

    /**
     * @brief Returns the position of the source of the request
     * @return The source position
     */
    [[nodiscard]] constexpr RelearnTypes::position_type get_source_position() const noexcept {
        return source_position;
    }

    /**
     * @brief Returns the id of the target node, if it is a leaf node.
     * @exception Throws a RelearnException if the target node type is not TargetNeuronType::Leaf
     * @return The leaf node id
     */
    [[nodiscard]] constexpr NeuronID::value_type get_leaf_node_id() const {
        RelearnException::check(target_neuron_type == TargetNeuronType::Leaf, "DistantNeuronRequest::get_leaf_node_id: The target_neuron_type was not Leaf.");
        return target_neuron_identifier;
    }

    /**
     * @brief Returns the RMA offset of the target node, if it is a virtual node.
     * @exception Throws a RelearnException if the target node type is not TargetNeuronType::VirtualNode
     * @return The RMA offset
     */
    [[nodiscard]] constexpr NeuronID::value_type get_rma_offset() const {
        RelearnException::check(target_neuron_type == TargetNeuronType::VirtualNode, "DistantNeuronRequest::get_leaf_node_id: The target_neuron_type was not VirtualNode.");
        return target_neuron_identifier;
    }

    /**
     * @brief Returns the type of target neuron
     * @return The type of the target neuron
     */
    [[nodiscard]] constexpr TargetNeuronType get_target_neuron_type() const noexcept {
        return target_neuron_type;
    }

    /**
     * @brief Returns the signal type of the request
     * @return The signal type
     */
    [[nodiscard]] constexpr SignalType get_signal_type() const noexcept {
        return signal_type;
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto& get() & {
        if constexpr (Index == 0) {
            return source_id;
        }
        if constexpr (Index == 1) {
            return source_position;
        }
        if constexpr (Index == 2) {
            return target_neuron_identifier;
        }
        if constexpr (Index == 3) {
            return target_neuron_type;
        }
        if constexpr (Index == 4) {
            return signal_type;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr const auto& get() const& {
        if constexpr (Index == 0) {
            return source_id;
        }
        if constexpr (Index == 1) {
            return source_position;
        }
        if constexpr (Index == 2) {
            return target_neuron_identifier;
        }
        if constexpr (Index == 3) {
            return target_neuron_type;
        }
        if constexpr (Index == 4) {
            return signal_type;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto&& get() && {
        if constexpr (Index == 0) {
            return source_id;
        }
        if constexpr (Index == 1) {
            return source_position;
        }
        if constexpr (Index == 2) {
            return target_neuron_identifier;
        }
        if constexpr (Index == 3) {
            return target_neuron_type;
        }
        if constexpr (Index == 4) {
            return signal_type;
        }
    }

private:
    NeuronID source_id;
    RelearnTypes::position_type source_position;
    NeuronID::value_type target_neuron_identifier{};
    TargetNeuronType target_neuron_type{};
    SignalType signal_type{};

    // static_assert(sizeof(target_neuron_identifier) >= sizeof(std::intptr_t), "DistantNeuronRequest: The size of target_neuron_identifier cannot hold a pointer");
};

namespace std {
template <>
struct tuple_size<DistantNeuronRequest> {
    static constexpr size_t value = 5;
};

template <>
struct tuple_element<0, DistantNeuronRequest> {
    using type = NeuronID;
};

template <>
struct tuple_element<1, DistantNeuronRequest> {
    using type = RelearnTypes::position_type;
};

template <>
struct tuple_element<2, DistantNeuronRequest> {
    using type = NeuronID::value_type;
};

template <>
struct tuple_element<3, DistantNeuronRequest> {
    using type = ::DistantNeuronRequest::TargetNeuronType;
};

template <>
struct tuple_element<4, DistantNeuronRequest> {
    using type = SignalType;
};

} // namespace std

/**
 * The response for a DistantNeuronRequest consists of the source of the response and a SynapseCreationResponse
 */
class DistantNeuronResponse {
public:
    /**
     * @brief Constructs an object with default-constructed members.
     *      This constructor is present for resizing vectors, etc.
     */
    constexpr DistantNeuronResponse() = default;

    /**
     * @brief Constructs a new response with the arguments
     * @param source_neuron_id The RankNeuronId of the source, must be an actual neuron id
     * @param creation_response The response if a synapse was successfully created
     * @exception Throws a RelearnException if source_neuron_id is virtual or not initialized
     */
    constexpr DistantNeuronResponse(const NeuronID source_neuron_id, const SynapseCreationResponse creation_response)
        : source_id(source_neuron_id)
        , response(creation_response) {
        RelearnException::check(source_id.is_actual_id(), "DistantNeuronRequest::DistantNeuronRequest: The source neuron must be initialized and non-virtual.");
    }

    /**
     * @brief Returns the source of the response
     * @return The source
     */
    [[nodiscard]] constexpr NeuronID get_source_id() const noexcept {
        return source_id;
    }

    /**
     * @brief Returns the creation response
     * @return The creation response
     */
    [[nodiscard]] constexpr SynapseCreationResponse get_creation_response() const noexcept {
        return response;
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto& get() & {
        if constexpr (Index == 0) {
            return source_id;
        }
        if constexpr (Index == 1) {
            return response;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr const auto& get() const& {
        if constexpr (Index == 0) {
            return source_id;
        }
        if constexpr (Index == 1) {
            return response;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto&& get() && {
        if constexpr (Index == 0) {
            return source_id;
        }
        if constexpr (Index == 1) {
            return response;
        }
    }

private:
    NeuronID source_id;
    SynapseCreationResponse response{};
};

namespace std {
template <>
struct tuple_size<typename ::DistantNeuronResponse> {
    static constexpr size_t value = 2;
};

template <>
struct tuple_element<0, typename ::DistantNeuronResponse> {
    using type = NeuronID;
};

template <>
struct tuple_element<1, typename ::DistantNeuronResponse> {
    using type = SynapseCreationResponse;
};

} // namespace std
