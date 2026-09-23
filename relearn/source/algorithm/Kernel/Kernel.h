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

#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/Kernel/KernelBase.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/RankNeuronId.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"
#include "util/ProbabilityPicker.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <fmt/ostream.h>

#include <range/v3/algorithm/transform.hpp>
#include <range/v3/range/conversion.hpp>

#include <algorithm>
#include <numeric>
#include <ostream>
#include <utility>
#include <vector>

/**
 * This class encapsulates the necessary probability kernels that determine how
 * likely it is that two neurons form a synapse.
 * It handles different use cases, depending on where in the pipeline it is inserted.
 * @tparam AdditionalCellAttributes The AdditionalCellAttributes for the OctreeNode.
 */
template <typename AdditionalCellAttributes>
class Kernel {
public:
    using attraction_type = RelearnTypes::attraction_type;
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    /**
     * @brief Calculates the attractiveness to connect on the basis of the kernel.
     *      Performs all necessary checks and passes the values to the actual kernel.
     * @param kernel The probability kernel
     * @param source_neuron_id The source neuron id
     * @param source_position The source position s
     * @param target_node The target node
     * @param element_type The element type
     * @param signal_type The signal type
     * @exception Throws a RelearnException if the position for (element_type, signal_type) from target_node is empty or not supported
     * @return The calculated attractiveness, might be 0.0 to avoid autapses
     */
    [[nodiscard]] static attraction_type calculate_attractiveness_to_connect(const KernelBase& kernel, const RankNeuronId& source_neuron_id, const position_type& source_position,
                                                                             const OctreeNode<AdditionalCellAttributes>* target_node, const ElementType element_type, const SignalType signal_type) {
        // A neuron must not form an autapse, i.e., a synapse to itself
        if (target_node->contains(source_neuron_id)) {
            return 0.0;
        }

        const auto& cell = target_node->get_cell();
        const auto& target_position = cell.get_position_for(element_type, signal_type);
        const auto number_elements = cell.get_number_elements_for(element_type, signal_type);

        RelearnException::check(target_position.has_value(), "Kernel::calculate_attractiveness_to_connect: target_position is bad");

        return kernel.get_probability(source_position, target_position.value()) * static_cast<attraction_type>(number_elements);
    }

    /**
     * @brief Calculates the probability for the source neuron to connect to each of the OctreeNodes in the vector,
     *      searching the specified element_type and signal_type.
     *      If all probabilities are 0.0 (by rounding errors), the probabilities are calculated as number_free_elements/euclidean_distance
     * @param kernel The probability kernel
     * @param source_neuron_id The id of the source neuron, is used to prevent autapses
     * @param source_position The position of the source neuron
     * @param nodes All nodes from which the source neuron can pick
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @exception Throws a RelearnException if one of the pointer in nodes is a nullptr, or if the kernel throws
     * @return A pair of (a) the total probability of all targets and (b) the respective probability of each target
     */
    [[nodiscard]] static std::pair<attraction_type, std::vector<attraction_type>> create_probability_interval(const KernelBase& kernel, const RankNeuronId& source_neuron_id, const position_type& source_position,
                                                                                                              const std::vector<OctreeNode<AdditionalCellAttributes>*>& nodes, const ElementType element_type, const SignalType signal_type) {

        if (nodes.empty()) {
            return { attraction_type{ 0 }, {} };
        }

        auto sum = attraction_type{ 0 };

        auto probabilities = std::vector<attraction_type>{};
        probabilities.reserve(nodes.size());

        ranges::transform(nodes, std::back_inserter(probabilities), [&](const OctreeNode<AdditionalCellAttributes>* target_node) {
            RelearnException::check(target_node != nullptr, "Kernel::create_probability_interval: target_node was nullptr");
            const auto prob = calculate_attractiveness_to_connect(kernel, source_neuron_id, source_position, target_node, element_type, signal_type);
            sum += prob;
            return prob;
        });

        if (sum == attraction_type{ 0 }) {
            // If all targets are so far away that rounding errors return a probability of 0, we fix this

            probabilities.resize(0);
            ranges::transform(nodes, std::back_inserter(probabilities), [&](const OctreeNode<AdditionalCellAttributes>* target_node) {
                if (target_node->contains(source_neuron_id)) {
                    return attraction_type{ 0 };
                }

                const auto& cell = target_node->get_cell();
                const auto& target_position = cell.get_position_for(element_type, signal_type);
                const auto& number_elements = cell.get_number_elements_for(element_type, signal_type);

                const auto prob = static_cast<attraction_type>(number_elements) / ((target_position.value() - source_position).template calculate_2_norm<attraction_type>());
                sum += prob;
                return prob;
            });
        }

        if (sum == attraction_type{ 0 }) {
            // If the vector still contains only the same node, return nothing
            return { attraction_type{ 0 }, {} };
        }

        return { sum, std::move(probabilities) };
    }

    /**
     * @brief Picks a target based on the kernel
     * @param kernel The probability kernel
     * @param source_neuron_id The id of the source neuron, is used to prevent autapses
     * @param source_position The position of the source neuron
     * @param nodes The target nodes, must not be empty
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @exception Throws a RelearnException if one of the pointer in nodes is a nullptr, or if the kernel throws
     * @return The selected target node, is nullptr if nodes.empty()
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* pick_target(const KernelBase& kernel, const RankNeuronId& source_neuron_id, const position_type& source_position,
                                                                           const std::vector<OctreeNode<AdditionalCellAttributes>*>& nodes, const ElementType element_type, const SignalType signal_type) {
        if (nodes.empty()) {
            return nullptr;
        }

        /**
         * Assign a probability to each node in the vector.
         * The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
         */
        const auto& [total_probability, all_probabilities]
            = create_probability_interval(kernel, source_neuron_id, source_position, nodes, element_type, signal_type);

        // Short cut to avoid exceptions later on
        if (total_probability == attraction_type{ 0 }) {
            return nullptr;
        }

        RelearnException::check(nodes.size() == all_probabilities.size(), "Kernel::pick_target: Had a different number of probabilities than nodes: {} vs {}", nodes.size(), all_probabilities.size());

        const auto picked_idx = ProbabilityPicker::pick_target(all_probabilities, RandomHolderKey::Algorithm);
        auto* const node_selected = nodes[picked_idx];

        RelearnException::check(node_selected != nullptr, "Kernel::pick_target: node_selected was nullptr");

        return node_selected;
    }
};