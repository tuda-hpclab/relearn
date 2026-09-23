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

#include "Config.h"

#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/Internal/octree/NodeCache.h"
#include "algorithm/Internal/octree/OctreeNode.h"
#include "algorithm/Internal/octree/OctreeNodeHelper.h"
#include "algorithm/Kernel/Gaussian.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "util/ProbabilityPicker.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"
#include "util/Vec3.h"

#include <cpp-utility/data-structure/Stack.hpp>
#include <cpp-utility/ranges/Functional.hpp>

#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/range/operations.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <ostream>
#include <span>
#include <vector>

/**
 * This enum classifies the different calculation types for the Fast Multipole Method.
 * The group-group interaction can be calculated directly, via a Hermite expansion, or
 * via a Taylor expansion.
 */
enum class CalculationType : std::uint8_t {
    Direct,
    Hermite,
    Taylor,
};

/**
 * @brief Pretty-prints the calculation type to the chosen stream
 * @param out The stream to which to print the calculation type
 * @param calc_type The calculation type to print
 * @return The argument out, now altered with the calculation type
 */
inline std::ostream& operator<<(std::ostream& out, const CalculationType& calc_type) {
    if (calc_type == CalculationType::Direct) {
        return out << "Direct";
    }

    if (calc_type == CalculationType::Hermite) {
        return out << "Hermite";
    }

    return out << "Taylor";
}

/**
 * This class represents a mathematical three-dimensional multi-index, which is required for the
 * series expansions and coefficient calculations.
 */
class MultiIndex {
public:
    /**
     * @brief Returns the number of all three-dimensional indices that the multi-index has. This depends on the selected p.
     * @return Returns the number of all indices.
     */
    [[nodiscard]] constexpr static unsigned int get_number_of_indices() noexcept {
        return Constants::p3;
    }

    /**
     * @brief Returns the multi-index as a matrix with the dimensions (p^3, 3).
     * @return Returns a array of arrays which represents the corresponding multi-index.
     */
    [[nodiscard]] constexpr static std::array<Vec3u, Constants::p3> get_indices() noexcept {
        auto result = std::array<Vec3u, Constants::p3>{};

        auto index = 0U;
        for (auto i = 0U; i < Constants::p; i++) {
            for (auto j = 0U; j < Constants::p; j++) {
                for (auto k = 0U; k < Constants::p; k++) {
                    // NOLINTNEXTLINE
                    result[index] = { i, j, k };
                    index++;
                }
            }
        }

        return result;
    }
};

/**
 * This class provides all computational elements of the Fast-Multipole-Method algorithm.
 * It purely calculates things, but does not change any state.
 */
class FastMultipoleMethodBase {
public:
    using AdditionalCellAttributes = FastMultipoleMethodCell;
    using interaction_list_type = std::vector<OctreeNode<AdditionalCellAttributes>*>;
    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;
    using attraction_type = RelearnTypes::attraction_type;
    using level_type = RelearnTypes::level_type;
    using node_pair = std::array<OctreeNode<AdditionalCellAttributes>*, 2>;

    struct stack_entry {
        OctreeNode<AdditionalCellAttributes>* source{};
        OctreeNode<AdditionalCellAttributes>* target{};
        bool unpacked = { false };
    };

    struct my_pair {
        OctreeNode<AdditionalCellAttributes>* current_source;
        OctreeNode<AdditionalCellAttributes>* current_target;
    };

    /**
     * @brief Calculates the n-th Hermite function at the point t
     * @param n Order of the Hermite function.
     * @param t Point of evaluation.
     * @return Value of the Hermite function of the n-th order at the point t.
     */
    [[nodiscard]] static attraction_type h(unsigned int n, attraction_type t) {
        const auto t_squared = t * t;

        RelearnException::check(n < 128, "FastMultipoleMethodBase::h: n must be smaller than 128 ({})", n);
        RelearnException::check(!std::isnan(t), "FastMultipoleMethodBase::h: t is NaN");

        const auto fac_1 = std::exp(-t_squared);

        const auto fac_2 = std::hermite(n, t);

        return fac_1 * fac_2;
    }

    /**
     * @brief Calculates the Hermite function for a multi index and a 3D vector.
     * @param multi_index A tuple of three natural numbers.
     * @param vector A 3D vector.
     * @return Value of the Hermite function.
     */
    [[nodiscard]] static attraction_type h_multi_index(const Vec3u& multi_index, const position_type& vector) {
        const auto h1 = h(multi_index.get_x(), vector.get_x());
        const auto h2 = h(multi_index.get_y(), vector.get_y());
        const auto h3 = h(multi_index.get_z(), vector.get_z());

        const auto h_total = h1 * h2 * h3;

        return h_total;
    }

    /**
     * @brief The Kernel from Butz&Ooyen "A Simple Rule for Dendritic Spine and Axonal Bouton Formation Can Account for Cortical Reorganization afterFocal Retinal Lesions"
     *       Calculates the attraction between two neurons, where a and b represent the position in three-dimensional space
     * @param a 3D position of the first neuron.
     * @param b 3D position of the second neuron.
     * @param sigma scaling parameter.
     * @return Returns the attraction between the two neurons.
     */
    [[nodiscard]] static attraction_type kernel(const position_type& a, const position_type& b, const attraction_type sigma) {
        const auto diff = a - b;
        const auto squared_norm = diff.calculate_squared_2_norm<attraction_type>();

        return std::exp(-squared_norm / (sigma * sigma));
    }

    /**
     * @brief Returns the OctreeNode at the given index, nullptr elements are not counted.
     * @param arr Interaction list containing OctreeNodes.
     * @param index Index of the desired node.
     * @return The specified element, can be nullptr if not enough non-nullptr elements are present
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* extract_element(const interaction_list_type& arr, const interaction_list_type::size_type index) noexcept {
        auto non_zero_counter = 0U;
        for (auto* const node : arr | ranges::views::filter(utility::not_nullptr)) {
            if (index == non_zero_counter) {
                return node;
            }
            ++non_zero_counter;
        }
        return nullptr;
    }

    /**
     * @brief Checks which calculation type is suitable for a given source and target node
     * @param source Node with vacant searching elements
     * @param target Node with vacant searched elements
     * @param searched_element_type The element type that is seached for (in the targets)
     * @param signal_type Specifies for which type of neurons the calculation is to be executed
     * @exception Throws a RelearnException if source or target are nullptr
     * @return The calculation type that is appropriate for the forces
     */
    [[nodiscard]] static CalculationType check_calculation_requirements(const OctreeNode<AdditionalCellAttributes>* source, const OctreeNode<AdditionalCellAttributes>* target,
                                                                        const ElementType searched_element_type, const SignalType signal_type) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodBase::check_calculation_requirements: source is nullptr");
        RelearnException::check(target != nullptr, "FastMultipoleMethodBase::check_calculation_requirements: target is nullptr");

        if (source->is_leaf() || target->is_leaf()) {
            return CalculationType::Direct;
        }

        const auto& source_cell = source->get_cell();
        const auto& target_cell = target->get_cell();

        const auto other_element_type = get_other_element_type(searched_element_type);
        if (target_cell.get_number_elements_for(searched_element_type, signal_type) <= Constants::max_neurons_in_target) {
            return CalculationType::Direct;
        }

        if (source_cell.get_number_elements_for(other_element_type, signal_type) > Constants::max_neurons_in_source) {
            return CalculationType::Hermite;
        }

        return CalculationType::Taylor;
    }

    /**
     * @brief Calculates the force of attraction between all neurons from the subtrees
     * @param sigma The scaling parameter for the Gaussian kernel
     * @param source The subtree that has all the sources in it
     * @param target The subtree that has all the targets in it
     * @param searched_element_type The element type that is seached for (in the targets)
     * @param signal_type Specifies for which type of neurons the calculation is to be executed
     * @exception Throws a RelearnException if source or target are nullptr
     * @return Returns the total attraction of the neurons.
     */
    [[nodiscard]] static attraction_type calc_direct_gauss(const attraction_type sigma, const NodeCache<AdditionalCellAttributes>& cache, OctreeNode<AdditionalCellAttributes>* source, OctreeNode<AdditionalCellAttributes>* target,
                                                           const ElementType searched_element_type, const SignalType signal_type) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodBase::calc_direct_gauss: source is nullptr");
        RelearnException::check(target != nullptr, "FastMultipoleMethodBase::calc_direct_gauss: target is nullptr");

        const auto other_element_type = get_other_element_type(searched_element_type);

        const auto& sources = OctreeNodeExtractor<AdditionalCellAttributes>::get_all_positions_for(source, cache, other_element_type, signal_type);
        const auto& targets = OctreeNodeExtractor<AdditionalCellAttributes>::get_all_positions_for(target, cache, searched_element_type, signal_type);

        return ranges::accumulate(
            ranges::views::cartesian_product(sources, targets)
                | ranges::views::transform([sigma](const auto& source_target_pair) {
                      const auto& [source_pair, target_pair] = source_target_pair;
                      const auto& [source_position, number_sources] = source_pair;
                      const auto& [target_position, number_targets] = target_pair;
                      return kernel(target_position, source_position, sigma) * static_cast<attraction_type>(number_sources) * static_cast<attraction_type>(number_targets);
                  }),
            attraction_type{ 0 });
    }

    /**
     * @brief Calculates the hermite coefficients for a source node. The calculation of coefficients and series
     *      expansion is executed separately, because the coefficients can be reused.
     * @param sigma The scaling parameter for the Gaussian kernel
     * @param source Node with vacant elements
     * @param searching_element_type The type of synaptic elements that searches for partners (in the sources)
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed
     * @exception Throws a RelearnException if source is nullptr, has no children, has no valid position for the specified combination of element type and signal type, or
     *      the children have no valid position for it
     * @returns Returns the hermite coefficients.
     */
    [[nodiscard]] static std::vector<attraction_type> calc_hermite_coefficients(const attraction_type sigma, const OctreeNode<AdditionalCellAttributes>* source, const ElementType searching_element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodBase::calc_hermite_coefficients: source is nullptr");
        RelearnException::check(source->is_parent(), "FastMultipoleMethodBase::calc_hermite_coefficients: source node was a leaf node");

        Timers::start(TimerRegion::CALC_HERMITE_COEFFICIENTS);

        const auto& indices = MultiIndex::get_indices();

        auto hermite_coefficients = std::vector<attraction_type>{};
        hermite_coefficients.resize(Constants::p3);

        const auto& source_cell = source->get_cell();
        const auto& source_position_opt = source_cell.get_position_for(searching_element_type, signal_type_needed);
        RelearnException::check(source_position_opt.has_value(), "FastMultipoleMethodBase::calc_hermite_coefficients: source has no valid position.");

        const auto& source_position = source_position_opt.value();

        for (auto index = 0U; index < Constants::p3; index++) {
            auto child_attraction = attraction_type{ 0.0 };

            const auto& children = source->get_children();
            for (auto* child : children) {
                if (child == nullptr) {
                    continue;
                }

                const auto& cell = child->get_cell();
                const auto child_number_axons = cell.get_number_elements_for(searching_element_type, signal_type_needed);
                if (child_number_axons == 0) {
                    continue;
                }

                const auto& child_pos = cell.get_position_for(searching_element_type, signal_type_needed);
                RelearnException::check(child_pos.has_value(), "FastMultipoleMethodBase::calc_hermite_coefficients: source child has no valid position.");

                const auto& temp_vec = (child_pos.value() - source_position) / sigma;
                child_attraction += static_cast<attraction_type>(child_number_axons) * temp_vec.get_componentwise_power<attraction_type>(indices[index]);
            }

            const auto hermite_coefficient = child_attraction / static_cast<attraction_type>(indices[index].get_componentwise_factorial());
            hermite_coefficients[index] = hermite_coefficient;
        }

        Timers::stop_and_add(TimerRegion::CALC_HERMITE_COEFFICIENTS);

        return hermite_coefficients;
    }

    /**
     * @brief Calculates the taylor coefficients for a pair of nodes. The calculation of coefficients and series
     *      expansion is executed separately.
     * @param sigma The scaling parameter for the Gaussian kernel
     * @param source Node with vacant elements
     * @param target_center Position of the target node
     * @param searching_element_type The type of synaptic elements that searches for partners (in the sources)
     * @param signal_type Specifies for which type of neurons the calculation is to be executed
     * @exception Throws a RelearnException if source is nullptr, has no children, has no valid position for the specified combination of element type
     *      and signal type (but a number of vacant elements), or the children have no valid position for it
     * @return Returns the taylor coefficients.
     */
    [[nodiscard]] static std::vector<attraction_type> calc_taylor_coefficients(const attraction_type sigma, const OctreeNode<AdditionalCellAttributes>* source, const position_type& target_center,
                                                                               const ElementType searching_element_type, const SignalType signal_type) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodBase::calc_taylor_coefficients: source is nullptr");
        RelearnException::check(source->is_parent(), "FastMultipoleMethodBase::calc_taylor_coefficients: source node was a leaf node");

        Timers::start(TimerRegion::CALC_TAYLOR_COEFFICIENTS);

        const auto& indices = MultiIndex::get_indices();

        auto taylor_coefficients = std::vector<attraction_type>{};
        taylor_coefficients.resize(Constants::p3);

        const auto& children = source->get_children();

        for (auto index = 0U; index < Constants::p3; index++) {
            // NOLINTNEXTLINE
            const auto& current_index = indices[index];

            auto child_attraction = attraction_type{ 0.0 };
            for (const auto* source_child : children) {
                if (source_child == nullptr) {
                    continue;
                }

                const auto& cell = source_child->get_cell();
                const auto number_elements = cell.get_number_elements_for(searching_element_type, signal_type);
                if (number_elements == 0) {
                    continue;
                }

                const auto& child_pos = cell.get_position_for(searching_element_type, signal_type);
                RelearnException::check(child_pos.has_value(), "FastMultipoleMethodBase::calc_taylor_coefficients: source child has no position.");

                const auto& temp_vec = (child_pos.value() - target_center) / sigma;
                child_attraction += static_cast<attraction_type>(number_elements) * h_multi_index(current_index, temp_vec);
            }

            const auto coefficient = child_attraction / static_cast<attraction_type>(current_index.get_componentwise_factorial());
            const auto absolute_multi_index = current_index.calculate_1_norm();

            if (absolute_multi_index % 2 == 0) {
                // NOLINTNEXTLINE
                taylor_coefficients[index] = coefficient;
            } else {
                // NOLINTNEXTLINE
                taylor_coefficients[index] = -coefficient;
            }
        }

        Timers::stop_and_add(TimerRegion::CALC_TAYLOR_COEFFICIENTS);

        return taylor_coefficients;
    }

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Hermite series expansion.
     * @param sigma The scaling parameter for the Gaussian kernel
     * @param source Node with vacant searching elements.
     * @param target Node with vacant searched elements.
     * @param coefficients_buffer Memory location where the coefficients are stored.
     * @param searching_element_type The type of synaptic elements that searches for partners (in the sources)
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed
     * @exception Throws a RelearnException if source or target is nullptr, the buffer does not have the size Constants::p3,
     *      target is a leaf node, or source does not have a position of the requested tuple
     * @return Returns the attraction force.
     */
    [[nodiscard]] static attraction_type calc_hermite(const attraction_type sigma, const NodeCache<AdditionalCellAttributes>& cache, const OctreeNode<AdditionalCellAttributes>* source, OctreeNode<AdditionalCellAttributes>* target,
                                                      std::span<const attraction_type> coefficients_buffer, const ElementType searching_element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodBase::calc_hermite::calc_direct_gauss: source is nullptr");
        RelearnException::check(target != nullptr, "FastMultipoleMethodBase::calc_hermite::calc_direct_gauss: target is nullptr");
        RelearnException::check(coefficients_buffer.size() == Constants::p3,
                                "FastMultipoleMethodBase::calc_hermite::calc_direct_gauss: The coefficients must have size {} but have size {}", Constants::p3, coefficients_buffer.size());

        RelearnException::check(target->is_parent(), "FastMultipoleMethodBase::calc_hermite: target node was a leaf node");

        const auto other_element_type = get_other_element_type(searching_element_type);

        const auto& opt_source_center = source->get_cell().get_position_for(searching_element_type, signal_type_needed);
        RelearnException::check(opt_source_center.has_value(), "FastMultipoleMethodBase::calc_hermite: source node has no axon position.");

        const auto& source_center = opt_source_center.value();

        constexpr const auto indices = MultiIndex::get_indices();
        constexpr const auto number_coefficients = MultiIndex::get_number_of_indices();

        auto total_attraction = attraction_type{ 0.0 };

        const auto& interaction_list = cache.get_children(target);
        for (const auto* child_target : interaction_list) {
            if (child_target == nullptr) {
                continue;
            }

            const auto& cell = child_target->get_cell();
            const auto number_searched_elements = cell.get_number_elements_for(other_element_type, signal_type_needed);
            if (number_searched_elements == 0) {
                continue;
            }

            const auto& child_pos = cell.get_position_for(other_element_type, signal_type_needed);
            RelearnException::check(child_pos.has_value(), "FastMultipoleMethodBase::calc_hermite: target child node has no axon position.");

            const auto& temp_vec = (child_pos.value() - source_center) / sigma;

            auto child_attraction = attraction_type{ 0.0 };
            for (auto a = 0U; a < number_coefficients; a++) {
                // NOLINTNEXTLINE
                child_attraction += coefficients_buffer[a] * h_multi_index(indices[a], temp_vec);
            }

            // A child cannot repel, this might need a different fix
            child_attraction = std::max(child_attraction, attraction_type{ 0 });

            total_attraction += static_cast<attraction_type>(number_searched_elements) * child_attraction;
        }

        return total_attraction;
    }

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Taylor series expansion.
     * @param sigma The scaling parameter for the Gaussian kernel
     * @param source Node with vacant searching elements.
     * @param target Node with vacant searched elements.
     * @param searching_element_type The type of synaptic elements that searches for partners (in the sources)
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Throws a RelearnException if source or target ar nullptr, source is a leaf node,
     *      or if target doesn't have the necessary positions
     * @return Returns the attraction force.
     */
    [[nodiscard]] static attraction_type calc_taylor(const attraction_type sigma, const NodeCache<AdditionalCellAttributes>& cache, const OctreeNode<AdditionalCellAttributes>* source, OctreeNode<AdditionalCellAttributes>* target,
                                                     const ElementType searching_element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodBase::calc_taylor: source is nullptr");
        RelearnException::check(target != nullptr, "FastMultipoleMethodBase::calc_taylor: target is nullptr");

        RelearnException::check(source->is_parent(), "FastMultipoleMethodBase::calc_taylor: source is a leaf node");

        const auto other_element_type = get_other_element_type(searching_element_type);

        const auto& opt_target_center = target->get_cell().get_position_for(other_element_type, signal_type_needed);
        RelearnException::check(opt_target_center.has_value(), "FastMultipoleMethodBase::calc_taylor: target node has no position.");

        const auto& target_center = opt_target_center.value();
        const auto& taylor_coefficients = calc_taylor_coefficients(sigma, source, target_center, searching_element_type, signal_type_needed);

        const auto& indices = MultiIndex::get_indices();
        const auto& target_children = cache.get_children(target);

        auto total_attraction = attraction_type{ 0.0 };
        for (const auto* target_child : target_children) {
            if (target_child == nullptr) {
                continue;
            }

            const auto& cell = target_child->get_cell();
            const auto number_searched_elements = cell.get_number_elements_for(other_element_type, signal_type_needed);
            if (number_searched_elements == 0) {
                continue;
            }

            const auto& child_pos = cell.get_position_for(other_element_type, signal_type_needed);
            RelearnException::check(child_pos.has_value(), "FastMultipoleMethodBase::calc_taylor: target child has no position.");

            const auto& temp_vec = (child_pos.value() - target_center) / sigma;

            auto child_attraction = attraction_type{ 0.0 };
            for (auto b = 0U; b < Constants::p3; b++) {
                // NOLINTNEXTLINE
                child_attraction += taylor_coefficients[b] * temp_vec.get_componentwise_power<attraction_type>(indices[b]);
            }

            // A child cannot repel, this might need a different fix
            child_attraction = std::max(child_attraction, attraction_type{ 0 });

            total_attraction += static_cast<attraction_type>(number_searched_elements) * child_attraction;
        }

        return total_attraction;
    }

    /**
     * @brief Calculates the attraction between a single source neuron and all target neurons in the interaction list.
     * @param sigma The scaling parameter for the Gaussian kernel
     * @param source Node with vacant searching elements
     * @param interaction_list List of Nodes with vacant searched elements
     * @param searching_element_type The type of synaptic elements that searches for partners (in the sources)
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException
     * @return Returns a vector with the calculated forces of attraction. This contains as many elements as the interaction list.
     */
    [[nodiscard]] static std::vector<attraction_type> calc_attractiveness_to_connect(const attraction_type sigma, const NodeCache<AdditionalCellAttributes>& cache, OctreeNode<AdditionalCellAttributes>* const source, const interaction_list_type& interaction_list,
                                                                                     const ElementType searching_element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodBase::calc_attractiveness_to_connect: Source was a nullptr.");

        const auto searched_element_type = get_other_element_type(searching_element_type);

        auto result = std::vector<attraction_type>{};
        result.reserve(interaction_list.size());

        auto hermite_coefficients = std::vector<attraction_type>{};
        auto hermite_coefficients_init = false;

        // For every target calculate the attractiveness
        for (auto* current_target : interaction_list | ranges::views::filter(utility::not_nullptr)) {
            const auto calculation_type = check_calculation_requirements(source, current_target, searching_element_type, signal_type_needed);

            if (calculation_type == CalculationType::Direct) {
                const auto direct_attraction = calc_direct_gauss(sigma, cache, source, current_target, searching_element_type, signal_type_needed);
                result.emplace_back(direct_attraction);
                continue;
            }

            if (calculation_type == CalculationType::Taylor) {
                const auto taylor_attraction = calc_taylor(sigma, cache, source, current_target, searching_element_type, signal_type_needed);
                result.emplace_back(taylor_attraction);
                continue;
            }

            if (!hermite_coefficients_init) {
                // When the Calculation Type is Hermite, initialize the coefficients once.
                hermite_coefficients = calc_hermite_coefficients(sigma, source, searching_element_type, signal_type_needed);
                hermite_coefficients_init = true;
            }

            const auto hermite_attraction = calc_hermite(sigma, cache, source, current_target, hermite_coefficients, searching_element_type, signal_type_needed);
            result.emplace_back(hermite_attraction);
        }

        const auto total_attraction = std::reduce(result.begin(), result.end(), attraction_type{ 0.0 }, std::plus<attraction_type>{});

        if (total_attraction == attraction_type{ 0 }) {
            // We need a fix here
            for (auto i = std::size_t{ 0 }; i < result.size(); i++) {
                const auto& source_cell = source->get_cell();
                const auto& target_cell = extract_element(interaction_list, i)->get_cell();

                const auto& source_opt_position = source_cell.get_position_for(searching_element_type, signal_type_needed);
                const auto& target_opt_position = target_cell.get_position_for(searched_element_type, signal_type_needed);

                RelearnException::check(source_opt_position.has_value(), "FastMultipoleMethodBase::calc_attractiveness_to_connect: source_opt_position has no value");
                RelearnException::check(target_opt_position.has_value(), "FastMultipoleMethodBase::calc_attractiveness_to_connect: target_opt_position has no value");
                const auto& source_position = source_opt_position.value();
                const auto& target_position = target_opt_position.value();

                const auto& distance = (source_position - target_position).calculate_2_norm<attraction_type>();
                result[i] = static_cast<attraction_type>(source_cell.get_number_elements_for(searching_element_type, signal_type_needed) * target_cell.get_number_elements_for(searched_element_type, signal_type_needed)) / distance;
            }
        }

        return result;
    }

    /**
     * @brief Returns all nodes that are the requested level beneath the given node. Also returns leaf nodes that are closer to the node.
     *      Discards all nodes that have no vacant elements of the requested type
     * @param node The node from where to go down, not nullptr
     * @param levels_to_unpack The number of levels
     * @param element_type_needed The requested element type
     * @param signal_type_needed The requested signal type
     * @exception Throws a RelearnException if node is nullptr
     * @return A vector of all found nodes
     */
    [[nodiscard]] static std::vector<OctreeNode<AdditionalCellAttributes>*> unpack_levels(OctreeNode<AdditionalCellAttributes>* const node, const NodeCache<AdditionalCellAttributes>& cache, const level_type levels_to_unpack,
                                                                                          const ElementType element_type_needed, const SignalType signal_type_needed) {
        RelearnException::check(node != nullptr, "FastMultipoleMethodBase::unpack_levels: node is nullptr");

        struct stack_entry_type {
            OctreeNode<AdditionalCellAttributes>* ptr = nullptr;
            level_type unpacked = 0;
        };

        auto unpacked_nodes = std::vector<OctreeNode<AdditionalCellAttributes>*>{};
        unpacked_nodes.reserve((Constants::number_oct * levels_to_unpack) + 1);

        auto stack = utility::Stack<stack_entry_type>{ (Constants::number_oct * levels_to_unpack) + 1 };
        stack.emplace_back(node, level_type{ 0 });

        while (!stack.empty()) {
            const auto [current_node, level_diff] = stack.pop_back();
            RelearnException::check(current_node->get_mpi_rank().is_initialized(), "FastMultipoleMethodBase::unpack_levels: current_node has no MPI rank");

            const auto number_available_elements = current_node->get_cell().get_number_elements_for(element_type_needed, signal_type_needed);
            if (number_available_elements == 0) {
                continue;
            }

            if (level_diff == levels_to_unpack) {
                unpacked_nodes.emplace_back(current_node);
                continue;
            }

            if (current_node->is_leaf()) {
                unpacked_nodes.emplace_back(current_node);
                continue;
            }

            const auto& children = cache.get_children(current_node);
            for (auto* child : children) {
                if (child == nullptr) {
                    continue;
                }

                RelearnException::check(child->get_mpi_rank().is_initialized(), "FastMultipoleMethodBase::unpack_levels: child has no MPI rank");
                stack.emplace_back(child, static_cast<level_type>(level_diff + 1));
            }
        }

        return unpacked_nodes;
    }

    /**
     * @brief Finds a target for a local root. This is necessary for distributed trees, as in that case, going level-by-level down the tree has issues for non-local sources
     * @param sigma The scaling parameter for the Gaussian kernel
     * @param root The root of the octree, not nullptr
     * @param local_root A local root on the branching level
     * @param branch_level The branching level
     * @param searching_element_type The type of synaptic elements that searches for partners (in the sources)
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @exception Throws a RelearnException if:
     *      (a) root is nullptr
     *      (b) local_root is nullptr
     *      (c) root has a level != 0
     *      (d) local_root has a level != branch_level
     *      (e) branch_level is 0 but root and local_root different
     *      (f) something unexcepted happens within the algorithms
     * @return A pair of (local_root, some_target), where some_target is on the branch_level of the octree. Can be empty if there was no suitable target
     */
    [[nodiscard]] static std::optional<my_pair> find_target_for_local_root(const attraction_type sigma, const NodeCache<AdditionalCellAttributes>& cache, OctreeNode<AdditionalCellAttributes>* root, OctreeNode<AdditionalCellAttributes>* local_root,
                                                                           const level_type branch_level, const ElementType searching_element_type, const SignalType signal_type_needed) {
        RelearnException::check(root != nullptr, "FastMultipoleMethodBase::find_target_for_local_root: root was nullptr");
        RelearnException::check(local_root != nullptr, "FastMultipoleMethodBase::find_target_for_local_root: local_root was nullptr");

        RelearnException::check(root->get_level() == 0, "FastMultipoleMethodBase::find_target_for_local_root: root didn't have level 0");
        RelearnException::check(local_root->get_level() == branch_level, "FastMultipoleMethodBase::find_target_for_local_root: local_root didn't have the branch_level");

        RelearnException::check(local_root->get_mpi_rank().is_initialized(), "FastMultipoleMethodBase::find_target_for_local_root: local_root has no MPI rank");
        RelearnException::check(local_root->get_mpi_rank() == mpiPP::MPIInfo::get_my_rank(), "FastMultipoleMethodBase::find_target_for_local_root: local_root is not local");

        const auto searched_element_type = get_other_element_type(searching_element_type);

        const auto number_searching_elements = local_root->get_cell().get_number_elements_for(searching_element_type, signal_type_needed);
        const auto number_searched_elements = root->get_cell().get_number_elements_for(searched_element_type, signal_type_needed);

        if (number_searching_elements == 0 || number_searched_elements == 0) {
            return {};
        }

        if (branch_level == 0) {
            RelearnException::check(root == local_root, "FastMultipoleMethodBase::find_target_for_local_root: branch_level is 0 but root and local_root are different");
            RelearnException::check(root->get_mpi_rank().is_initialized(), "FastMultipoleMethodBase::find_target_for_local_root: The root has no MPI rank");
            return my_pair{ .current_source = root, .current_target = root };
        }

        const auto current_interaction_list = unpack_levels(root, cache, branch_level, searched_element_type, signal_type_needed);
        if (current_interaction_list.empty()) {
            // This should not happen
            return {};
        }

        const auto attractivenesses = calc_attractiveness_to_connect(sigma, cache, local_root, current_interaction_list, searching_element_type, signal_type_needed);
        const auto index = ProbabilityPicker::pick_target(attractivenesses, RandomHolderKey::Algorithm);

        auto* target = current_interaction_list[index];
        RelearnException::check(target->get_level() == branch_level, "FastMultipoleMethodBase::find_target_for_local_root: The picked target is not on the branch level");
        RelearnException::check(target->get_mpi_rank().is_initialized(), "FastMultipoleMethodBase::find_target_for_local_root: The picked target has no MPI rank");

        return my_pair{ .current_source = local_root, .current_target = target };
    }

    /**
     * @brief Uses the current_pair as a starting point, unpacks both trees the requested number of levels, and finds for each new source a new target.
     * @param sigma The scaling parameter for the Gaussian kernel
     *      Alternatively, uses leaf nodes if they are higher than the requested number of levels.
     * @param current_pair The current source and target, both not nullptr
     * @param searching_element_type The type of synaptic elements that searches for partners (in the sources)
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @param levels_to_unpack The number of levels to unpack, must be > 0
     * @exception Throws a RelearnExpection if one of current_pair is nullptr or if levels_to_unpack is == 0
     * @return New pairs at the lower level to evaluate again
     */
    [[nodiscard]] static std::vector<my_pair> find_partners(const attraction_type sigma, const NodeCache<AdditionalCellAttributes>& cache, const my_pair current_pair, const ElementType searching_element_type, const SignalType signal_type_needed, const level_type levels_to_unpack) {
        const auto& [current_source, current_target] = current_pair;

        RelearnException::check(current_source != nullptr, "FastMultipoleMethodBase::find_partners: The source is nullptr");
        RelearnException::check(current_target != nullptr, "FastMultipoleMethodBase::find_partners: The target is nullptr");
        RelearnException::check(levels_to_unpack > 0, "FastMultipoleMethodBase::find_partners: Must unpack at least one level");

        const auto searched_element_type = get_other_element_type(searching_element_type);

        const auto number_searching_elements = current_source->get_cell().get_number_elements_for(searching_element_type, signal_type_needed);
        const auto number_searched_elements = current_target->get_cell().get_number_elements_for(searched_element_type, signal_type_needed);

        if (number_searching_elements == 0 || number_searched_elements == 0) {
            return {};
        }

        const auto unpacked_sources = unpack_levels(current_source, cache, levels_to_unpack, searching_element_type, signal_type_needed);
        const auto unpacked_targets = unpack_levels(current_target, cache, levels_to_unpack, searched_element_type, signal_type_needed);

        auto new_pairs = std::vector<my_pair>{};
        new_pairs.reserve(unpacked_sources.size());

        for (auto* source : unpacked_sources) {
            const auto attractivenesses = calc_attractiveness_to_connect(sigma, cache, source, unpacked_targets, searching_element_type, signal_type_needed);
            const auto index = ProbabilityPicker::pick_target(attractivenesses, RandomHolderKey::Algorithm);

            new_pairs.emplace_back(source, unpacked_targets[index]);

            // What if number vacant elements in source id larger than number vacant elements in target?
        }

        return new_pairs;
    }

    /**
     * @brief Appends synapse creation requests for all local neurons that search a partner
     * @param sigma The scaling parameter for the Gaussian kernel
     * @param root The root of the octree, not nullptr
     * @param local_roots The local branch nodes, each not nullptr and get_level() == branch_level
     * @param branch_level The branch level
     * @param searching_element_type The type of synaptic elements that searches for partners (in the sources)
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @param levels_to_unpack The levels to unpack, must be > 0
     * @param request Where to append the requests
     * @exception Throws a RelearnException if root == nullptr, any local_root is nullptr, levels_to_unpack == 0, or anything goes wrong internally
     */
    static void make_creation_request_for(const attraction_type sigma, const NodeCache<AdditionalCellAttributes>& cache, OctreeNode<AdditionalCellAttributes>* root, const std::vector<OctreeNode<AdditionalCellAttributes>*>& local_roots, const level_type branch_level,
                                          const ElementType searching_element_type, const SignalType signal_type_needed, const level_type levels_to_unpack, RelearnTypes::comm_map_creation<SynapseCreationRequest>& request) {
        RelearnException::check(root != nullptr, "FastMultipoleMethodBase::make_creation_request_for: root is nullptr");
        RelearnException::check(levels_to_unpack > 0, "FastMultipoleMethodBase::make_creation_request_for: lvels_to_unpack was 0");

        auto stack = utility::Stack<my_pair>{ 200 };

        for (auto* local_root : local_roots) {
            RelearnException::check(local_root != nullptr, "FastMultipoleMethodBase::make_creation_request_for: A local root was nullptr");
            const auto& opt_pair = find_target_for_local_root(sigma, cache, root, local_root, branch_level, searching_element_type, signal_type_needed);

            if (opt_pair.has_value()) {
                stack.emplace_back(opt_pair.value());
            }
        }

        while (!stack.empty()) {
            const auto pair = stack.pop_back();
            const auto& [current_source, current_target] = pair;

            if (current_source->is_leaf() && current_target->is_leaf()) {
                request.append(current_target->get_mpi_rank(), SynapseCreationRequest{ current_target->get_cell_neuron_id(), current_source->get_cell_neuron_id(), signal_type_needed });
                continue;
            }

            const auto& new_pairs = find_partners(sigma, cache, pair, searching_element_type, signal_type_needed, levels_to_unpack);
            for (const auto& new_pair : new_pairs) {
                stack.emplace_back(new_pair);
            }
        }
    }

    static void print_calculation(const attraction_type sigma, const NodeCache<AdditionalCellAttributes>& cache, std::ostream& out_stream, OctreeNode<FastMultipoleMethodCell>* source, OctreeNode<FastMultipoleMethodCell>* target,
                                  const ElementType element_type, const SignalType needed) {

        const auto calc_type = check_calculation_requirements(source, target, element_type, needed);

        const auto direct = calc_direct_gauss(sigma, cache, source, target, element_type, needed);
        const auto taylor = calc_taylor(sigma, cache, source, target, element_type, needed);
        const auto& coefficients = calc_hermite_coefficients(sigma, source, element_type, needed);
        const auto hermite = calc_hermite(sigma, cache, source, target, coefficients, element_type, needed);

        out_stream << std::fixed;
        out_stream << direct << ",\t";
        out_stream << taylor << ",\t";
        out_stream << hermite << ",\t" << calc_type << '\n';
    }
};
