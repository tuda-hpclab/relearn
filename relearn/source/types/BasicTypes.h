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

#include "Macros.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

// The types of this project that are spelled with the standard library alone. Of this project it includes
// nothing but Macros.h, which includes nothing itself, and that is what keeps it includable from
// everywhere; every other header in types/ that needs one of the aliases below builds on it.

namespace RelearnTypes {

/**
 * @brief The floating point type in which the simulation calculates, i.e., the precision of every alias
 *      below that is neither a count nor an id. PRECISION in CMake picks it, cmake/Precision.cmake hands
 *      the choice to the compilers, and Macros.h turns it into RELEARN_PRECISION_FP32; a translation unit
 *      that is compiled without that definition calculates in double, which is the default of the build as
 *      well. CudaTypes::cuda_real is the counterpart of this alias on the device.
 */
#if RELEARN_PRECISION_FP32
using real = float;
#else
using real = double;
#endif

/** @brief The weight of a plastic synapse, i.e., the number of connections it bundles */
using plastic_synapse_weight = int;

/** @brief The weight of a static synapse, which the plasticity never changes */
using static_synapse_weight = real;

/** @brief A number of synapses, e.g., how many of them a step of the plasticity deleted */
using number_synapse_type = std::size_t;

/** @brief The value a neuron is identified by on its own rank, i.e., its local index */
using neuron_id = std::size_t;

/** @brief Counts the objects of the simulation, e.g., the vacant synaptic elements of a neuron */
using counter_type = unsigned int;

/** @brief The electrical activity of a neuron, i.e., its membrane potential, the input it receives, and the recovery variables of the neuron models */
using activity_type = real;

/** @brief The intra-cellular calcium concentration of a neuron */
using calcium_type = real;

/** @brief The number of synaptic elements a neuron has grown, which is continuous in contrast to the counter_type of the bound ones */
using grown_type = real;

/** @brief The attractiveness with which a (virtual) neuron draws a synapse to itself, >= 0.0 and not normalized to [0, 1] */
using attraction_type = real;

/** @brief The rate at which a neuron fires, i.e., the fraction of the steps in which it spiked */
using fire_rate_type = real;

/** @brief The criterion by which an approximating algorithm accepts a virtual neuron instead of descending, e.g., Barnes-Hut's theta */
using acceptance_criterion_type = real;

/** @brief A share of a whole in [0.0, 1.0], e.g., the fraction of the neurons that are excitatory */
using percentage_type = real;

/** @brief A step of the simulation */
using step_type = std::uint32_t;

/** @brief A number of neurons, which is counted more widely than the steps of the simulation are */
using number_neurons_type = std::uint64_t;

/**
 * @brief A level in the octree, i.e., the depth at which a node sits, and likewise a number of levels, e.g.,
 *      how far down the tree an algorithm descends. The space filling curves cap the refinement at 21 levels,
 *      so a single byte holds every level the simulation can reach.
 */
using level_type = std::uint8_t;

/** @brief The name of a group of neurons */
using group_name = std::string;

/** @brief The id of a group of neurons */
using group_id = std::size_t;

/** @brief Multiple group names in the order they were read */
using group_names = std::vector<group_name>;

/** @brief Multiple group names without an order, e.g., to test for membership */
using group_names_unordered = std::unordered_set<group_name>;

/** @brief Multiple group ids in the order they were read */
using group_ids = std::vector<group_id>;

/** @brief Multiple group ids without an order, e.g., to test for membership */
using group_ids_unordered = std::unordered_set<group_id>;

/**
 * @brief Spells a literal as one of the aliases above, i.e., the counterpart of utility::as for the
 *      headers that cannot include cpp-utility, such as Config.h, which nvcc compiles as C++17.
 *      Writing the conversion as a cast at the call site instead would either truncate implicitly
 *      (when the alias is narrower than the literal) or cast to the type the literal already has
 *      (when it is not), and one of the two warns in every configuration of the aliases.
 * @tparam T The type the literal shall have, i.e., one of the aliases above
 * @tparam U The type of the literal, which is deduced
 * @param value The literal
 * @return The value represented as @p T
 */
template <typename T, typename U>
[[nodiscard]] constexpr T as(const U value) noexcept {
    return static_cast<T>(value);
}

} // namespace RelearnTypes
