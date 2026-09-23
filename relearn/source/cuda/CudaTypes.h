#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Macros.h"

/**
 * A 3-D position with double-precision components, used for neuron positions in the octree.
 */
struct SimpleVec3d {
    double x, y, z;
};

/**
 * Classifies a node in the linearized Barnes-Hut octree.
 */
enum class NodeType : char {
    Leaf,        ///< Local leaf node representing a single neuron.
    VirtualNode, ///< Internal aggregation node without a physical neuron.
    RemoteNode,  ///< Placeholder for a subtree owned by another MPI rank.
    Placeholder  ///< Reserved slot (padding or not-yet-filled entry).
};

namespace CudaTypes {

/**
 * @brief The floating point type in which the device code calculates, i.e., the precision of the algorithms
 *      on the GPU. It follows the same PRECISION of CMake as RelearnTypes::real does, which is what keeps
 *      both sides of every value that crosses the host-device boundary in the same precision.
 */
#if RELEARN_PRECISION_FP32
using cuda_real = float;
#else
using cuda_real = double;
#endif

} // namespace CudaTypes

#if CUDA_FOUND
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <cuComplex.h>
#pragma GCC diagnostic pop

namespace CudaTypes {

namespace detail {
/**
 * @brief Maps a floating point type to the vector type of the CUDA runtime that holds three of its values.
 *      The primary template is left undefined, so a cuda_real for which the runtime offers no such vector
 *      type fails to compile instead of silently picking one of the two.
 * @tparam T The floating point type, i.e., float or double
 */
template <typename T>
struct real3;

template <>
struct real3<float> {
    using type = float3;
};

template <>
struct real3<double> {
    using type = double3;
};
} // namespace detail

/** @brief Three values of cuda_real, i.e., the vector type of the CUDA runtime that matches the precision of the device code */
using cuda_real3 = detail::real3<cuda_real>::type;

} // namespace CudaTypes

#else

#include <array>

namespace CudaTypes {

/** @brief Three values of cuda_real, i.e., the stand-in for the vector type of the CUDA runtime in a build without CUDA */
using cuda_real3 = std::array<cuda_real, 3>;

} // namespace CudaTypes
#endif

#include <cstdint>

// The device-side counterparts of the domain types in RelearnTypes. They are spelled apart so that the
// precision of the device code could be chosen independently one day; today both sides read the same
// PRECISION, and every value that is copied across the host-device boundary must use the alias that
// matches its RelearnTypes counterpart.
namespace CudaTypes {

/** @brief A single coordinate or length in the simulation space, i.e., the counterpart of RelearnTypes::space_type */
using cuda_space = cuda_real;

/** @brief The criterion by which Barnes-Hut accepts a virtual neuron instead of descending, i.e., the counterpart of RelearnTypes::acceptance_criterion_type */
using cuda_acceptance_criterion = cuda_real;

/** @brief The attractiveness of a target neuron, i.e., the counterpart of RelearnTypes::attraction_type */
using cuda_attraction = cuda_real;

/** @brief A number of synaptic elements, i.e., the counterpart of RelearnTypes::counter_type */
using cuda_counter = std::uint32_t;

} // namespace CudaTypes
