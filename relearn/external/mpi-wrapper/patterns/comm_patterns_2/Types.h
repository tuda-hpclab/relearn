#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/core/MPIRank.h"

#include <cstdint>
#include <tuple>
#include <utility>

namespace mpiPP {

namespace comm_patterns_2 {

/** Default application-defined identifier type for a node-local target. */
using LocalIdentifierType = std::uint64_t;

/** Index type used to address flattened question and answer storage. */
using Index = std::uint64_t;

/** Half-open interval [first, second) in flattened question or answer storage. */
using IndexRange = std::pair<Index, Index>;

/**
 * @brief One question of a local node: the rank and the local node id the question is addressed to,
 *      plus the question payload itself.
 */
template <typename QParameter, typename IdentifierType>
using TargetedQuestion = std::tuple<MPIRank, IdentifierType, QParameter>;

} // namespace comm_patterns_2

} // namespace mpiPP
