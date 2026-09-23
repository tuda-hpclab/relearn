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
#include "mpi-wrapper/core/MPITypes.h"

#include <concepts>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace mpiPP {

namespace comm_patterns {
/** Default unsigned type used for local node identifiers. */
using LocalIdentifierType = std::uint64_t;

/** Unsigned offset into a flattened question or answer buffer. */
using Index = std::uint64_t;

/** Half-open [begin, end) range in a flattened question or answer buffer. */
using IndexRange = std::pair<Index, Index>;

/** Question payload together with its target MPI rank and target-local identifier. */
template <typename QParameter, typename IdentifierType>
using TargetedQuestion = std::tuple<MPIRank, IdentifierType, QParameter>;

/** Callback that produces all targeted questions for one local identifier. */
template <typename QParameter, typename IdentifierType>
using GenerateQuestionsFunction = std::function<std::vector<TargetedQuestion<QParameter, IdentifierType>>(IdentifierType local_id)>;

/** Callback that produces an answer from the addressed local identifier and question payload. */
template <typename AParameter, typename QParameter, typename IdentifierType>
using GenerateAnswersFunction = std::function<AParameter(IdentifierType, QParameter)>;

/** Callback used when processing questions has side effects but produces no answer payload. */
template <typename QParameter, typename IdentifierType>
using VoidAnswersFunction = std::function<void(IdentifierType, QParameter)>;

/** Types supported as an answer payload, including void for one-way questions. */
template <typename T>
concept MPICompatibleAnswer = MPICompatible<T> || std::same_as<void, T>;

} // namespace comm_patterns

} // namespace mpiPP
