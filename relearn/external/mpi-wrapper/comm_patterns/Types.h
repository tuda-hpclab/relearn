#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/MPIRank.h"
#include "mpi-wrapper/MPITypes.h"

#include <concepts>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace mpiPP {

namespace comm_patterns {
using local_identifier_type = std::uint64_t;

template <typename A_parameter, typename identifier_type>
class NodeToNodeAnswer;

using Index = std::uint64_t;

using IndexRange = std::pair<Index, Index>;

template <typename Q_parameter, typename identifier_type>
using TargetedQuestion = std::tuple<MPIRank, identifier_type, Q_parameter>;

template <typename Q_parameter, typename identifier_type>
using GenerateQuestionsFunction = std::function<std::vector<TargetedQuestion<Q_parameter, identifier_type>>(identifier_type local_id)>;

template <typename A_parameter, typename Q_parameter, typename identifier_type>
using GenerateAnswersFunction = std::function<A_parameter(identifier_type, Q_parameter)>;

template <typename Q_parameter, typename identifier_type>
using VoidAnswersFunction = std::function<void(identifier_type, Q_parameter)>;

template <typename Q_parameter, typename T, typename identifier_type>
concept QFunction = std::convertible_to<GenerateQuestionsFunction<Q_parameter, identifier_type>, T>;

template <typename T>
concept MPICompatibleAnswer = MPICompatible<T> || std::same_as<void, T>;

} // namespace comm_patterns

} // namespace mpiPP
