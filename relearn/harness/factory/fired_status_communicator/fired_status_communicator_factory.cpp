/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "fired_status_communicator_factory.h"

#include "../../../source/cuda/firing/FireStatusCommunicatorGPUUncompressed.h"
#include "neurons/firing/FiredStatusApproximator.h"
#include "neurons/firing/FiredStatusCommunicationMap.h"
#include "neurons/firing/FiredStatusCommunicator.h"

#include <cstddef>
#include <memory>

std::shared_ptr<FiredStatusCommunicator> FiredStatusCommunicatorFactory::construct_map_communicator(const int number_ranks, const std::size_t size_hint) {
    return std::make_shared<FiredStatusCommunicationMap>(mpiPP::MPIRank::root_rank(),number_ranks, size_hint);
}

std::shared_ptr<FiredStatusCommunicator> FiredStatusCommunicatorFactory::construct_approximator(const int number_ranks) {
    return std::make_shared<FiredStatusApproximator>(mpiPP::MPIRank::root_rank(), number_ranks);
}

std::shared_ptr<FiredStatusCommunicator> FiredStatusCommunicatorFactory::construct_default_communicator(const int number_ranks) {
#ifdef RELEARN_CUDA_ENABLED
    return std::make_shared<FireStatusCommunicatorGPUUncompressed>(mpiPP::MPIRank::root_rank(),number_ranks);
#else
    return std::make_shared<FiredStatusCommunicationMap>(mpiPP::MPIRank::root_rank(),number_ranks, 1);
#endif
}