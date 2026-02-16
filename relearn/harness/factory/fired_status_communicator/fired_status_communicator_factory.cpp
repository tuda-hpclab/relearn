/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "fired_status_communicator_factory.h"

#include "neurons/firing/FiredStatusApproximator.h"
#include "neurons/firing/FiredStatusCommunicationMap.h"
#include "neurons/firing/FiredStatusCommunicator.h"

#include <cstddef>
#include <memory>

std::shared_ptr<FiredStatusCommunicator> FiredStatusCommunicatorFactory::construct_map_communicator(const int number_ranks, const std::size_t size_hint) {
    return std::make_shared<FiredStatusCommunicationMap>(number_ranks, size_hint);
}

std::shared_ptr<FiredStatusCommunicator> FiredStatusCommunicatorFactory::construct_approximator(const int number_ranks) {
    return std::make_shared<FiredStatusApproximator>(number_ranks);
}
