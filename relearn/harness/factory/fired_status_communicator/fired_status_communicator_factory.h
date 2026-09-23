#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/firing/FiredStatusCommunicator.h"

#include <memory>

class FiredStatusCommunicatorFactory {
public:
    static std::shared_ptr<FiredStatusCommunicator> construct_map_communicator(int number_ranks = 1, std::size_t size_hint = 1);

    static std::shared_ptr<FiredStatusCommunicator> construct_approximator(int number_ranks = 1);

    static std::shared_ptr<FiredStatusCommunicator> construct_default_communicator(int number_ranks = 1);
};
