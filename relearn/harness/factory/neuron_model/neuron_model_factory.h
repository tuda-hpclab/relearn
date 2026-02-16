#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/models/NeuronModel.h"

#include <memory>

class NeuronModelFactory {
public:
    static std::unique_ptr<NeuronModel> construct_poisson_model(unsigned int h = 10);

    static std::unique_ptr<NeuronModel> construct_izhikevich_model(unsigned int h = 10);

    static std::unique_ptr<NeuronModel> construct_fitzhughnaguma_model(unsigned int h = 10);

    static std::unique_ptr<NeuronModel> construct_aeif_model(unsigned int h = 10);

    static std::unique_ptr<NeuronModel> construct_poisson_model(unsigned int h, std::shared_ptr<ActivityInput>&& activity_input,
                                                                std::shared_ptr<FiredStatusCommunicator>&& fired_status_comm);

    static std::unique_ptr<NeuronModel> construct_izhikevich_model(unsigned int h, std::shared_ptr<ActivityInput>&& activity_input,
                                                                   std::shared_ptr<FiredStatusCommunicator>&& fired_status_comm);

    static std::unique_ptr<NeuronModel> construct_fitzhughnaguma_model(unsigned int h, std::shared_ptr<ActivityInput>&& activity_input,
                                                                       std::shared_ptr<FiredStatusCommunicator>&& fired_status_comm);

    static std::unique_ptr<NeuronModel> construct_aeif_model(unsigned int h, std::shared_ptr<ActivityInput>&& activity_input,
                                                             std::shared_ptr<FiredStatusCommunicator>&& fired_status_comm);
};
