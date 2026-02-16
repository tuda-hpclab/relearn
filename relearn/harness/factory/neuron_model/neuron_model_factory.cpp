/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neuron_model_factory.h"

#include "neurons/firing/FiredStatusCommunicationMap.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/aeif/AEIFModel.h"
#include "neurons/models/aeif/Parameters.h"
#include "neurons/models/fitzhughnagumo/FitzHughNagumoModel.h"
#include "neurons/models/fitzhughnagumo/Parameters.h"
#include "neurons/models/izhikevich/IzhikevichModel.h"
#include "neurons/models/izhikevich/Parameters.h"
#include "neurons/models/poisson/Parameters.h"
#include "neurons/models/poisson/PoissonModel.h"

#include "factory/activity_input/activity_input_factory.h"

#include <memory>
#include <utility>

std::unique_ptr<NeuronModel> NeuronModelFactory::construct_poisson_model(const unsigned int h) {
    auto fired_status_comm = std::make_shared<FiredStatusCommunicationMap>(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity();

    return construct_poisson_model(h, std::move(activity_input), std::move(fired_status_comm));
}

std::unique_ptr<NeuronModel> NeuronModelFactory::construct_izhikevich_model(const unsigned int h) {
    auto fired_status_comm = std::make_shared<FiredStatusCommunicationMap>(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity();

    return construct_izhikevich_model(h, std::move(activity_input), std::move(fired_status_comm));
}

std::unique_ptr<NeuronModel> NeuronModelFactory::construct_fitzhughnaguma_model(const unsigned int h) {
    auto fired_status_comm = std::make_shared<FiredStatusCommunicationMap>(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity();

    return construct_fitzhughnaguma_model(h, std::move(activity_input), std::move(fired_status_comm));
}

std::unique_ptr<NeuronModel> NeuronModelFactory::construct_aeif_model(const unsigned int h) {
    auto fired_status_comm = std::make_shared<FiredStatusCommunicationMap>(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity();

    return construct_aeif_model(h, std::move(activity_input), std::move(fired_status_comm));
}

std::unique_ptr<NeuronModel> NeuronModelFactory::construct_poisson_model(const unsigned int h,
                                                                         std::shared_ptr<ActivityInput>&& activity_input,
                                                                         std::shared_ptr<FiredStatusCommunicator>&& fired_status_comm) {

    const auto parameters = models::poisson::Parameters<double, unsigned int>{};
    return std::make_unique<models::PoissonModel>(h, std::move(activity_input), std::move(fired_status_comm), parameters);
}

std::unique_ptr<NeuronModel> NeuronModelFactory::construct_izhikevich_model(const unsigned int h,
                                                                            std::shared_ptr<ActivityInput>&& activity_input,
                                                                            std::shared_ptr<FiredStatusCommunicator>&& fired_status_comm) {

    const auto parameters = models::izhikevich::Parameters<double>{};
    return std::make_unique<models::IzhikevichModel>(h, std::move(activity_input), std::move(fired_status_comm), parameters);
}

std::unique_ptr<NeuronModel> NeuronModelFactory::construct_fitzhughnaguma_model(const unsigned int h,
                                                                                std::shared_ptr<ActivityInput>&& activity_input,
                                                                                std::shared_ptr<FiredStatusCommunicator>&& fired_status_comm) {

    const auto parameters = models::fitzhughnagumo::Parameters<double>{};
    return std::make_unique<models::FitzHughNagumoModel>(h, std::move(activity_input), std::move(fired_status_comm), parameters);
}

std::unique_ptr<NeuronModel> NeuronModelFactory::construct_aeif_model(const unsigned int h,
                                                                      std::shared_ptr<ActivityInput>&& activity_input,
                                                                      std::shared_ptr<FiredStatusCommunicator>&& fired_status_comm) {

    const auto parameters = models::aeif::Parameters<double>{};
    return std::make_unique<models::AEIFModel>(h, std::move(activity_input), std::move(fired_status_comm), parameters);
}
