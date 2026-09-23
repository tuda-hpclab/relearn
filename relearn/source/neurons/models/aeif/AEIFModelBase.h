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

#include "neurons/models/NeuronModel.h"
#include "neurons/models/aeif/Parameters.h"

namespace models {
/**
 * This class inherits from NeuronModel and implements an exponential spiking model from Brette and Gerstner.
 * The differential equations are:
 *      d/dt v(t) = (-g_L * (v(t) - E_L) + g_L * d_T * exp((v(t) - V_T) / d_T) - w(t) + input) / C
 *      d/dt w(t) = (a * (v(t) - E_L) - w(t)) / tau_W
 * If v(t) >= V_spike:
 *      v(t) = E_L
 *      w(t) += b
 *      Holds everything that is common to the CPU and GPU AEIF models; AEIFModelCPU and
 *      AEIFModelGPU add the parts that differ (update_activity(_benchmark)).
 */
class AEIFModelBase : public NeuronModel {
public:
    /**
     * @brief Constructs a new instance of type AEIFModelBase with 0 neurons and the passed values for all parameters.
     * @param h See NeuronModel(...)
     * @param activity_input See NeuronModel(...)
     * @param fired_status_communicator See NeuronModel(...)
     * @param params The model parameters
     */
    AEIFModelBase(
        unsigned int h,
        std::shared_ptr<ActivityInput> activity_input,
        std::shared_ptr<FiredStatusCommunicator> fired_status_communicator,
        const models::aeif::Parameters<RelearnTypes::activity_type>& params);

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(number_neurons_type number_neurons) override;

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(number_neurons_type creation_count) override;

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor) override;

    /**
     * @brief Returns the adaptation variable w
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The adaptation variable w
     */
    [[nodiscard]] activity_type get_w(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < get_number_neurons(), "In PoissonModel::get_secondary_variable, id is too large");
        return w[local_neuron_id];
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override;

    /**
     * @brief Returns the parameters of the model
     * @return A constant reference to the parameters
     */
    [[nodiscard]] const models::aeif::Parameters<RelearnTypes::activity_type>& get_model_parameters() const noexcept {
        return parameters;
    }

protected:
    void init_neurons(number_neurons_type start_id, number_neurons_type end_id) final;

    LazySyncedArray<activity_type> w; // adaption variable

    models::aeif::Parameters<RelearnTypes::activity_type> parameters;
};

} // namespace models
