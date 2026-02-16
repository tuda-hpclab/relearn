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
#include "neurons/models/poisson/Parameters.h"

namespace models {
/**
 * This class inherits from NeuronModel and implements a poisson spiking model
 */
class PoissonModel : public NeuronModel {
public:
    /**
     * @brief Constructs a new instance of type PoissonModel with 0 neurons and the passed values for all parameters.
     * @param h See NeuronModel(...)
     * @param activity_input See NeuronModel(...)
     * @param fired_status_communicator See NeuronModel(...)
     * @param params The model parameters
     */
    PoissonModel(
        unsigned int h,
        std::shared_ptr<ActivityInput> activity_input,
        std::shared_ptr<FiredStatusCommunicator> fired_status_communicator,
        const models::poisson::Parameters<double, unsigned int>& params);

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     *      Sets the initial refractory_time counter to 0
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
     * @brief Returns the refractory time
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The refractory time
     */
    [[nodiscard]] unsigned int get_refractory_time(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < get_number_neurons(), "In PoissonModel::get_secondary_variable, id is too large");
        return refractory_time[local_neuron_id];
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
    [[nodiscard]] const models::poisson::Parameters<double, unsigned int>& get_model_parameters() const noexcept {
        return parameters;
    }

protected:
    void update_activity() final;

    void update_activity_benchmark() final;

    void init_neurons(number_neurons_type start_id, number_neurons_type end_id) final;

private:
    std::vector<unsigned int, RelearnAllocator<unsigned int>> refractory_time{}; // refractory time

    models::poisson::Parameters<double, unsigned int> parameters{};
};

} // namespace models
