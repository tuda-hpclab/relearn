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

#include "Types.h"

#include <concepts>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <variant>
#include <vector>

// The types that can be interesting
using Parameter = std::variant<float, bool, std::int32_t, std::uint32_t>;

/**
 * @brief This class is used to monitor the values of neurons during the simulation.
 *  It can handle arbitrary parameters as long as they have the type Parameter.
 */
class NeuronMonitor {
    // All values for one neuron in one time step
    using NeuronInformation = std::vector<Parameter>;

    // All values for one neuron in increasing time steps
    using NeuronInformations = std::vector<NeuronInformation>;

    // The callback signature
    using Signature = Parameter(RelearnTypes::number_neurons_type);

public:
    /**
     * @brief Register a parameter with a name and a callable that will be called with the neuron id
     * @tparam Callable The callable type
     * @param name The name of the parameter
     * @param callable The callback that returns a parameter's value for a neuron
     */
    template <typename Callable>
    void register_paramter(const std::string& name, Callable&& callable)
        requires std::invocable<Callable, RelearnTypes::number_neurons_type>
    {
        parameter_names.push_back(name);
        callbacks.emplace_back(std::forward<Callable>(callable));
    }

    /**
     * @brief Sets wjere the output should be written to
     * @param path The path to write to
     * @param clear_contents True iff other files should be deleted
     */
    void set_output_path(std::filesystem::path path, bool clear_contents);

    /**
     * @brief Registers a neuron to be monitored
     * @param neuron The id
     */
    void register_neuron(RelearnTypes::number_neurons_type neuron);

    /**
     * @brief Records the data for the current step, i.e.,
     *  calls all registered callbacks and stores the results
     * @param current_step The current simulation step
     */
    void record_data(RelearnTypes::step_type current_step);

    /**
     * @brief Flushes the contents to the files
     */
    void flush_current_contents();

private:
    std::unordered_map<RelearnTypes::number_neurons_type, NeuronInformations> information{};

    std::vector<RelearnTypes::number_neurons_type> neurons_to_monitor{};
    std::vector<RelearnTypes::step_type> steps{};

    std::vector<std::function<Signature>> callbacks{};
    std::vector<std::string> parameter_names{};

    std::filesystem::path output_path{};
};
