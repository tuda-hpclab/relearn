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

#include "neurons/enums/FiredStatus.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIRank.h"

#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace utility {
class MemoryFootprint;
}

class NeuronMonitor;
class NeuronsExtraInfo;

/**
 * @brief This class focuses on calculating the inter-cellular calcium concentration of the neurons.
 *      It offers the functionality for neuron-dependent target values.
 */
class CalciumCalculator {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Constructs a new calculator without decaying target calcium
     */
    CalciumCalculator() = default;

    CalciumCalculator(const CalciumCalculator&) = default;
    CalciumCalculator& operator=(const CalciumCalculator&) = default;

    CalciumCalculator(CalciumCalculator&&) = default;
    CalciumCalculator& operator=(CalciumCalculator&&) = default;

    virtual ~CalciumCalculator() = default;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its electrical activity
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info != nullptr;
        RelearnException::check(is_filled, "CalciumCalculator::set_extra_infos: new_extra_info is empty");
        extra_infos = std::move(new_extra_info);
    }

    /**
     * @brief Sets beta, the constant by which the calcium increases every time a neuron spikes
     * @param new_beta The new value for beta
     * @exception Throws a RelearnException if the new value is not in the given interval by minimum and maximum
     */
    void set_beta(const double new_beta) {
        RelearnException::check(min_beta <= new_beta, "CalciumCalculator::set_beta: new_beta was smaller than the minimum: {} vs {}", new_beta, min_beta);
        RelearnException::check(new_beta <= max_beta, "CalciumCalculator::set_beta: new_beta was larger than the maximum: {} vs {}", new_beta, max_beta);
        beta = new_beta;
    }

    /**
     * @brief Returns beta, increase-in-calcium constant
     * @return beta
     */
    [[nodiscard]] double get_beta() const noexcept {
        return beta;
    }

    /**
     * @brief Sets the dampening factor for the calcium decrease (the decay constant)
     * @param new_tau_C The dampening factor
     * @exception Throws a RelearnException if the new value is not in the given interval by minimum and maximum
     */
    void set_tau_C(const double new_tau_C) {
        RelearnException::check(min_tau_C <= new_tau_C, "CalciumCalculator::set_tau_C: new_tau_C was smaller than the minimum: {} vs {}", new_tau_C, min_tau_C);
        RelearnException::check(new_tau_C <= max_tau_C, "CalciumCalculator::set_tau_C: new_tau_C was larger than the maximum: {} vs {}", new_tau_C, max_tau_C);
        tau_C = new_tau_C;
    }

    /**
     * @brief Returns tau_C (The dampening factor by which the calcium decreases)
     * @return the dampening factor
     */
    [[nodiscard]] double get_tau_C() const noexcept {
        return tau_C;
    }

    /**
     * @brief Sets the numerical integration's step size
     * @param new_h The new step size
     * @exception Throws a RelearnException if the new value is not in the given interval by minimum and maximum
     */
    void set_h(const unsigned int new_h) {
        RelearnException::check(min_h <= new_h, "CalciumCalculator::set_h: new_h was smaller than the minimum: {} vs {}", new_h, min_h);
        RelearnException::check(new_h <= max_h, "CalciumCalculator::set_h: new_h was larger than the maximum: {} vs {}", new_h, max_h);
        h = new_h;
    }

    /**
     * @brief Returns the numerical integration's step size
     * @return The step size
     */
    [[nodiscard]] unsigned int get_h() const noexcept {
        return h;
    }

    /**
     * @brief Returns the inter-cellular calcium concentration
     * @return The calcium values
     */
    [[nodiscard]] std::span<const double> get_calcium() const noexcept {
        return calcium;
    }

    /**
     * @brief Returns the target calcium values
     * @return The target calcium values
     */
    [[nodiscard]] std::span<const double> get_target_calcium() const noexcept {
        return target_calcium;
    }

    /**
     * @brief Registers parameters for the calcium calculator.
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor);

    /**
     * @brief Sets the function that is used to determine the initial calcium value of the neurons.
     *      When calling init(...), the initial calcium calculator must not be empty. It can be so inbetween.
     * @param initiator The function that maps neuron id to initial calcium value
     */
    void set_initial_calcium_calculator(std::function<double(mpiPP::MPIRank, NeuronID::value_type)> initiator) noexcept {
        initial_calcium_initiator = std::move(initiator);
    }

    /**
     * @brief Sets the function that is used to determine the target calcium value of the neurons
     *      When calling init(...), the target calcium calculator must not be empty. It can be so inbetween.
     * @param calculator The function that maps neuron id to target calcium value
     */
    void set_target_calcium_calculator(std::function<double(mpiPP::MPIRank, NeuronID::value_type)> calculator) noexcept {
        target_calcium_calculator = std::move(calculator);
    }

    /**
     * @brief Initializes the given number of neurons, uses the previously passed functions to determine the initial and target values
     * @param number_neurons The number of neurons, must be > 0
     * @exception Throws a RelearnException if any of the functions is empty or number_neurons == 0
     */
    virtual void init(number_neurons_type number_neurons);

    /**
     * @brief Creates the given number of neurons, uses the previously passed functions to determine the initial and target values
     * @param number_neurons The number of neurons, must be > 0
     * @exception Throws a RelearnException if any of the functions is empty or number_neurons == 0
     */
    virtual void create_neurons(number_neurons_type number_neurons);

    /**
     * @brief Updates the calcium values for each neuron
     * @param step The current update step
     * @param fired_status Indicates if a neuron fired
     * @exception Throws a RelearnException if the size of the vectors doesn't match the size of the stored vectors
     */
    void update_calcium(step_type step, std::span<const FiredStatus> fired_status);

    /**
     * @brief Returns the id of the neuron that has the currently lowest calcium value (on the MPI rank).
     *      Can return an unitialized ID if there was no neuron to update.
     * @return The id
     */
    [[nodiscard]] NeuronID get_current_minimum() const noexcept {
        return current_minimum;
    }

    /**
     * @brief Returns the id of the neuron that has the currently highest calcium value (on the MPI rank).
     *      Can return an unitialized ID if there was no neuron to update.
     * @return The id
     */
    [[nodiscard]] NeuronID get_current_maximum() const noexcept {
        return current_maximum;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint);

    static constexpr double default_C_target{ 0.7 }; // In Sebastian's work: 0.5

    static constexpr double default_tau_C{ 10000 }; // In Sebastian's work: 5000
    static constexpr double default_beta{ 0.001 };  // In Sebastian's work: 0.001
    static constexpr unsigned int default_h{ 10 };

    static constexpr double min_tau_C{ 0 };
    static constexpr double min_beta{ 0.0 };
    static constexpr unsigned int min_h{ 1 };

    static constexpr double max_tau_C{ 10.0e+6 };
    static constexpr double max_beta{ 1.0 };
    static constexpr unsigned int max_h{ 1000 };

protected:
    /**
     * @brief Returns the inter-cellular calcium concentration
     * @return The calcium values
     */
    [[nodiscard]] std::span<double> get_calcium_internal() noexcept {
        return calcium;
    }

    /**
     * @brief Returns the target calcium values
     * @return The target calcium values
     */
    [[nodiscard]] std::span<double> get_target_calcium_internal() noexcept {
        return target_calcium;
    }

    /**
     * @brief Returns the shared poitner to the extra infos
     * @return The extra infos
     */
    [[nodiscard]] const std::shared_ptr<NeuronsExtraInfo>& get_neuron_extra_info() noexcept {
        return extra_infos;
    }

    virtual void update_target_calcium([[maybe_unused]] const step_type step) noexcept { }

private:
    void update_current_calcium(std::span<const FiredStatus> fired_status) noexcept;

    std::function<double(mpiPP::MPIRank, NeuronID::value_type)> initial_calcium_initiator{};
    std::function<double(mpiPP::MPIRank, NeuronID::value_type)> target_calcium_calculator{};

    std::vector<double, RelearnAllocator<double>> calcium{};
    std::vector<double, RelearnAllocator<double>> target_calcium{};

    std::shared_ptr<NeuronsExtraInfo> extra_infos{};

    double beta{ default_beta };
    double tau_C{ default_tau_C }; // Decay time of calcium
    unsigned int h{ default_h };   // Precision for Euler integration

    NeuronID current_minimum{ NeuronID::uninitialized_id() };
    NeuronID current_maximum{ NeuronID::uninitialized_id() };
};
