#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/helper/NeuronMonitor.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <functional>
#include <memory>
#include <span>

/**
 * @brief This class focuses on calculating the inter-cellular calcium concentration of the neurons.
 *      It offers the functionality for neuron-dependent target values.
 *      Holds everything that is common to the CPU and GPU calculators; CalciumCalculatorCPU and
 *      CalciumCalculatorGPU add the parts that differ (how the current calcium is updated from the
 *      fired status, which is a host span on the CPU and a device pointer on the GPU).
 *      Templated on the per-neuron array storage (Storage<T>): CalciumCalculatorCPU instantiates
 *      it with std::vector, CalciumCalculatorGPU with LazySyncedArray -- this is the only thing
 *      that differs between the two, and keeping it a template parameter (rather than naming
 *      LazySyncedArray directly here) means this header has no dependency on anything under cuda/.
 */
template <template <typename...> class Storage>
class CalciumCalculatorBase {
public:
    using calcium_type = RelearnTypes::calcium_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Constructs a new calculator without decaying target calcium
     */
    CalciumCalculatorBase() = default;

    CalciumCalculatorBase(const CalciumCalculatorBase&) = delete;
    CalciumCalculatorBase& operator=(const CalciumCalculatorBase&) = delete;

    CalciumCalculatorBase(CalciumCalculatorBase&&) = default;
    CalciumCalculatorBase& operator=(CalciumCalculatorBase&&) = default;

    virtual ~CalciumCalculatorBase() = default;

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its electrical activity
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) { // NOLINT(performance-unnecessary-value-param) - moved into extra_infos below
        const auto is_filled = new_extra_info != nullptr;
        RelearnException::check(is_filled, "CalciumCalculatorBase::set_extra_infos: new_extra_info is empty");
        extra_infos = std::move(new_extra_info);
    }

    /**
     * @brief Sets beta, the constant by which the calcium increases every time a neuron spikes
     * @param new_beta The new value for beta
     * @exception Throws a RelearnException if the new value is not in the given interval by minimum and maximum
     */
    void set_beta(const calcium_type new_beta) {
        RelearnException::check(min_beta <= new_beta, "CalciumCalculatorBase::set_beta: new_beta was smaller than the minimum: {} vs {}", new_beta, min_beta);
        RelearnException::check(new_beta <= max_beta, "CalciumCalculatorBase::set_beta: new_beta was larger than the maximum: {} vs {}", new_beta, max_beta);
        beta = new_beta;
    }

    /**
     * @brief Returns beta, increase-in-calcium constant
     * @return beta
     */
    [[nodiscard]] calcium_type get_beta() const noexcept {
        return beta;
    }

    /**
     * @brief Sets the dampening factor for the calcium decrease (the decay constant)
     * @param new_tau_C The dampening factor
     * @exception Throws a RelearnException if the new value is not in the given interval by minimum and maximum
     */
    void set_tau_C(const calcium_type new_tau_C) {
        RelearnException::check(min_tau_C <= new_tau_C, "CalciumCalculatorBase::set_tau_C: new_tau_C was smaller than the minimum: {} vs {}", new_tau_C, min_tau_C);
        RelearnException::check(new_tau_C <= max_tau_C, "CalciumCalculatorBase::set_tau_C: new_tau_C was larger than the maximum: {} vs {}", new_tau_C, max_tau_C);
        tau_C = new_tau_C;
    }

    /**
     * @brief Returns tau_C (The dampening factor by which the calcium decreases)
     * @return the dampening factor
     */
    [[nodiscard]] calcium_type get_tau_C() const noexcept {
        return tau_C;
    }

    /**
     * @brief Sets the numerical integration's step size
     * @param new_h The new step size
     * @exception Throws a RelearnException if the new value is not in the given interval by minimum and maximum
     */
    void set_h(const unsigned int new_h) {
        RelearnException::check(min_h <= new_h, "CalciumCalculatorBase::set_h: new_h was smaller than the minimum: {} vs {}", new_h, min_h);
        RelearnException::check(new_h <= max_h, "CalciumCalculatorBase::set_h: new_h was larger than the maximum: {} vs {}", new_h, max_h);
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
    [[nodiscard]] std::span<const calcium_type> get_calcium() const noexcept {
        return calcium;
    }

    /**
     * @brief Returns the target calcium values
     * @return The target calcium values
     */
    [[nodiscard]] std::span<const calcium_type> get_target_calcium() const noexcept {
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
    void set_initial_calcium_calculator(std::function<calcium_type(mpiPP::MPIRank, NeuronID::value_type)> initiator) noexcept {
        initial_calcium_initiator = std::move(initiator);
    }

    /**
     * @brief Sets the function that is used to determine the target calcium value of the neurons
     *      When calling init(...), the target calcium calculator must not be empty. It can be so inbetween.
     * @param calculator The function that maps neuron id to target calcium value
     */
    void set_target_calcium_calculator(std::function<calcium_type(mpiPP::MPIRank, NeuronID::value_type)> calculator) noexcept {
        target_calcium_calculator = std::move(calculator);
    }

    /**
     * @brief Initializes the given number of neurons, uses the previously passed functions to determine the initial and target values
     * @param number_neurons The number of neurons, must be > 0
     * @exception Throws a RelearnException if any of the functions is empty or number_neurons == 0
     */
    void init(number_neurons_type number_neurons);

    /**
     * @brief Creates the given number of neurons, uses the previously passed functions to determine the initial and target values
     * @param number_neurons The number of neurons, must be > 0
     * @exception Throws a RelearnException if any of the functions is empty or number_neurons == 0
     */
    void create_neurons(number_neurons_type number_neurons);

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

    static constexpr calcium_type default_C_target{ utility::as<calcium_type>(0.7) }; // In Sebastian's work: 0.5

    static constexpr calcium_type default_tau_C{ 10000 };                           // In Sebastian's work: 5000
    static constexpr calcium_type default_beta{ utility::as<calcium_type>(0.001) }; // In Sebastian's work: 0.001
    static constexpr unsigned int default_h{ 10 };

    static constexpr calcium_type min_tau_C{ 0 };
    static constexpr calcium_type min_beta{ 0.0 };
    static constexpr unsigned int min_h{ 1 };

    static constexpr calcium_type max_tau_C{ 10.0e+6 };
    static constexpr calcium_type max_beta{ 1.0 };
    static constexpr unsigned int max_h{ 1000 };

protected:
    /**
     * @brief Returns the inter-cellular calcium concentration
     * @return The calcium values
     */
    [[nodiscard]] std::span<calcium_type> get_calcium_internal() noexcept {
        return calcium;
    }

    /**
     * @brief Returns the target calcium values
     * @return The target calcium values
     */
    [[nodiscard]] std::span<calcium_type> get_target_calcium_internal() noexcept {
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

    std::function<calcium_type(mpiPP::MPIRank, NeuronID::value_type)> initial_calcium_initiator{};
    std::function<calcium_type(mpiPP::MPIRank, NeuronID::value_type)> target_calcium_calculator{};

    Storage<calcium_type> calcium{};
    Storage<calcium_type> target_calcium{};

    std::shared_ptr<NeuronsExtraInfo> extra_infos;

    calcium_type beta{ default_beta };
    calcium_type tau_C{ default_tau_C }; // Decay time of calcium
    unsigned int h{ default_h };         // Precision for Euler integration

    NeuronID current_minimum{ NeuronID::uninitialized_id() };
    NeuronID current_maximum{ NeuronID::uninitialized_id() };
};

template <template <typename...> class Storage>
void CalciumCalculatorBase<Storage>::init(const number_neurons_type number_neurons) {
    RelearnException::check(calcium.empty(), "CalciumCalculatorBase::init: Was already initialized");
    RelearnException::check(number_neurons > 0, "CalciumCalculatorBase::init: number_neurons was 0");

    RelearnException::check(initial_calcium_initiator != nullptr, "CalciumCalculatorBase::init: initial_calcium_initiator is empty");
    RelearnException::check(target_calcium_calculator != nullptr, "CalciumCalculatorBase::init: target_calcium_calculator is empty");

    calcium.resize(number_neurons);
    target_calcium.resize(number_neurons);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    for (auto neuron_id = 0U; neuron_id < number_neurons; neuron_id++) {
        calcium[neuron_id] = initial_calcium_initiator(my_rank, neuron_id);
        target_calcium[neuron_id] = target_calcium_calculator(my_rank, neuron_id);
    }
}

template <template <typename...> class Storage>
void CalciumCalculatorBase<Storage>::create_neurons(const number_neurons_type number_neurons) {
    RelearnException::check(!calcium.empty(), "CalciumCalculatorBase::create_neurons: Was not initialized");
    RelearnException::check(number_neurons > 0, "CalciumCalculatorBase::create_neurons: number_neurons was 0");

    RelearnException::check(initial_calcium_initiator != nullptr, "CalciumCalculatorBase::create_neurons: initial_calcium_initiator is empty");
    RelearnException::check(target_calcium_calculator != nullptr, "CalciumCalculatorBase::create_neurons: target_calcium_calculator is empty");

    const auto old_size = calcium.size();
    const auto new_size = old_size + number_neurons;

    calcium.resize(new_size);
    target_calcium.resize(new_size);

    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    for (auto neuron_id = old_size; neuron_id < new_size; neuron_id++) {
        calcium[neuron_id] = initial_calcium_initiator(my_rank, neuron_id);
        target_calcium[neuron_id] = target_calcium_calculator(my_rank, neuron_id);
    }
}

template <template <typename...> class Storage>
void CalciumCalculatorBase<Storage>::record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
    const auto my_footprint = sizeof(*this)
                              + (calcium.capacity() * sizeof(calcium_type))
                              + (target_calcium.capacity() * sizeof(calcium_type));
    footprint->emplace("CalciumCalculator", my_footprint);
}

template <template <typename...> class Storage>
void CalciumCalculatorBase<Storage>::register_neuron_monitor(NeuronMonitor& monitor) {
    monitor.register_paramter("Calcium", [this](const RelearnTypes::number_neurons_type neuron_id) { return utility::cast<float>(calcium[neuron_id]); }, []() { }, []() { });

    monitor.register_paramter("Target Calcium", [this](const RelearnTypes::number_neurons_type neuron_id) { return utility::cast<float>(target_calcium[neuron_id]); }, []() { }, []() { });

    monitor.register_paramter("Calcium Difference", [this](const RelearnTypes::number_neurons_type neuron_id) { return utility::cast<float>(target_calcium[neuron_id]) - utility::cast<float>(calcium[neuron_id]); }, []() { }, []() { });
}
