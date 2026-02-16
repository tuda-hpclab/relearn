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

#include "Config.h"
#include "Types.h"

#include "neurons/enums/FiredStatus.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"

#include <boost/dynamic_bitset.hpp>

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace utility {
class MemoryFootprint;
}

class NeuronMonitor;

/**
 * This types records the number of times a neuron has fired.
 */
class FiredStatusRecorder {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using counter_type = unsigned int;

    /**
     * This enum defines the different periods for which the fire recorder can be used.
     */
    enum class FireRecorderPeriod : std::uint8_t {
        NeuronMonitor = 0,
        GroupMonitor = 1,
        Plasticity = 2
    };

    constexpr static std::size_t number_fire_recorders = 3;

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0 or if init(...) has been called before
     */
    void init(number_neurons_type number_neurons);

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(number_neurons_type creation_count);

    /**
     * @brief Disables the given neurons
     * @param neuron_ids The neurons to disable
     * @exception Throws a RelearnException if a NeuronID is out of bounds
     */
    void disable_neurons(std::span<const NeuronID> neuron_ids);

    /**
     * @brief Does nothing
     * @param neuron_ids Ingored
     */
    void enable_neurons([[maybe_unused]] std::span<const NeuronID> neuron_ids) { }

    /**
     * @brief Returns the number of local neurons
     * @return The number of local neurons
     */
    [[nodiscard]] number_neurons_type get_number_local_neurons() const noexcept {
        return number_local_neurons;
    }

    /**
     * @brief Sets the fired status of the given neuron. Has only an effect if new_value == FiredStatus::Fired
     * @param neuron_id The neuron id to set the fired status for
     * @param new_value The status of the neuron
     * @exception Throws a RelearnException if the neuron_id is out of bounds
     */
    void set_fired(NeuronID::value_type neuron_id, FiredStatus new_value);

    /**
     * @brief Sets the fired status of the given neuron. Has only an effect if new_value == FiredStatus::Fired
     * @param neuron_id The neuron id to set the fired status for
     * @param new_value The status of the neuron
     * @exception Throws a RelearnException if the neuron_id is out of bounds
     */
    void set_fired(NeuronID neuron_id, FiredStatus new_value);

    /**
     * @brief Returns a non-owning view on the fired status of the local neurons
     * @return A non-owning span
     */
    [[nodiscard]] std::span<const FiredStatus> get_fired() const noexcept {
        return fired;
    }

    /**
     * @brief Checks if the local neuron fired
     * @param neuron_id The local neuron
     * @exception Throws a RelearnException if the neuron_id is too large
     * @return True iff the neuron fired
     */
    [[nodiscard]] bool has_fired(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "FiredStatusRecorder::has_fired: id is too large: {}", neuron_id);
        return fired[local_neuron_id] == FiredStatus::Fired;
    }

    /**
     * @brief Returns the number of times a neuron has fired in the given period
     * @param fire_recorder_period The period to get the data for
     * @return A span to the data
     */
    [[nodiscard]] std::span<const counter_type> get_fired_recorder(FireRecorderPeriod fire_recorder_period) const noexcept;

    /**
     * @brief Resets the data for the given period
     * @param period The period to reset
     */
    void reset(FireRecorderPeriod period);

    /**
     * @brief Registers parameters for the neuron monitor.
     *      Also calls this function recursively
     * @param monitor The monitor that collects the data
     */
    void register_neuron_monitor(NeuronMonitor& monitor);

    /**
     * @brief Returns the fire history of a specified neuron.
     * @param neuron_id The local neuron's id
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     * @return The fire history of the neuron
     */
    [[nodiscard]] const boost::dynamic_bitset<>& get_fire_history(NeuronID neuron_id) const;

    /**
     * @brief Returns the fire history of a specified neuron. Downloads data via MPI if the neuron does not
     *  belong to the local rank.
     * @param rank_neuron_id The specification of the neuron
     * @exception Throws an RelearnException if the specified id exceeds the number of stored neurons
     * @return The fire history of the neuron
     */
    [[nodiscard]] boost::dynamic_bitset<> get_fire_history(const RankNeuronId& rank_neuron_id) const;

    /**
     * @brief Returns for how many recent steps we can store if a neuron fired
     * @return Number of steps
     */
    [[nodiscard]] RelearnTypes::step_type get_fire_history_size() const noexcept {
        return fire_history_length;
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint);

private:
    number_neurons_type number_local_neurons{ 0 };

    // How often the neurons have spiked
    std::array<std::vector<counter_type, RelearnAllocator<counter_type>>, number_fire_recorders> fired_recorder{};

    RelearnTypes::step_type fire_history_length{ Config::fire_history_reset_step };
    std::vector<boost::dynamic_bitset<>> fire_history{};

    // The current fired status
    std::vector<FiredStatus, RelearnAllocator<FiredStatus>> fired{};
};
