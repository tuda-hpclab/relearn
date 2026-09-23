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

#include "neurons/enums/FiredStatus.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"

#include <array>
#include <span>
#include <vector>

/**
 * This types records the number of times a neuron has fired.
 *      Holds everything that is common to the CPU and GPU recorders; FiredStatusRecorderCPU and
 *      FiredStatusRecorderGPU add the parts that differ (how the per-period fire counters are
 *      stored and updated, which is host-only on the CPU and additionally mirrored to the device
 *      on the GPU).
 *      Templated on the per-neuron array storage (Storage<T>): FiredStatusRecorderCPU instantiates
 *      it with std::vector, FiredStatusRecorderGPU with LazySyncedArray.
 */
template <template <typename...> class Storage>
class FiredStatusRecorderBase {
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

        RelearnException::check(local_neuron_id < number_local_neurons,
                                "FiredStatusRecorderBase::has_fired: id is too large: {}", neuron_id);
        return fired[local_neuron_id] == FiredStatus::Fired;
    }

protected:
    number_neurons_type number_local_neurons{ 0 };

    // How often the neurons have spiked
    std::array<std::vector<counter_type, RelearnAllocator<counter_type>>, number_fire_recorders> fired_recorder{};

    // The current fired status
    Storage<FiredStatus> fired{};
};

template <template <typename...> class Storage>
void FiredStatusRecorderBase<Storage>::disable_neurons(const std::span<const NeuronID> neuron_ids) {

    for (const auto neuron_id : neuron_ids) {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        RelearnException::check(local_neuron_id < number_local_neurons,
                                "FiredStatusRecorderBase::disable_neurons: Neuron ID out of bounds.");

        for (auto& recorder : fired_recorder) {
            recorder[local_neuron_id] = 0U;
        }

        fired[local_neuron_id] = FiredStatus::Inactive;
    }
}

template <template <typename...> class Storage>
void FiredStatusRecorderBase<Storage>::set_fired(const NeuronID::value_type neuron_id, const FiredStatus new_value) {
    RelearnException::check(neuron_id < number_local_neurons,
                            "FiredStatusRecorderBase::set_fired: Neuron ID out of bounds.");

    fired[neuron_id] = new_value;

    if (new_value != FiredStatus::Fired) {
        return;
    }

    for (auto& recorder : fired_recorder) {
        recorder[neuron_id]++;
    }
}
