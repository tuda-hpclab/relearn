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

#include "neurons/synaptic_elements/Axons.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

class MultiPositionAxons : public Axons {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;
    using calcium_type = RelearnTypes::calcium_type;

    MultiPositionAxons() = default;

    MultiPositionAxons(const MultiPositionAxons& other) = delete;
    MultiPositionAxons(MultiPositionAxons&& other) = default;

    MultiPositionAxons& operator=(const MultiPositionAxons& other) = delete;
    MultiPositionAxons& operator=(MultiPositionAxons&& other) = default;

    ~MultiPositionAxons() override = default;

    /**
     * @brief Initializes the object to contain number_neurons elements.
     * @param number_neurons The number of that should be stored, >0
     * @exception Throws a RelearnException if number_neurons == 0 or if init() has been called before
     */
    void init(number_neurons_type number_neurons) override;

    /**
     * @brief Creates additional creation_count elements.
     * @param creation_count The number of that should be created, >0
     * @exception Throws a RelearnException if creation_count == 0 or if init() was not called before
     */
    void create_neurons(number_neurons_type creation_count) override;

    /**
     * @brief Disables the specified neurons. Sets the grown elements, deltas, vacant and connected elements to 0.
     * @param disabled_neuron_ids The neurons to disable
     * @exception Throws a RelearnException if a neuron_id is too large
     */
    void disable_neurons(std::span<const number_neurons_type> disabled_neuron_ids) override;

    /**
     * @brief Sets the bouton positions for all neurons.
     *      Deletes all previous posiitons (if there are any).
     * @param positions The positions of the axonal boutons
     * @exception Throws a RelearnException if positions.size() != get_size()
     */
    void set_bouton_positions(const std::vector<std::vector<RelearnTypes::position_type>>& positions) {
        RelearnException::check(positions.size() >= get_size(), "MultiPositionAxons::set_bouton_positions: positions.size() < get_size() is not supported");

        auto current_position = std::size_t{ 0 };

        number_axonal_boutons.clear();
        number_axonal_boutons_cumulative.clear();
        bouton_positions_flat.clear();

        number_axonal_boutons.reserve(positions.size());
        number_axonal_boutons_cumulative.reserve(positions.size());
        bouton_positions_flat.reserve(positions.size() * 32);

        for (auto neuron_id = number_neurons_type{ 0 }; neuron_id < positions.size(); ++neuron_id) {
            const auto& pos = positions[neuron_id];

            number_axonal_boutons.push_back(pos.size());
            number_axonal_boutons_cumulative.push_back(current_position);
            current_position += number_axonal_boutons[neuron_id];

            for (const auto& position : pos) {
                bouton_positions_flat.push_back(position);
            }
        }
    }

    /**
     * @brief Returns the number of axonal boutons for the specified neuron.
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The number of boutons
     */
    [[nodiscard]] std::size_t get_number_boutons(number_neurons_type neuron_id) const override;

    /**
     * @brief Returns for a specified neuron the position of the specified axonal bouton.
     * @param neuron_id The neuron's id
     * @param bouton_id The bouton's id
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The position of the axonal bouton
     */
    [[nodiscard]] RelearnTypes::position_type get_bouton_position(number_neurons_type neuron_id, std::size_t bouton_id) const override;

    /**
     * @brief Returns for a specified neuron the position of an axonal bouton.
     *      Can use a random generator to determine which bouton's position is returned.
     * @param neuron_id The neuron's id
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The position of the axonal bouton
     */
    [[nodiscard]] RelearnTypes::position_type get_bouton_position(number_neurons_type neuron_id) const override;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        Axons::record_memory_footprint(footprint);

        constexpr auto my_size = sizeof(*this) - sizeof(Axons);

        const auto number_axonal_boutons_size = bouton_positions_flat.size() * sizeof(std::vector<std::size_t>);
        const auto number_axonal_boutons_cumulative_size = bouton_positions_flat.size() * sizeof(std::vector<std::size_t>);
        const auto bouton_positions_flat_size = bouton_positions_flat.size() * sizeof(std::vector<RelearnTypes::position_type>);

        footprint->emplace("MultiPositionAxons", my_size + number_axonal_boutons_size + number_axonal_boutons_cumulative_size + bouton_positions_flat_size);
    }

private:
    std::vector<std::size_t> number_axonal_boutons;
    std::vector<std::size_t> number_axonal_boutons_cumulative;
    std::vector<RelearnTypes::position_type> bouton_positions_flat;
};
