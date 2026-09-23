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

#include "Config.h"

#include "sim/LoadedNeuron.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <cpp-utility/Cast.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

class Partition;

/**
 * This class provides the functionality to place neurons within small boxes, i.e.,
 * it divides the simulation space into small boxes of a given side length and places one or none neuron inside each box.
 * The position within the box is drawn uniformly at random.
 */
class BoxBasedRandomSubdomainAssignment : public NeuronToSubdomainAssignment {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Construct a new object with the given parameter
     * @param _partition The partition, not nullptr
     * @param fraction_excitatory_neurons The fraction of excitatory neurons, must be from [0.0, 1.0]
     * @param um_per_neuron The box side length in micrometer
     */
    BoxBasedRandomSubdomainAssignment(std::shared_ptr<Partition> _partition, const percentage_type fraction_excitatory_neurons, const space_type um_per_neuron)
        : NeuronToSubdomainAssignment(std::move(_partition))
        , um_per_neuron_(um_per_neuron) {
        RelearnException::check(partition != nullptr, "BoxBasedRandomSubdomainAssignment::BoxBasedRandomSubdomainAssignment: partition was nullptr");
        RelearnException::check(fraction_excitatory_neurons >= percentage_type{ 0 } && fraction_excitatory_neurons <= percentage_type{ 1 },
                                "BoxBasedRandomSubdomainAssignment::BoxBasedRandomSubdomainAssignment: The requested fraction of excitatory neurons is not in [0.0, 1.0]: {}", fraction_excitatory_neurons);
        RelearnException::check(um_per_neuron > space_type{ 0 }, "BoxBasedRandomSubdomainAssignment::BoxBasedRandomSubdomainAssignment: The requested um per neuron is <= 0.0: {}", um_per_neuron);

        set_requested_ratio_excitatory_neurons(fraction_excitatory_neurons);
    }

    BoxBasedRandomSubdomainAssignment(const BoxBasedRandomSubdomainAssignment& other) = delete;
    BoxBasedRandomSubdomainAssignment(BoxBasedRandomSubdomainAssignment&& other) = delete;

    BoxBasedRandomSubdomainAssignment& operator=(const BoxBasedRandomSubdomainAssignment& other) = delete;
    BoxBasedRandomSubdomainAssignment& operator=(BoxBasedRandomSubdomainAssignment&& other) = delete;

    ~BoxBasedRandomSubdomainAssignment() override = default;

    /**
     * @brief Returns a function object that is used to fix calculated subdomain boundaries.
     *      It rounds the boundaries up to the next multiple of um_per_neuron
     * @return A function object that corrects subdomain boundaries
     */
    [[nodiscard]] std::function<position_type(position_type)> get_subdomain_boundary_fix() const override {
        auto lambda = [multiple = um_per_neuron_](position_type arg) -> position_type {
            // A boundary that the partition derived from the simulation box sits on a multiple of the box length
            // only up to the resolution of space_type. Vec3::round_to_larger_multiple decides that with an
            // absolute epsilon of 1e-5, which is finer than that resolution at the size of a simulation box once
            // space_type is float, so such a boundary would be pushed a whole box past the simulation box.
            const auto round_up = [multiple](const space_type component) {
                const auto number_boxes = static_cast<double>(component) / static_cast<double>(multiple);
                const auto nearest = std::round(number_boxes);

                const auto resolution = std::abs(nearest) * double{ 128 } * utility::cast<double>(std::numeric_limits<space_type>::epsilon());
                const auto tolerance = std::max(resolution, Constants::eps / static_cast<double>(multiple));

                const auto corrected = (std::abs(number_boxes - nearest) <= tolerance) ? nearest : std::ceil(number_boxes);
                return utility::cast<space_type>(corrected * static_cast<double>(multiple));
            };

            return position_type{ round_up(arg.get_x()), round_up(arg.get_y()), round_up(arg.get_z()) };
        };

        return lambda;
    }

    /**
     * @brief Returns the micrometer per neuron box
     * @return The micrometer per neuron box
     */
    [[nodiscard]] space_type get_um_per_neuron() const noexcept {
        return um_per_neuron_;
    }

    /**
     * @brief Places a given number of neurons within a box
     * @param offset The offset to place to neurons into the box
     * @param length_of_box The length of the box, must be positive in each component
     * @param number_neurons The number of neurons to place
     * @param first_id The starting ID of the placed neurons
     * @return The placed neurons and the number of exhitatory neurons
     */
    std::pair<std::vector<LoadedNeuron>, number_neurons_type> place_neurons_in_box(const position_type& offset, const position_type& length_of_box,
                                                                                   number_neurons_type number_neurons, NeuronID::value_type first_id);

private:
    const space_type um_per_neuron_{}; // Micrometer per neuron in one dimension
};
