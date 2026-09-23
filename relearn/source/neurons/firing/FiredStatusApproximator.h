#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

/**
 * @brief This class approximates the fired status of distant neurons by their frequency.
 * Local neurons are handled in a precise manner
 */
class FiredStatusApproximator : public FiredStatusCommunicator {
public:
    using fire_rate_type = RelearnTypes::fire_rate_type;

    /**
     * @brief Approximates the firing rate of distant neurons by a constant frequency
     * @param num_ranks The number of ranks
     */
    explicit FiredStatusApproximator(const mpiPP::MPIRank _my_rank, const int num_ranks)
        : FiredStatusCommunicator(_my_rank, num_ranks)
        , firing_rate_cache(static_cast<std::size_t>(num_ranks)) {
        RelearnException::check(num_ranks > 0, "FiredStatusApproximator::FiredStatusApproximator: num_ranks is too small: {}", num_ranks);
    }

    /**
     * @brief Initializes this instance to hold the given number of neurons
     * @param number_neurons The number of neurons for this instance, must be > 0
     * @exception Throws a RelearnException if number_neurons == 0
     */
    void init(const number_neurons_type number_neurons) override {
        FiredStatusCommunicator::init(number_neurons);

        accumulated_fired.resize(number_neurons, 0);
        latest_firing_rate.resize(number_neurons, 0.0);
    }

    /**
     * @brief Additionally created the given number of neurons
     * @param creation_count The number of neurons to create, must be > 0
     * @exception Throws a RelearnException if creation_count == 0 or if init(...) was not called before
     */
    void create_neurons(const number_neurons_type creation_count) override {
        const auto old_size = get_number_local_neurons();

        FiredStatusCommunicator::create_neurons(creation_count);

        const auto new_size = old_size + creation_count;

        accumulated_fired.resize(new_size, 0);
        latest_firing_rate.resize(new_size, 0.0);
    }

    /**
     * @brief Registers the fired status of the local neurons that are not disabled.
     * @param step The current update step
     * @exception Can throw a RelearnException
     */
    void commit_local_fired_status(step_type step) override;

    /**
     * @brief Exchanges the fired status with all MPI ranks
     * @param step The current update step
     * @exception Can throw a RelearnException
     */
    void exchange_fired_status(step_type step) override;

    /**
     * @brief Checks if the communicator contains the specified neuron of the rank,
     *      i.e., whether that neuron fired in the last update step.
     * @param rank The MPI rank that owns the neuron
     * @param neuron_id The neuron in question
     * @exception Throws a RelearnException if rank is not from [0, number_ranks) or the neuron_id is virtual
     */
    [[nodiscard]] bool contains(mpiPP::MPIRank rank, NeuronID neuron_id) const override;

    /**
     * @brief Recalculate the cached firing rates for the distant neurons
     * @param step The current simulation step
     */
    void notify_of_plasticity_change(step_type step) override;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override;

    void wait_for_exchange_to_finish() override { }

private:
    std::vector<std::size_t> accumulated_fired{};
    std::vector<fire_rate_type> latest_firing_rate{};
    std::vector<std::unordered_map<NeuronID, fire_rate_type>> firing_rate_cache{};

    step_type last_synced{ 0 };
};
