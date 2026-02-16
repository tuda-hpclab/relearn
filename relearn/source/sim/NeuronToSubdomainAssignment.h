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

#include "neurons/enums/SynapticElementType.h"
#include "sim/LoadedNeuron.h"
#include "util/NeuronFilePaths.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

class Essentials;
class LocalGroupTranslator;
class SynapseLoader;
class Partition;

/**
 * This class provides an interface for every algorithm that is used to assign neurons to MPI processes
 */
class NeuronToSubdomainAssignment {
public:
    using position_type = RelearnTypes::position_type;
    using box_size_type = RelearnTypes::box_size_type;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new object with the given partition
     * @param _partition The partition to use
     */
    explicit NeuronToSubdomainAssignment(std::shared_ptr<Partition> _partition)
        : partition(std::move(_partition)) {
    }

    virtual ~NeuronToSubdomainAssignment() = default;

    NeuronToSubdomainAssignment(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment(NeuronToSubdomainAssignment&& other) = delete;

    NeuronToSubdomainAssignment& operator=(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment& operator=(NeuronToSubdomainAssignment&& other) = delete;

    /**
     * @brief Initializes the assignment class, i.e., loads all neurons for the subdomains
     * @exception Can throw a RelearnException
     */
    void initialize();

    /**
     * @brief Prints relevant metrics to the essentials
     * @param essentials The essentials
     */
    virtual void print_essentials(const std::unique_ptr<Essentials>& essentials) = 0;

    void initialize_groups_and_local_group_translator(std::optional<std::filesystem::path> opt_file_path = std::nullopt);

    /**
     * @brief Returns the associated SynapseLoader (some type that inherits from SynapseLoader)
     * @exception Throws a RelearnException if synapse_loader is nullptr
     * @return The associated SynapseLoader
     */
    std::shared_ptr<SynapseLoader> get_synapse_loader() const {
        RelearnException::check(synapse_loader != nullptr, "NeuronToSubdomainAssignment::get_synapse_loader: synapse_loader is empty");
        return synapse_loader;
    }

    /**
     * @brief Returns a function object that is used to fix calculated subdomain boundaries.
     *      This might be necessary if special boundaries must be considered
     * @return A function object that corrects subdomain boundaries
     */
    virtual std::function<box_size_type(box_size_type)> get_subdomain_boundary_fix() const {
        return [](Vec3d arg) { return arg; };
    }

    [[nodiscard]] std::size_t get_number_local_neurons() const {
        return number_local_neurons;
    }

    /**
     * @brief Returns the total number of neurons that should be placed
     * @return The total number of neurons that should be placed
     */
    [[nodiscard]] number_neurons_type get_requested_number_neurons() const noexcept {
        return requested_number_neurons;
    }

    /**
     * @brief Returns the current number of placed neurons on this MPI rank
     * @return The current number of placed neurons
     */
    [[nodiscard]] number_neurons_type get_number_placed_neurons() const noexcept {
        return number_placed_neurons;
    }

    /**
     * @brief Returns the total number of placed neurons across all MPI ranks
     * @return The total number of placed neurons across all MPI ranks
     */
    [[nodiscard]] number_neurons_type get_total_number_placed_neurons() const noexcept {
        return total_number_neurons;
    }

    /**
     * @brief Returns the total fraction of excitatory neurons that should be placed
     * @return The total fraction of excitatory neurons that should be placed
     */
    [[nodiscard]] double get_requested_ratio_excitatory_neurons() const noexcept {
        return requested_ratio_excitatory_neurons;
    }

    /**
     * @brief Returns the current fraction of placed excitatory neurons in the local subdomains
     * @return The total current fraction of placed excitatory neurons in the local subdomains
     */
    [[nodiscard]] double get_ratio_placed_excitatory_neurons() const noexcept {
        return ratio_placed_excitatory_neurons;
    }

    /**
     * @brief Returns the total number of neurons in the local subdomains
     * @return The total number of neurons in the local subdomains
     */
    [[nodiscard]] number_neurons_type get_number_neurons_in_subdomains() const noexcept {
        const auto total_number_neurons_in_subdomains = loaded_neurons.size();
        return total_number_neurons_in_subdomains;
    }

    /**
     * @brief Returns all positions of neurons in the local subdomains, indexed by the neuron id
     * @return The all position of neurons in the local subdomains
     */
    [[nodiscard]] std::vector<position_type> get_neuron_positions_in_subdomains() const {
        auto positions = std::vector<position_type>{};
        positions.reserve(loaded_neurons.size());

        for (const auto& loaded_neuron : loaded_neurons) {
            positions.push_back(loaded_neuron.pos);
        }

        return positions;
    }

    /**
     * @brief Returns all signal types of neurons in the local subdomains, indexed by the neuron id
     * @return The all position of neurons in the local subdomains
     */
    [[nodiscard]] std::vector<SignalType> get_neuron_types_in_subdomains() const {
        auto types = std::vector<SignalType>{};
        types.reserve(loaded_neurons.size());

        for (const auto& loaded_neuron : loaded_neurons) {
            types.push_back(loaded_neuron.signal_type);
        }

        return types;
    }

    /**
     * @brief Returns the group translator that translates between the local group id on the current mpi rank and its group name
     * @return the local group translator for this mpi rank
     */
    [[nodiscard]] std::shared_ptr<LocalGroupTranslator> get_local_group_translator() const noexcept {
        return local_group_translator;
    }

    /**
     * @brief Writes positions and signal types of all loaded neurons into the specified file.
     *      The format is
     *      # ID, Position (x y z), type
     * @param file_path The filepath where to write the positions and signals of the neurons
     * @exception Might throw a RelearnException
     */
    virtual void write_neuron_positions_and_signals_to_file(const std::filesystem::path& file_path) const;

    /**
     * @brief Writes groups of all loaded neurons into the specified file.
     * @param file_path The file path where to write the groups of the neurons
     */
    virtual void write_neuron_groups_to_file(const std::filesystem::path& file_path) const;

    /**
     * @brief Writes all loaded neurons into the specified files.
     * @param paths The object containing the paths to positions and groups to write in
     */
    virtual void write_neurons_to_files(const NeuronFilePaths& paths) const;

protected:
    virtual void fill_all_subdomains() = 0;

    void set_requested_ratio_excitatory_neurons(const double desired_frac_neurons_exc) noexcept {
        requested_ratio_excitatory_neurons = desired_frac_neurons_exc;
    }

    void set_number_local_neurons(const std::size_t _number_local_neurons) noexcept {
        number_local_neurons = _number_local_neurons;
    }

    void set_requested_number_neurons(const std::size_t get_requested_number_neurons) noexcept {
        requested_number_neurons = get_requested_number_neurons;
    }

    void set_ratio_placed_excitatory_neurons(const double current_frac_neurons_exc) noexcept {
        ratio_placed_excitatory_neurons = current_frac_neurons_exc;
    }

    void set_number_placed_neurons(const std::size_t current_num_neurons) noexcept {
        number_placed_neurons = current_num_neurons;
    }

    void set_total_number_placed_neurons(const std::size_t total_number_placed_neurons) const {
        total_number_neurons = total_number_placed_neurons;
    }

    void set_loaded_nodes(std::vector<LoadedNeuron>&& neurons) {
        loaded_neurons = std::move(neurons);
    }

    std::shared_ptr<Partition> partition{};

    std::shared_ptr<SynapseLoader> synapse_loader{};

    bool initialized{ false };

    std::vector<LoadedNeuron> loaded_neurons{};

private:
    double requested_ratio_excitatory_neurons{ 0.0 };
    number_neurons_type requested_number_neurons{ 0 };

    double ratio_placed_excitatory_neurons{ 0.0 };
    number_neurons_type number_placed_neurons{ 0 };

    std::shared_ptr<LocalGroupTranslator> local_group_translator;

    mutable number_neurons_type total_number_neurons{ Constants::uninitialized };

    std::size_t number_local_neurons{};
};
