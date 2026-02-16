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

#include "structure/SpaceFillingCurve.h"
#include "structure/SpaceFillingCurveType.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIRank.h"

#include <functional>
#include <memory>
#include <vector>

/**
 * This class provides all kinds of functionality that deals with the the local portion of neurons on the current MPI rank.
 * The local neurons are divided into Subdomains, from which each MPI rank has 1, 2, or 4
 */
class Partition {
public:
    using position_type = RelearnTypes::position_type;
    using box_size_type = RelearnTypes::box_size_type;
    using bounding_box_type = RelearnTypes::bounding_box_type;

    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * Subdomain is a type that represents one part of the octree at the level of the branching nodes.
     * It's composed of the min and max positions of the subdomain, the number of neurons in this subdomain,
     * the start and end local neuron ids, and its 1d and 3d index for all Subdomains.
     */
    struct Subdomain {
        bounding_box_type subdomain_size{};

        number_neurons_type number_neurons{ Constants::uninitialized };

        NeuronID neuron_local_id_start{ NeuronID::uninitialized_id() };
        NeuronID neuron_local_id_end{ NeuronID::uninitialized_id() };

        std::size_t index_1d{ Constants::uninitialized };
        Vec3s index_3d{ Constants::uninitialized };
    };

    /**
     * @brief Constructs a new object and uses the number of MPI ranks and the current MPI rank as foundation for the calculations
     * @param num_ranks The number of MPI ranks, must be of the form 2^k
     * @param my_rank The current MPI rank, must be initialized
     * @param curve_type The type of the space filling curve to use
     * @exception Throws a RelearnException if my_rank is not initialized or if the number of MPI ranks is not of the form 2^k
     */
    Partition(std::size_t num_ranks, mpiPP::MPIRank my_rank, SpaceFillingCurveType curve_type = SpaceFillingCurveType::Morton);

    ~Partition() = default;

    Partition(const Partition& other) = delete;
    Partition(Partition&& other) = default;

    Partition& operator=(const Partition& other) = delete;
    Partition& operator=(Partition&& other) = default;

    /**
     * @brief Prints the current local_subdomains as messages on the rank
     */
    void print_my_subdomains_info_rank();

    /**
     * @brief Sets the total number of neurons
     * @param total_num The total number of neurons
     */
    void set_total_number_neurons(const number_neurons_type total_num) noexcept {
        total_number_neurons = total_num;
    }

    /**
     * @brief Returns the total number of neurons
     * @exception Throws a RelearnException if the number has not been set previously
     * @return The total number of neurons
     */
    [[nodiscard]] number_neurons_type get_total_number_neurons() const {
        RelearnException::check(total_number_neurons < Constants::uninitialized, "Partition::get_total_number_neurons: total_number_neurons was not set");
        return total_number_neurons;
    }

    /**
     * @brief Returns the number of local neurons
     * @exception Throws a RelearnException if the calculate_local_ids has not been called
     * @return The number of local neurons
     */
    [[nodiscard]] number_neurons_type get_number_local_neurons() const {
        RelearnException::check(number_local_neurons < Constants::uninitialized, "Partition::get_number_local_neurons: Neurons are not loaded yet");
        return number_local_neurons;
    }

    /**
     * @brief Returns the total number of local_subdomains
     * @return The total number of local_subdomains
     */
    [[nodiscard]] std::size_t get_total_number_subdomains() const noexcept {
        return total_number_subdomains;
    }

    /**
     * @brief Returns the number of local_subdomains per dimension (total number of local_subdomains)^(1/3)
     * @return The number of local_subdomains per dimension
     */
    [[nodiscard]] std::size_t get_number_subdomains_per_dimension() const noexcept {
        return number_subdomains_per_dimension;
    }

    /**
     * @brief Returns the level in the octree on which the local_subdomains start
     * @return The level in the octree on which the local_subdomains start
     */
    [[nodiscard]] std::uint16_t get_level_of_subdomain_trees() const noexcept {
        return level_of_subdomain_trees;
    }

    /**
     * @brief Returns the number of local subdomains
     * @return The number of local subdomains (1, 2, or 4)
     */
    [[nodiscard]] std::size_t get_number_local_subdomains() const noexcept {
        return number_local_subdomains;
    }

    /**
     * @brief Returns the first id of the local subdomains in the global setting
     * @return The first id of the local local_subdomains in the global setting
     */
    [[nodiscard]] std::size_t get_local_subdomain_id_start() const noexcept {
        return local_subdomain_id_start;
    }

    /**
     * @brief Returns the last id of the local subdomains in the global setting
     * @return The last id of the local local_subdomains in the global setting
     */
    [[nodiscard]] std::size_t get_local_subdomain_id_end() const noexcept {
        return local_subdomain_id_end;
    }

    /**
     * @brief Returns the MPI rank id that was passed in the constructor
     * @return The MPI rank id
     */
    [[nodiscard]] mpiPP::MPIRank get_my_mpi_rank() const noexcept {
        return my_mpi_rank;
    }

    /**
     * @brief Returns the number of MPI ranks that was passed in the constructor
     * @return The number of MPI ranks
     */
    [[nodiscard]] std::size_t get_number_mpi_ranks() const noexcept {
        return number_mpi_ranks;
    }

    /**
     * @brief Returns the mpi rank that is responsible for the position
     * @param position The position which shall be resolved
     * @exception Throws a RelearnException if the calculate_local_ids has not been called
     * @return Returns the MPI rank that is responsible for the position
     */
    [[nodiscard]] int get_mpi_rank_from_position(const position_type& position) const {
        const auto& [simulation_box_min, simulation_box_max] = simulation_box;

        const auto half_constant = static_cast<double>(Constants::uninitialized) / 2;

        RelearnException::check(simulation_box_min.get_x() < half_constant, "Partition::get_mpi_rank_from_position: Neurons are not loaded yet");
        RelearnException::check(simulation_box_min.get_y() < half_constant, "Partition::get_mpi_rank_from_position: Neurons are not loaded yet");
        RelearnException::check(simulation_box_min.get_z() < half_constant, "Partition::get_mpi_rank_from_position: Neurons are not loaded yet");

        const auto& relative_position = position - simulation_box_min;
        const auto& simulation_box_length = simulation_box_max - simulation_box_min;

        const auto subdomain_length = simulation_box_length / static_cast<double>(number_subdomains_per_dimension);

        const auto subdomain_3d = box_size_type{ relative_position.get_x() / subdomain_length.get_x(), relative_position.get_y() / subdomain_length.get_y(), relative_position.get_z() / subdomain_length.get_z() };
        const auto id_3d = subdomain_3d.floor_componentwise();
        const auto id_1d = space_curve->map_3d_to_1d(id_3d);

        const auto rank = id_1d / number_local_subdomains;

        const auto cast_rank = static_cast<int>(rank);

        return cast_rank;
    }

    /**
     * @brief Returns the flattened index of the subdomain in the global setting
     * @param local_subdomain_index The local subdomain index
     * @exception Throws a RelearnException if local_subdomain_index is larger or equal to the number of local subdomains
     * @return The flattened index of the subdomain in the local index
     */
    [[nodiscard]] std::size_t get_1d_index_of_subdomain(const std::size_t local_subdomain_index) const {
        RelearnException::check(local_subdomain_index < local_subdomains.size(),
                                "Partition::get_1d_index_of_subdomain: index ({}) was too large for the number of local subdomains ({})", local_subdomain_index, local_subdomains.size());
        return local_subdomains[local_subdomain_index].index_1d;
    }

    /**
     * @brief Returns the 3-dimensional index of the subdomain in the global setting
     * @param local_subdomain_index The local subdomain index
     * @exception Throws a RelearnException if local_subdomain_index is larger or equal to the number of local subdomains
     * @return The 3-dimensional of the subdomain in the global setting
     */
    [[nodiscard]] Vec3s get_3d_index_of_subdomain(const std::size_t local_subdomain_index) const {
        RelearnException::check(local_subdomain_index < local_subdomains.size(),
                                "Partition::get_3d_index_of_subdomain: index ({}) was too large for the number of local subdomains ({})", local_subdomain_index, local_subdomains.size());
        return local_subdomains[local_subdomain_index].index_3d;
    }

    /**
     * @brief Sets the number of local neurons (i.e., on this MPI rank)
     * @param number_neurons The number of local neurons
     */
    void set_number_local_neurons(const number_neurons_type number_neurons) noexcept {
        number_local_neurons = number_neurons;
    }

    /**
     * @brief Sets the boundaries of the simulation box
     * @param simulation_box_size The size of the whole simulation
     * @exception Throws a RelearnException if any value is outside of [-Constants::uninitialized, Constants::uninitialized]
     */
    void set_simulation_box_size(const bounding_box_type& simulation_box_size);

    /**
     * @brief Calculates the subdomain boundaries for all local subdomains and sets them accordingly
     */
    void calculate_and_set_subdomain_boundaries() {
        for (auto& subdomain : local_subdomains) {
            const auto& [min, max] = calculate_subdomain_boundaries(subdomain.index_1d);
            subdomain.subdomain_size = { min, max };
        }
    }

    /**
     * @brief Calculates the boundaries of the subdomain
     * @param subdomain_index_1d The flattened index of the subdomain
     * @return (minimum, maximum) of the subdomain
     */
    [[nodiscard]] bounding_box_type calculate_subdomain_boundaries(const std::size_t subdomain_index_1d) const {
        return calculate_subdomain_boundaries(space_curve->map_1d_to_3d(subdomain_index_1d));
    }

    /**
     * @brief Calculates the boundaries of the subdomain
     * @param subdomain_index_3d The 3-dimensional index of the subdomain
     * @return (minimum, maximum) of the subdomain
     */
    [[nodiscard]] bounding_box_type calculate_subdomain_boundaries(const Vec3s& subdomain_index_3d) const {
        const auto& [requested_subdomain_x, requested_subdomain_y, requested_subdomain_z] = subdomain_index_3d;

        const auto& [sim_box_min, sim_box_max] = get_simulation_box_size();
        const auto& simulation_box_length = (sim_box_max - sim_box_min);

        const auto& subdomain_length = simulation_box_length / static_cast<box_size_type::value_type>(number_subdomains_per_dimension);

        const auto& [subdomain_length_x, subdomain_length_y, subdomain_length_z] = subdomain_length;

        const auto min = box_size_type{
            static_cast<double>(requested_subdomain_x) * subdomain_length_x,
            static_cast<double>(requested_subdomain_y) * subdomain_length_y,
            static_cast<double>(requested_subdomain_z) * subdomain_length_z
        };

        const auto next_x = static_cast<box_size_type::value_type>(requested_subdomain_x + 1) * subdomain_length_x;
        const auto next_y = static_cast<box_size_type::value_type>(requested_subdomain_y + 1) * subdomain_length_y;
        const auto next_z = static_cast<box_size_type::value_type>(requested_subdomain_z + 1) * subdomain_length_z;

        const auto max = box_size_type{ next_x, next_y, next_z };

        const auto& [simulation_box_min, _] = simulation_box;

        const auto adjusted_min = min + simulation_box_min;
        const auto adjusted_max = max + simulation_box_min;

        auto corrected_min = boundary_corrector(adjusted_min);
        auto corrected_max = boundary_corrector(adjusted_max);

        return { corrected_min, corrected_max };
    }

    /**
     * @brief Returns the boundaries of the subdomain
     * @param local_subdomain_index The local index of the subdomain
     * @return (minimum, maximum) of the subdomain
     */
    [[nodiscard]] bounding_box_type get_subdomain_boundaries(const std::size_t local_subdomain_index) const {
        RelearnException::check(local_subdomain_index < local_subdomains.size(),
                                "Partition::get_subdomain_boundaries: index ({}) was too large for the number of local subdomains ({})", local_subdomain_index, local_subdomains.size());
        return { local_subdomains[local_subdomain_index].subdomain_size };
    }

    /**
     * @brief Returns the boundaries of the local subdomains
     * @return The boundaries encoded in boundary boxes
     */
    [[nodiscard]] std::vector<bounding_box_type> get_all_local_subdomain_boundaries() const;

    /**
     * @brief Returns the size of the simulation box
     * @return The size of the simulation box as boundary box
     */
    [[nodiscard]] bounding_box_type get_simulation_box_size() const {
        const auto& simulation_box_minimum = simulation_box.get_minimum();

        RelearnException::check(simulation_box_minimum.get_x() < Constants::uninitialized / 2, "Partition::get_simulation_box_size: set_simulation_box_size was not called before"); // NOLINT(bugprone-integer-division)
        RelearnException::check(simulation_box_minimum.get_y() < Constants::uninitialized / 2, "Partition::get_simulation_box_size: set_simulation_box_size was not called before"); // NOLINT(bugprone-integer-division)
        RelearnException::check(simulation_box_minimum.get_z() < Constants::uninitialized / 2, "Partition::get_simulation_box_size: set_simulation_box_size was not called before"); // NOLINT(bugprone-integer-division)

        return simulation_box;
    }

    /**
     * @brief Sets the correction function for the boundaries of the subdomains
     * @param corrector The correction function
     * @exception Throws a RelearnException if corrector is not valid
     */
    void set_boundary_correction_function(std::function<box_size_type(box_size_type)> corrector) {
        RelearnException::check(corrector != nullptr, "Partition::set_boundary_correction_function: corrector was empty");
        boundary_corrector = std::move(corrector);
    }

    /**
     * @brief Returns the currently used space-filling curve (can be passed to the Octree)
     * @return The space-filling curve
     */
    [[nodiscard]] const std::shared_ptr<SpaceFillingCurve>& get_space_filling_curve() const noexcept {
        return space_curve;
    }

    /**
     * @brief Records the memory footprint as "Partition" of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) const {
        const auto my_footprint = sizeof(*this)
                                  + (local_subdomains.capacity() * sizeof(Subdomain));

        footprint->emplace("Partition", my_footprint);
    }

private:
    mpiPP::MPIRank my_mpi_rank{ mpiPP::MPIRank::root_rank() };
    std::size_t number_mpi_ranks{ Constants::uninitialized };

    number_neurons_type total_number_neurons{ Constants::uninitialized };
    number_neurons_type number_local_neurons{ Constants::uninitialized };

    std::size_t total_number_subdomains{ Constants::uninitialized };
    std::size_t number_subdomains_per_dimension{ Constants::uninitialized };
    std::uint16_t level_of_subdomain_trees{ std::numeric_limits<std::uint16_t>::max() };

    std::size_t number_local_subdomains{ Constants::uninitialized };
    std::size_t local_subdomain_id_start{ Constants::uninitialized };
    std::size_t local_subdomain_id_end{ Constants::uninitialized };

    bounding_box_type simulation_box{};

    std::vector<Subdomain> local_subdomains{};
    std::shared_ptr<SpaceFillingCurve> space_curve{};

    std::function<box_size_type(box_size_type)> boundary_corrector{
        [](box_size_type bst) { return bst; }
    };
};
