/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Partition.h"

#include "Config.h"
#include "io/LogFiles.h"
#include "structure/Hilbert.h"
#include "structure/Morton.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIRank.h"

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

Partition::Partition(const std::size_t num_ranks, const mpiPP::MPIRank my_rank, const SpaceFillingCurveType curve_type)
    : my_mpi_rank{ my_rank }
    , number_mpi_ranks{ num_ranks } {
    RelearnException::check(num_ranks > 0, "Partition::Partition: Number of MPI ranks must be a positive number: {}", num_ranks);
    RelearnException::check(num_ranks > my_rank.get_rank_cast(), "Partition::Partition: My rank must be smaller than number of ranks: {} vs {}", num_ranks, my_rank);

    /**
     * Total number of local_subdomains is smallest power of 8 that is >= num_ranks.
     * We choose power of 8 as every domain subdivision creates 8 local_subdomains (in 3d).
     */
    const auto smallest_exponent = std::ceil(std::log(num_ranks) / std::log(8.0));
    level_of_subdomain_trees = static_cast<std::uint16_t>(smallest_exponent);
    total_number_subdomains = 1ULL << (3U * level_of_subdomain_trees); // 8^level_of_subdomain_trees

    // Every rank should get at least one subdomain
    RelearnException::check(total_number_subdomains >= num_ranks, "Partition::Partition: Total num local_subdomains is smaller than number ranks: {} vs {}", total_number_subdomains, num_ranks);

    /**
     * Calc my number of local_subdomains
     *
     * NOTE:
     * Every rank gets the same number of local_subdomains first.
     * The remaining m local_subdomains are then assigned to the first m ranks,
     * one subdomain more per rank.
     *
     * For #procs = 2^n and 8^level_of_subdomain_trees local_subdomains, every proc's #local_subdomains is the same power of two of {1, 2, 4}.
     */
    // NOLINTNEXTLINE
    number_local_subdomains = total_number_subdomains / num_ranks;
    const auto rest = total_number_subdomains % num_ranks;
    number_local_subdomains += (my_rank.get_rank_cast() < rest) ? 1U : 0U;

    if (rest != 0) {
        LogFiles::print_message_rank(mpiPP::MPIRank::uninitialized_rank(), "My rank is: {}; There are {} ranks in total; The rest is: {}", my_rank, num_ranks, rest);
        RelearnException::fail("Partition::Partition: Number of ranks must be of the form 2^n but was {}", num_ranks);
    }

    /**
     * Set parameter of space filling curve before it can be used.
     * total_number_subdomains = 8^level_of_subdomain_trees = (2^3)^level_of_subdomain_trees = 2^(3*level_of_subdomain_trees).
     * Thus, number of local_subdomains per dimension (3d) is (2^(3*level_of_subdomain_trees))^(1/3) = 2^level_of_subdomain_trees.
     */
    number_subdomains_per_dimension = 1ULL << level_of_subdomain_trees;

    if (level_of_subdomain_trees > std::numeric_limits<std::uint8_t>::max()) {
        RelearnException::fail("Partition::Partition: level_of_subdomain_trees was too large: {}", level_of_subdomain_trees);
    }

    if (curve_type == SpaceFillingCurveType::Morton) {
        space_curve = std::make_shared<Morton>(static_cast<std::uint8_t>(level_of_subdomain_trees));
    } else {
        RelearnException::check(curve_type == SpaceFillingCurveType::Hilbert, "Partition::Partition: curve_type was not Morton or Hilbert");
        space_curve = std::make_shared<Hilbert>(static_cast<std::uint8_t>(level_of_subdomain_trees));
    }

    // Calc start and end index of subdomain
    local_subdomain_id_start = (total_number_subdomains / num_ranks) * my_rank.get_rank_cast();
    local_subdomain_id_end = local_subdomain_id_start + number_local_subdomains - 1;

    // Allocate vector with my number of local_subdomains
    local_subdomains = std::vector<Subdomain>(number_local_subdomains);

    for (auto i = std::size_t{ 0 }; i < number_local_subdomains; i++) {
        auto& current_subdomain = local_subdomains[i];

        // Set space filling curve indices in 1d and 3d
        current_subdomain.index_1d = local_subdomain_id_start + i;
        current_subdomain.index_3d = space_curve->map_1d_to_3d(current_subdomain.index_1d);
    }

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Total number local_subdomains        : {}", total_number_subdomains);
    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Number subdomains per dimension: {}", number_subdomains_per_dimension);
}

void Partition::print_my_subdomains_info_rank() {
    auto sstream = std::stringstream{};

    sstream << "My number of neurons   : " << number_local_neurons << '\n';
    sstream << "My number of local_subdomains: " << number_local_subdomains << '\n';
    sstream << "My subdomain ids       : [ " << local_subdomain_id_start
            << " , "
            << local_subdomain_id_end
            << " ]"
            << '\n';

    for (auto i = std::size_t{ 0 }; i < number_local_subdomains; i++) {
        const auto& [minimum_position, maximum_position] = local_subdomains[i].subdomain_size;

        sstream << "Subdomain: " << i << '\n';
        sstream << "    number_neurons: " << local_subdomains[i].number_neurons << '\n';
        sstream << "    index_1d   : " << local_subdomains[i].index_1d << '\n';

        sstream << "    index_3d   : "
                << "( " << local_subdomains[i].index_3d.get_x()
                << " , " << local_subdomains[i].index_3d.get_y()
                << " , " << local_subdomains[i].index_3d.get_z()
                << " )"
                << '\n';

        sstream << "    minimum_position    : "
                << "( " << minimum_position.get_x()
                << " , " << minimum_position.get_y()
                << " , " << minimum_position.get_z()
                << " )"
                << '\n';

        sstream << "    maximum_position    : "
                << "( " << maximum_position.get_x()
                << " , " << maximum_position.get_y()
                << " , " << maximum_position.get_z()
                << " )\n";
    }

    LogFiles::write_to_file(LogFiles::EventType::Cout, false, sstream.str());
}

void Partition::set_simulation_box_size(const bounding_box_type& simulation_box_size) {
    const auto& [min, max] = simulation_box_size;
    const auto& [min_x, min_y, min_z] = min;
    const auto& [max_x, max_y, max_z] = max;

    const auto half_constant = static_cast<double>(Constants::uninitialized) / 2;

    RelearnException::check(-half_constant < min_x && min_x < half_constant, "Partition::set_simulation_box_size: minimum had a bad value for x: {}", min_x);
    RelearnException::check(-half_constant < min_y && min_y < half_constant, "Partition::set_simulation_box_size: minimum had a bad value for y: {}", min_y);
    RelearnException::check(-half_constant < min_z && min_z < half_constant, "Partition::set_simulation_box_size: minimum had a bad value for y: {}", min_z);

    RelearnException::check(-half_constant < max_x && max_x < half_constant, "Partition::set_simulation_box_size: maximum had a bad value for x: {}", max_x);
    RelearnException::check(-half_constant < max_y && max_y < half_constant, "Partition::set_simulation_box_size: maximum had a bad value for y: {}", max_y);
    RelearnException::check(-half_constant < max_z && max_z < half_constant, "Partition::set_simulation_box_size: maximum had a bad value for y: {}", max_z);

    simulation_box = simulation_box_size;

    const auto& simulation_box_length = max - min;
    const auto& subdomain_length = simulation_box_length / static_cast<double>(number_subdomains_per_dimension);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Simulation box length (height, width, depth)\t: ({}, {}, {})",
                                 simulation_box_length.get_x(), simulation_box_length.get_y(), simulation_box_length.get_z());
    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Subdomain length (height, width, depth)\t: ({}, {}, {})",
                                 subdomain_length.get_x(), subdomain_length.get_y(), subdomain_length.get_z());
}

std::vector<Partition::bounding_box_type> Partition::get_all_local_subdomain_boundaries() const {
    return local_subdomains
           | ranges::views::transform([](const auto& subdomain) { return subdomain.subdomain_size; })
           | ranges::to_vector;
}
