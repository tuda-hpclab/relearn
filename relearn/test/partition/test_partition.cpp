/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_partition.h"

#include "Config.h"

#include "structure/Morton.h"
#include "structure/Partition.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/MemoryFootprint.hpp>

#include <fmt/core.h>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPIRankRange.h>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/indices.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {
bool is_power_of_two(std::integral auto number) {
    auto counter = 0;

    while (number != 0) {
        const auto res = number & decltype(number){ 1 };
        if (res == 1) {
            counter++;
        }
        number >>= 1;
    }

    return counter == 1;
}
} // namespace

TEST_F(PartitionTest, testPartitionZeroRanks) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = MPIRankFactory::get_random_mpi_rank(mt);
    ASSERT_THROW_NO_PRINT_MSG(Partition part(0, my_rank), RelearnException, my_rank);
}

TEST_F(PartitionTest, testPartitionConstructorArguments) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto my_rank = MPIRankFactory::get_random_mpi_rank(mt);
    const auto num_ranks = MPIRankFactory::get_random_number_ranks(mt);
    const auto is_rank_power_2 = is_power_of_two(num_ranks);

    if (!is_rank_power_2) {
        ASSERT_THROW_NO_PRINT_MSG(Partition part(num_ranks, my_rank), RelearnException, fmt::format("{} {}", num_ranks, my_rank));
        return;
    }

    if (my_rank.get_rank() >= num_ranks) {
        ASSERT_THROW_NO_PRINT_MSG(Partition part(num_ranks, my_rank), RelearnException, fmt::format("{} {}", num_ranks, my_rank));
        return;
    }

    ASSERT_NO_THROW(Partition part(num_ranks, my_rank)) << num_ranks << " " << my_rank;
}

TEST_F(PartitionTest, testPartitionConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto num_ranks_cast = static_cast<std::size_t>(num_ranks);
    const auto num_subdomains = round_to_next_exponent(num_ranks_cast, 8);

    const auto my_subdomains = num_subdomains / num_ranks_cast;

    const auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));
    const auto num_subdomains_per_dim = static_cast<size_t>(std::ceil(std::pow(static_cast<double>(num_subdomains), 1.0 / 3.0)));

    for (const auto my_rank : mpiPP::MPIRankRange::range(num_ranks)) {
        const auto partition = Partition(num_ranks, my_rank);

        ASSERT_EQ(partition.get_number_mpi_ranks(), num_ranks);
        ASSERT_EQ(partition.get_my_mpi_rank(), my_rank);

        ASSERT_EQ(partition.get_total_number_subdomains(), num_subdomains) << num_subdomains;
        ASSERT_EQ(partition.get_number_local_subdomains(), my_subdomains) << my_subdomains;

        ASSERT_EQ(partition.get_local_subdomain_id_start(), my_subdomains * my_rank.get_rank_cast()) << my_subdomains << ' ' << my_rank;
        ASSERT_EQ(partition.get_local_subdomain_id_end(), my_subdomains * (my_rank.get_rank_cast() + 1) - 1) << my_subdomains << ' ' << my_rank;

        ASSERT_EQ(partition.get_level_of_subdomain_trees(), oct_exponent) << oct_exponent;
        ASSERT_EQ(partition.get_number_subdomains_per_dimension(), num_subdomains_per_dim) << num_subdomains_per_dim;
    }
}

TEST_F(PartitionTest, testPartitionNumberNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto num_ranks_cast = static_cast<std::size_t>(num_ranks);
    const auto num_subdomains = round_to_next_exponent(num_ranks_cast, 8);
    const auto my_subdomains = num_subdomains / num_ranks_cast;

    auto number_local_neurons = std::vector<std::vector<size_t>>(num_ranks_cast);
    auto number_total_neurons = std::size_t{ 0 };

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        const auto my_rank_cast = static_cast<std::size_t>(my_rank);
        number_local_neurons[my_rank_cast] = std::vector<size_t>(my_subdomains);

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            const auto num_local_neurons = NeuronIdFactory::get_random_number_neurons(mt);

            number_local_neurons[my_rank_cast][my_subdomain] = num_local_neurons;
            number_total_neurons += num_local_neurons;
        }
    }

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        const auto my_rank_cast = static_cast<std::size_t>(my_rank);
        const auto& local_neurons = number_local_neurons[my_rank_cast];

        auto local_ids_start = std::vector<NeuronID>(my_subdomains, NeuronID{ 0 });
        auto local_ids_ends = std::vector<NeuronID>(my_subdomains, NeuronID{ 0 });

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            if (my_subdomain > 0) {
                const auto local_start = local_ids_ends[static_cast<size_t>(my_subdomain) - 1].get_neuron_id() + 1;
                local_ids_start[my_subdomain] = NeuronID(false, local_start);
            }

            const auto local_end = local_ids_start[my_subdomain].get_neuron_id() + local_neurons[my_subdomain] - 1;
            local_ids_ends[my_subdomain] = NeuronID(false, local_end);
        }

        auto partition = Partition(num_ranks, mpiPP::MPIRank(my_rank));

        ASSERT_THROW_NO_PRINT(std::ignore = partition.get_total_number_neurons(), RelearnException);
        partition.set_total_number_neurons(number_total_neurons);
        ASSERT_EQ(partition.get_total_number_neurons(), number_total_neurons);

        const auto num_local_neurons = ranges::accumulate(local_neurons, size_t{ 0 });
        partition.set_number_local_neurons(num_local_neurons);

        ASSERT_EQ(partition.get_number_local_neurons(), num_local_neurons);
    }
}

TEST_F(PartitionTest, testPartitionSubdomainIndices) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto num_ranks_cast = static_cast<std::size_t>(num_ranks);
    const auto num_subdomains = round_to_next_exponent(num_ranks_cast, 8);
    const auto my_subdomains = num_subdomains / num_ranks_cast;

    const auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));

    if (oct_exponent >= std::numeric_limits<RelearnTypes::level_type>::max()) {
        return;
    }

    const auto oct_exponent_cast = static_cast<RelearnTypes::level_type>(oct_exponent);
    const auto sfc = Morton(oct_exponent_cast);

    auto found_indices = std::vector<bool>(num_subdomains, false);

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        const auto partition = Partition(num_ranks, mpiPP::MPIRank(my_rank));

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            const auto index_1 = partition.get_1d_index_of_subdomain(my_subdomain);
            const auto index_3 = partition.get_3d_index_of_subdomain(my_subdomain);

            const auto translated_index_3 = sfc.map_1d_to_3d(index_1);
            const auto translated_index_1 = sfc.map_3d_to_1d(index_3);

            ASSERT_EQ(index_1, translated_index_1) << index_1 << ' ' << translated_index_1;
            ASSERT_EQ(index_3, translated_index_3) << index_3 << ' ' << translated_index_3;

            ASSERT_FALSE(found_indices[index_1]) << index_1;
            ASSERT_TRUE(index_1 < num_subdomains) << index_1 << ' ' << num_subdomains;

            found_indices[index_1] = true;
        }

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            ASSERT_THROW_NO_PRINT(std::ignore = partition.get_1d_index_of_subdomain(my_subdomain + num_subdomains), RelearnException);
            ASSERT_THROW_NO_PRINT(std::ignore = partition.get_3d_index_of_subdomain(my_subdomain + num_subdomains), RelearnException);
        }
    }
}

TEST_F(PartitionTest, testPartitionSubdomainBoundaries) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto num_ranks_cast = static_cast<std::size_t>(num_ranks);
    const auto num_subdomains = round_to_next_exponent(num_ranks_cast, 8);
    const auto my_subdomains = num_subdomains / num_ranks_cast;

    const auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));
    if (oct_exponent >= std::numeric_limits<RelearnTypes::level_type>::max()) {
        return;
    }

    const auto oct_exponent_cast = static_cast<RelearnTypes::level_type>(oct_exponent);
    const auto num_subdomains_per_dim = utility::cast<RelearnTypes::space_type>(std::ceil(std::pow(static_cast<double>(num_subdomains), 1.0 / 3.0)));

    const auto sfc = Morton(oct_exponent_cast);

    const auto& [simulation_box_minimum, simulation_box_maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    const auto& simulation_box_dimensions = simulation_box_maximum - simulation_box_minimum;
    const auto& subdomain_box_dimensions = simulation_box_dimensions / num_subdomains_per_dim;

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        auto partition = Partition(num_ranks, mpiPP::MPIRank(my_rank));

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_maximum, simulation_box_minimum }), RelearnException);

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x() + static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() }, simulation_box_maximum }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x(), simulation_box_minimum.get_y() + static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_minimum.get_z() }, simulation_box_maximum }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x(), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() + static_cast<RelearnTypes::space_type>(Constants::uninitialized) }, simulation_box_maximum }), RelearnException);

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x() + static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() } }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y() + static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_maximum.get_z() } }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() + static_cast<RelearnTypes::space_type>(Constants::uninitialized) } }), RelearnException);

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x() - static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() }, simulation_box_maximum }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x(), simulation_box_minimum.get_y() - static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_minimum.get_z() }, simulation_box_maximum }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x(), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() - static_cast<RelearnTypes::space_type>(Constants::uninitialized) }, simulation_box_maximum }), RelearnException);

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x() - static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() } }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y() - static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_maximum.get_z() } }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() - static_cast<RelearnTypes::space_type>(Constants::uninitialized) } }), RelearnException);

        ASSERT_THROW_NO_PRINT(std::ignore = partition.get_simulation_box_size(), RelearnException);

        partition.set_simulation_box_size({ simulation_box_minimum, simulation_box_maximum });
        const auto& [retrieved_min, retrieved_max] = partition.get_simulation_box_size();

        ASSERT_EQ(simulation_box_minimum, retrieved_min);
        ASSERT_EQ(simulation_box_maximum, retrieved_max);
    }

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        auto partition = Partition(num_ranks, mpiPP::MPIRank(my_rank));
        partition.set_simulation_box_size({ simulation_box_minimum, simulation_box_maximum });

        for (auto subdomain_index_1 = std::size_t{ 0 }; subdomain_index_1 < num_subdomains; subdomain_index_1++) {
            const auto& subdomain_index_3 = sfc.map_1d_to_3d(subdomain_index_1);

            const auto& [min_1, max_1] = partition.calculate_subdomain_boundaries(subdomain_index_1);
            const auto& [min_3, max_3] = partition.calculate_subdomain_boundaries(subdomain_index_3);

            ASSERT_EQ(min_1, min_3) << min_1 << min_3;
            ASSERT_EQ(max_1, max_3) << max_1 << max_3;

            const auto subdomain_expected_min = RelearnTypes::position_type{
                subdomain_box_dimensions.get_x() * static_cast<RelearnTypes::space_type>(subdomain_index_3.get_x()),
                subdomain_box_dimensions.get_y() * static_cast<RelearnTypes::space_type>(subdomain_index_3.get_y()),
                subdomain_box_dimensions.get_z() * static_cast<RelearnTypes::space_type>(subdomain_index_3.get_z())
            } + simulation_box_minimum;

            const auto subdomain_expected_max = subdomain_expected_min + subdomain_box_dimensions;

            const auto& difference_min = min_1 - subdomain_expected_min;
            const auto& difference_max = max_1 - subdomain_expected_max;

            // The boundaries are derived from the simulation box by division and multiplication, so they agree
            // with the expectation above only to the resolution space_type has at that magnitude.
            ASSERT_NEAR(difference_min.calculate_p_norm(1.0), 0.0, tolerance_for<RelearnTypes::space_type>(subdomain_expected_min.calculate_p_norm(1.0))) << min_1 << subdomain_expected_min;
            ASSERT_NEAR(difference_max.calculate_p_norm(1.0), 0.0, tolerance_for<RelearnTypes::space_type>(subdomain_expected_max.calculate_p_norm(1.0))) << max_1 << subdomain_expected_max;
        }

        partition.calculate_and_set_subdomain_boundaries();

        auto local_subdomain_boundaries = std::vector<RelearnTypes::bounding_box_type>{};

        for (const auto my_subdomain : ranges::views::indices(my_subdomains)) {
            const auto& [min, max] = partition.get_subdomain_boundaries(my_subdomain);
            const auto index_1 = partition.get_1d_index_of_subdomain(my_subdomain);

            const auto& [min1, max1] = partition.calculate_subdomain_boundaries(index_1);

            ASSERT_EQ(min, min1) << min << min1;
            ASSERT_EQ(max, max1) << max << max1;

            local_subdomain_boundaries.emplace_back(min, max);
        }

        auto partition_local_subdomain_boundaries = partition.get_all_local_subdomain_boundaries();

        struct {
            bool operator()(RelearnTypes::bounding_box_type& a, RelearnTypes::bounding_box_type& b) const { return a < b; }
        } customLess;

        ranges::sort(local_subdomain_boundaries, customLess);
        ranges::sort(partition_local_subdomain_boundaries, customLess);

        ASSERT_EQ(local_subdomain_boundaries.size(), partition_local_subdomain_boundaries.size());
        for (auto i = 0U; i < local_subdomain_boundaries.size(); i++) {
            const auto& bb1 = local_subdomain_boundaries[i];
            const auto& bb2 = partition_local_subdomain_boundaries[i];

            ASSERT_TRUE(bb1.almost_equal(bb2, static_cast<RelearnTypes::space_type>(Constants::eps)));
        }

        for (const auto my_subdomain : ranges::views::indices(num_ranks)) {
            const auto idx = static_cast<std::size_t>(my_subdomain) + num_subdomains;
            ASSERT_THROW_NO_PRINT_MSG(std::ignore = partition.get_subdomain_boundaries(idx), RelearnException, fmt::format("{} {}", my_subdomain, num_ranks));
        }
    }
}

TEST_F(PartitionTest, testPartitionPositionToMpi) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto num_ranks_cast = static_cast<std::size_t>(num_ranks);
    const auto num_subdomains = round_to_next_exponent(num_ranks_cast, 8);
    const auto my_subdomains = num_subdomains / num_ranks_cast;

    const auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));
    if (oct_exponent >= std::numeric_limits<RelearnTypes::level_type>::max()) {
        return;
    }

    const auto& [simulation_box_maximum, simulation_box_minimum] = SimulationFactory::get_random_simulation_box_size(mt);

    for (const auto my_rank : ranges::views::indices(num_ranks)) {
        auto partition = Partition(num_ranks, mpiPP::MPIRank(my_rank));

        for ([[maybe_unused]] const auto j : ranges::views::indices(iterations)) {
            const auto& position = SimulationFactory::get_random_position_in_box({ simulation_box_maximum, simulation_box_minimum }, mt);
            ASSERT_THROW_NO_PRINT(std::ignore = partition.get_mpi_rank_from_position(position), RelearnException);
        }

        partition.set_simulation_box_size({ simulation_box_maximum, simulation_box_minimum });

        for ([[maybe_unused]] const auto j : ranges::views::indices(iterations)) {
            const auto& position = SimulationFactory::get_random_position_in_box({ simulation_box_maximum, simulation_box_minimum }, mt);
            const auto proposed_rank = partition.get_mpi_rank_from_position(position);

            const auto index_1_start = static_cast<std::size_t>(proposed_rank) * my_subdomains;

            auto correct = false;

            for (auto subdomain_id = index_1_start; subdomain_id < index_1_start + my_subdomains; subdomain_id++) {
                const auto& [min, max] = partition.calculate_subdomain_boundaries(subdomain_id);
                const auto& is_in_subdomain = position.check_in_box(min, max);
                correct |= is_in_subdomain;
            }

            ASSERT_TRUE(correct);
        }
    }
}

TEST_F(PartitionTest, testMemoryFootprint) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto footprint = std::make_unique<utility::MemoryFootprint>(10);

    const auto num_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);

    const auto part = Partition(num_ranks, mpiPP::MPIRank::root_rank());

    part.record_memory_footprint(footprint);

    const auto& footprint_description = footprint->get_descriptions();
    ASSERT_EQ(footprint_description.size(), 1);

    ASSERT_TRUE(footprint_description.contains("Partition"));
    ASSERT_GT(footprint_description.at("Partition"), sizeof(Partition));
}

TEST_F(PartitionTest, testSetBoundaryCorrector) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);

    auto part = Partition(num_ranks, mpiPP::MPIRank::root_rank());

    ASSERT_THROW_NO_PRINT(part.set_boundary_correction_function({}), RelearnException);
    auto corrector = [](RelearnTypes::position_type /*val*/) -> RelearnTypes::position_type {
        return RelearnTypes::position_type{ 1.0 };
    };

    part.set_boundary_correction_function(corrector);
    ASSERT_THROW_NO_PRINT(part.set_boundary_correction_function({}), RelearnException);
}

TEST_F(PartitionTest, testCorrectedSubdomainBoundaries) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto num_ranks_cast = static_cast<std::size_t>(num_ranks);
    const auto num_subdomains = round_to_next_exponent(num_ranks_cast, 8);
    const auto my_subdomains = num_subdomains / num_ranks_cast;

    const auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));
    if (oct_exponent >= std::numeric_limits<RelearnTypes::level_type>::max()) {
        return;
    }

    const auto oct_exponent_cast = static_cast<RelearnTypes::level_type>(oct_exponent);
    const auto sfc = Morton(oct_exponent_cast);

    const auto& [simulation_box_minimum, simulation_box_maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    for (auto my_rank = 0; my_rank < num_ranks; my_rank++) {
        auto partition = Partition(num_ranks, mpiPP::MPIRank(my_rank));

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_maximum, simulation_box_minimum }), RelearnException);

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x() + static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() }, simulation_box_maximum }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x(), simulation_box_minimum.get_y() + static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_minimum.get_z() }, simulation_box_maximum }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x(), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() + static_cast<RelearnTypes::space_type>(Constants::uninitialized) }, simulation_box_maximum }), RelearnException);

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x() + static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() } }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y() + static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_maximum.get_z() } }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() + static_cast<RelearnTypes::space_type>(Constants::uninitialized) } }), RelearnException);

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x() - static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() }, simulation_box_maximum }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x(), simulation_box_minimum.get_y() - static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_minimum.get_z() }, simulation_box_maximum }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ { simulation_box_minimum.get_x(), simulation_box_minimum.get_y(), simulation_box_minimum.get_z() - static_cast<RelearnTypes::space_type>(Constants::uninitialized) }, simulation_box_maximum }), RelearnException);

        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x() - static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() } }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y() - static_cast<RelearnTypes::space_type>(Constants::uninitialized), simulation_box_maximum.get_z() } }), RelearnException);
        ASSERT_THROW_NO_PRINT(partition.set_simulation_box_size({ simulation_box_minimum, { simulation_box_maximum.get_x(), simulation_box_maximum.get_y(), simulation_box_maximum.get_z() - static_cast<RelearnTypes::space_type>(Constants::uninitialized) } }), RelearnException);

        ASSERT_THROW_NO_PRINT(std::ignore = partition.get_simulation_box_size(), RelearnException);

        partition.set_simulation_box_size({ simulation_box_minimum, simulation_box_maximum });
        const auto& [retrieved_min, retrieved_max] = partition.get_simulation_box_size();

        ASSERT_EQ(simulation_box_minimum, retrieved_min);
        ASSERT_EQ(simulation_box_maximum, retrieved_max);
    }

    for (auto my_rank = 0; my_rank < num_ranks; my_rank++) {
        auto partition = Partition(num_ranks, mpiPP::MPIRank(my_rank));
        partition.set_simulation_box_size({ simulation_box_minimum, simulation_box_maximum });

        auto corrector = [](RelearnTypes::position_type /*val*/) -> RelearnTypes::position_type {
            return RelearnTypes::position_type{ 1.0 };
        };

        partition.set_boundary_correction_function(corrector);

        for (auto subdomain_index_1 = std::size_t{ 0 }; subdomain_index_1 < num_subdomains; subdomain_index_1++) {
            const auto& subdomain_index_3 = sfc.map_1d_to_3d(subdomain_index_1);

            const auto& [min_1, max_1] = partition.calculate_subdomain_boundaries(subdomain_index_1);
            const auto& [min_3, max_3] = partition.calculate_subdomain_boundaries(subdomain_index_3);

            ASSERT_EQ(min_1, min_3) << min_1 << min_3;
            ASSERT_EQ(max_1, max_3) << max_1 << max_3;

            ASSERT_EQ(min_1, RelearnTypes::position_type(1.0));
            ASSERT_EQ(max_1, RelearnTypes::position_type(1.0));
        }

        partition.calculate_and_set_subdomain_boundaries();

        const auto local_subdomain_boundaries = std::vector<std::pair<RelearnTypes::position_type, RelearnTypes::position_type>>{};

        for (auto my_subdomain = 0U; my_subdomain < my_subdomains; my_subdomain++) {
            const auto& [min, max] = partition.get_subdomain_boundaries(my_subdomain);
            const auto index_1 = partition.get_1d_index_of_subdomain(my_subdomain);

            const auto& [min1, max1] = partition.calculate_subdomain_boundaries(index_1);

            ASSERT_EQ(min, min1) << min << min1;
            ASSERT_EQ(max, max1) << max << max1;

            ASSERT_EQ(min1, RelearnTypes::position_type(1.0));
            ASSERT_EQ(max1, RelearnTypes::position_type(1.0));
        }
    }
}

TEST_F(PartitionTest, testPrintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto num_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);

    auto part = Partition(num_ranks, mpiPP::MPIRank::root_rank());

    ASSERT_NO_THROW(part.print_my_subdomains_info_rank());
}
