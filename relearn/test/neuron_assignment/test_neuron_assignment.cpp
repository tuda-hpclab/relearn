/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_assignment.h"

#include "Config.h"

#include "io/NeuronIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "neurons/enums/SynapticElementType.h"
#include "sim/Essentials.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/SynapseLoader.h"
#include "sim/file/MultipleSubdomainsFromFile.h"
#include "sim/random/SubdomainFromNeuronDensity.h"
#include "sim/random/SubdomainFromNeuronPerRank.h"
#include "structure/Partition.h"
#include "types/BasicTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronFilePaths.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnAllocator.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include "factory/local_group_translator/local_group_translator_factory.h"
#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"
#include "factory/synapses/synapses_factory.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPIRankRange.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static double calculate_excitatory_fraction(const std::span<const SignalType>& types) {
    auto number_excitatory = 0;
    auto number_inhibitory = 0;

    for (const auto& type : types) {
        if (type == SignalType::Excitatory) {
            number_excitatory++;
        } else {
            number_inhibitory++;
        }
    }

    const auto ratio = static_cast<double>(number_excitatory) / static_cast<double>(number_excitatory + number_inhibitory);
    return ratio;
}

static void write_synapses_to_file(const std::vector<PlasticLocalSynapse>& synapses, const std::filesystem::path& path) {
    auto in_of = std::ofstream(path / "rank_0_in_network.txt");
    auto out_of = std::ofstream(path / "rank_0_out_network.txt");

    for (const auto& [target, source, weight] : synapses) {
        in_of << "0 " << (target.get_neuron_id() + 1) << '\t' << "0 " << (source.get_neuron_id() + 1) << '\t' << weight << '\t' << '1' << '\n';
        out_of << "0 " << (target.get_neuron_id() + 1) << '\t' << "0 " << (source.get_neuron_id() + 1) << '\t' << weight << '\t' << '1' << '\n';
    }
}

TEST_F(NeuronAssignmentTest, testDensityTooManyRanks) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt) * 2;
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW_NO_PRINT(SubdomainFromNeuronDensity sfnd(golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part), RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testDensityConstructor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    const auto sfnd = SubdomainFromNeuronDensity{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

    const auto number_neurons = sfnd.get_requested_number_neurons();
    const auto fraction_excitatory_neurons = sfnd.get_requested_ratio_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, number_neurons);
    ASSERT_EQ(golden_fraction_excitatory_neurons, fraction_excitatory_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    // box_length is a float; at the magnitudes here (thousands of um) its ULP can exceed
    // 1/golden_number_neurons, so also allow a magnitude-scaled tolerance for float32 rounding.
    ASSERT_NEAR(box_length, golden_box_length, std::max(1.0 / static_cast<double>(golden_number_neurons), std::abs(golden_box_length) * static_cast<double>(std::numeric_limits<float>::epsilon()) * 4.0));

    ASSERT_EQ(0, sfnd.get_number_placed_neurons());
    ASSERT_EQ(0.0, sfnd.get_ratio_placed_excitatory_neurons());
}

TEST_F(NeuronAssignmentTest, testDensityInitialize) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto sfnd = SubdomainFromNeuronDensity{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    // box_length is a float; at the magnitudes here (thousands of um) its ULP can exceed
    // 1/golden_number_neurons, so also allow a magnitude-scaled tolerance for float32 rounding.
    ASSERT_NEAR(box_length, golden_box_length, std::max(1.0 / static_cast<double>(golden_number_neurons), std::abs(golden_box_length) * static_cast<double>(std::numeric_limits<float>::epsilon()) * 4.0));

    sfnd.initialize();

    const auto requested_number_neurons = sfnd.get_requested_number_neurons();
    const auto placed_number_neurons = sfnd.get_number_placed_neurons();

    const auto requested_fraction_excitatory_neurons = sfnd.get_requested_ratio_excitatory_neurons();
    const auto placed_fraction_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, requested_number_neurons);

    ASSERT_NEAR(golden_fraction_excitatory_neurons, requested_fraction_excitatory_neurons, 1.0 / static_cast<double>(golden_number_neurons));

    ASSERT_EQ(requested_number_neurons, placed_number_neurons);
    ASSERT_NEAR(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons, 1.0 / static_cast<double>(golden_number_neurons));

    ASSERT_LE(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons);
}

TEST_F(NeuronAssignmentTest, testDensityNeuronAttributesSizes) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto sfnd = SubdomainFromNeuronDensity{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

    sfnd.initialize();
    sfnd.initialize_groups_and_local_group_translator();

    const auto placed_number_neurons = sfnd.get_number_placed_neurons();

    const auto& positions = sfnd.get_neuron_positions_in_subdomains();
    const auto& types = sfnd.get_neuron_types_in_subdomains();
    const auto placed_number_neurons_in_subdomain = sfnd.get_number_neurons_in_subdomains();

    ASSERT_EQ(placed_number_neurons, placed_number_neurons_in_subdomain);
    ASSERT_EQ(placed_number_neurons, positions.size());
    ASSERT_EQ(placed_number_neurons, types.size());
    ASSERT_EQ(placed_number_neurons, sfnd.get_local_group_translator()->get_number_neurons_in_total());
}

TEST_F(NeuronAssignmentTest, testDensityNeuronAttributesSemantic) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto sfnd = SubdomainFromNeuronDensity{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

    sfnd.initialize();

    const auto placed_ratio_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

    const auto& positions = sfnd.get_neuron_positions_in_subdomains();
    const auto& types = sfnd.get_neuron_types_in_subdomains();

    const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(types);
    ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, 1.0 / static_cast<double>(golden_number_neurons));

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto neurons_per_dimension = pow(static_cast<double>(golden_number_neurons), 1. / 3);
    const auto number_boxes = static_cast<size_t>(ceil(neurons_per_dimension));

    auto box_full = std::vector<bool>(number_boxes * number_boxes * number_boxes, false);

    for (const auto& position : positions) {
        ASSERT_TRUE(position.check_in_box(sim_box_min, sim_box_max));
        auto cast_position = Vec3s{ position / static_cast<RelearnTypes::space_type>(golden_um_per_neuron) };

        const auto x = cast_position.get_x();
        const auto y = cast_position.get_y();
        const auto z = cast_position.get_z();

        ASSERT_LE(x, number_boxes);
        ASSERT_LE(y, number_boxes);
        ASSERT_LE(z, number_boxes);

        const auto flag = box_full[(z * number_boxes * number_boxes) + (y * number_boxes) + x];

        ASSERT_FALSE(flag);
        box_full[(z * number_boxes * number_boxes) + (y * number_boxes) + x] = true;
    }
}

TEST_F(NeuronAssignmentTest, testDensityWritePositionsToFile) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto positions = std::vector<RelearnTypes::position_type>{};
    auto types = std::vector<SignalType>{};

    const auto path = std::filesystem::path("neurons0.tmp");

    NeuronsFactory::generate_random_neuron_positions_and_signals(positions, types, mt, path);

    const auto number_neurons = positions.size();

    auto file = std::ifstream(path, std::ios::binary | std::ios::in);

    auto lines = std::vector<std::string>{};

    auto str = std::string{};
    while (std::getline(file, str)) {
        if (str[0] == '#') {
            continue;
        }

        lines.emplace_back(str);
    }

    file.close();

    ASSERT_EQ(number_neurons, lines.size());

    auto is_there = std::vector<bool>(number_neurons, false);

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& desired_position = positions[neuron_id];
        const auto& desired_signal_type = types[neuron_id];

        const auto& current_line = lines[neuron_id];

        auto sstream = std::stringstream(current_line);

        auto id = std::size_t{};
        auto x = double{};
        auto y = double{};
        auto z = double{};
        auto type_string = std::string{};

        sstream
            >> id
            >> x
            >> y
            >> z
            >> type_string;

        ASSERT_TRUE(0 < id);
        ASSERT_TRUE(id <= number_neurons);

        ASSERT_FALSE(is_there[id - 1]);
        is_there[id - 1] = true;

        ASSERT_NEAR(x, desired_position.get_x(), eps);
        ASSERT_NEAR(y, desired_position.get_y(), eps);
        ASSERT_NEAR(z, desired_position.get_z(), eps);

        auto type = SignalType::Excitatory;
        if (type_string == "ex") {
            type = SignalType::Excitatory;
        } else if (type_string == "in") {
            type = SignalType::Inhibitory;
        } else {
            ASSERT_TRUE(false);
        }

        ASSERT_TRUE(type == desired_signal_type);
    }
    std::filesystem::remove(path);
}

TEST_F(NeuronAssignmentTest, testDensityWriteNeuronsToFiles) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path_groups_in = std::filesystem::path{ "./groups_in.tmp" };

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto um_per_neuron = RandomFactory::get_random_double(RelearnTypes::space_type{ 1 }, RelearnTypes::space_type{ 100 }, mt);

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank(0));
    part->set_total_number_neurons(number_neurons);
    auto sfnd = SubdomainFromNeuronDensity{ number_neurons, fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(um_per_neuron), part };

    sfnd.initialize();

    auto positions = sfnd.get_neuron_positions_in_subdomains();
    auto types = sfnd.get_neuron_types_in_subdomains();

    auto golden_neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    auto golden_group_id_to_group_name = RelearnTypes::group_names{};

    NeuronsFactory::generate_random_neuron_groups(golden_neuron_id_to_group_ids, golden_group_id_to_group_name, number_neurons, mt, path_groups_in);

    const auto& golden_neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(golden_neuron_id_to_group_ids);

    sfnd.initialize_groups_and_local_group_translator(path_groups_in);

    const auto path_positions_signals = std::filesystem::path{ "./positions.tmp" };
    const auto path_groups_out = std::filesystem::path{ "./groups_out.tmp" };

    const auto paths = NeuronFilePaths{ path_positions_signals, path_groups_out };

    sfnd.write_neurons_to_files(paths);

    // check positions and types
    auto file_positions = std::ifstream(path_positions_signals, std::ios::binary | std::ios::in);

    auto lines = std::vector<std::string>{};

    auto str = std::string{};
    while (std::getline(file_positions, str)) {
        if (str[0] == '#') {
            continue;
        }

        lines.emplace_back(str);
    }

    file_positions.close();

    ASSERT_EQ(number_neurons, lines.size());

    auto is_there = std::vector<bool>(number_neurons, false);

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& desired_position = positions[neuron_id];
        const auto& desired_signal_type = types[neuron_id];

        const auto& current_line = lines[neuron_id];

        auto sstream = std::stringstream(current_line);

        auto id = std::size_t{};
        auto x = double{};
        auto y = double{};
        auto z = double{};
        auto type_string = std::string{};

        sstream
            >> id
            >> x
            >> y
            >> z
            >> type_string;

        ASSERT_TRUE(0 < id);
        ASSERT_TRUE(id <= number_neurons);

        ASSERT_FALSE(is_there[id - 1]);
        is_there[id - 1] = true;

        ASSERT_NEAR(x, desired_position.get_x(), eps);
        ASSERT_NEAR(y, desired_position.get_y(), eps);
        ASSERT_NEAR(z, desired_position.get_z(), eps);

        auto type = SignalType::Excitatory;
        if (type_string == "ex") {
            type = SignalType::Excitatory;
        } else if (type_string == "in") {
            type = SignalType::Inhibitory;
        } else {
            ASSERT_TRUE(false);
        }

        ASSERT_TRUE(type == desired_signal_type);
    }

    // check groups
    const auto& [neuron_id_to_group_ids, group_id_to_group_name] = NeuronIO::read_neuron_groups(path_groups_out, number_neurons);

    const auto& neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(golden_neuron_id_to_group_ids);

    ASSERT_EQ(golden_neuron_id_to_group_ids_unordered, neuron_id_to_group_ids_unordered); // only test with unordered group ids because order of group ids in regular group ids can be different
    ASSERT_EQ(golden_group_id_to_group_name, golden_group_id_to_group_name);

    std::filesystem::remove(path_groups_in);
    std::filesystem::remove(path_groups_out);
    std::filesystem::remove(path_positions_signals);
}

TEST_F(NeuronAssignmentTest, testPerRankTooFewNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<double>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<double>(mt) * 100;

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW_NO_PRINT(SubdomainFromNeuronPerRank sfnpr(0, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part), RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testPerRankSingleSubdomain) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    const auto sfnpr = SubdomainFromNeuronPerRank{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

    const auto number_neurons = sfnpr.get_requested_number_neurons();
    const auto fraction_excitatory_neurons = sfnpr.get_requested_ratio_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, number_neurons);
    ASSERT_EQ(golden_fraction_excitatory_neurons, fraction_excitatory_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    // box_length is a float; at the magnitudes here (thousands of um) its ULP can exceed
    // 1/golden_number_neurons, so also allow a magnitude-scaled tolerance for float32 rounding.
    ASSERT_NEAR(box_length, golden_box_length, std::max(1.0 / static_cast<double>(golden_number_neurons), std::abs(golden_box_length) * static_cast<double>(std::numeric_limits<float>::epsilon()) * 4.0));

    ASSERT_EQ(0, sfnpr.get_number_placed_neurons());
    ASSERT_EQ(0.0, sfnpr.get_ratio_placed_excitatory_neurons());
}

TEST_F(NeuronAssignmentTest, testPerRankConstructorMultipleSubdomains) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto number_subdomains = round_to_next_exponent(static_cast<std::size_t>(golden_number_ranks), 8);
    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + (number_subdomains * 50);
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        const auto sfnpr = SubdomainFromNeuronPerRank{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

        const auto number_neurons = sfnpr.get_requested_number_neurons();
        const auto fraction_excitatory_neurons = sfnpr.get_requested_ratio_excitatory_neurons();

        ASSERT_EQ(golden_number_neurons * static_cast<RelearnTypes::number_neurons_type>(golden_number_ranks), number_neurons);

        ASSERT_NEAR(golden_fraction_excitatory_neurons, fraction_excitatory_neurons, 1.0 / static_cast<double>(golden_number_neurons));

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        const auto box_length = (sim_box_max - sim_box_min).get_maximum();

        const auto number_neurons_per_box_max = static_cast<size_t>(ceil(static_cast<double>(golden_number_neurons) / static_cast<double>(part->get_number_local_subdomains())));

        const auto golden_box_length = calculate_box_length(number_neurons_per_box_max, golden_um_per_neuron) * static_cast<RelearnTypes::space_type>(part->get_number_subdomains_per_dimension());
        // box_length is a float; at the magnitudes here (thousands of um) its ULP can exceed
        // 1/golden_number_neurons, so also allow a magnitude-scaled tolerance for float32 rounding.
        ASSERT_NEAR(box_length, golden_box_length, std::max(1.0 / static_cast<double>(golden_number_neurons), std::abs(golden_box_length) * static_cast<double>(std::numeric_limits<float>::epsilon()) * 4.0));

        ASSERT_EQ(0, sfnpr.get_number_placed_neurons());
        ASSERT_EQ(0.0, sfnpr.get_ratio_placed_excitatory_neurons());
    }
}

TEST_F(NeuronAssignmentTest, testPerRankInitializeSingleSubdomain) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto sfnpr = SubdomainFromNeuronPerRank{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    // box_length is a float; at the magnitudes here (thousands of um) its ULP can exceed
    // 1/golden_number_neurons, so also allow a magnitude-scaled tolerance for float32 rounding.
    ASSERT_NEAR(box_length, golden_box_length, std::max(1.0 / static_cast<double>(golden_number_neurons), std::abs(golden_box_length) * static_cast<double>(std::numeric_limits<float>::epsilon()) * 4.0));

    sfnpr.initialize();

    const auto requested_number_neurons = sfnpr.get_requested_number_neurons();
    const auto placed_number_neurons = sfnpr.get_number_placed_neurons();

    const auto requested_fraction_excitatory_neurons = sfnpr.get_requested_ratio_excitatory_neurons();
    const auto placed_fraction_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, requested_number_neurons);

    ASSERT_NEAR(golden_fraction_excitatory_neurons, requested_fraction_excitatory_neurons, 1.0 / static_cast<double>(golden_number_neurons));

    ASSERT_EQ(requested_number_neurons, placed_number_neurons);
    ASSERT_NEAR(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons, 1.0 / static_cast<double>(golden_number_neurons));

    ASSERT_LE(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons);
}

TEST_F(NeuronAssignmentTest, testPerRankInitializeMultipleSubdomains) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto number_subdomains = round_to_next_exponent(static_cast<std::size_t>(golden_number_ranks), 8);
    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + (number_subdomains * 50);
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        auto sfnpr = SubdomainFromNeuronPerRank{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

        sfnpr.initialize();

        const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
        const auto placed_ratio_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

        ASSERT_EQ(placed_number_neurons, golden_number_neurons);
        ASSERT_NEAR(golden_fraction_excitatory_neurons, placed_ratio_excitatory_neurons, static_cast<double>(number_subdomains) / static_cast<double>(golden_number_neurons));
    }
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSizesSingleSubdomain) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto sfnpr = SubdomainFromNeuronPerRank{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

    sfnpr.initialize();
    sfnpr.initialize_groups_and_local_group_translator();

    const auto placed_number_neurons = sfnpr.get_number_placed_neurons();

    const auto& positions = sfnpr.get_neuron_positions_in_subdomains();
    const auto& types = sfnpr.get_neuron_types_in_subdomains();
    const auto placed_number_neurons_in_subdomain = sfnpr.get_number_neurons_in_subdomains();

    ASSERT_EQ(placed_number_neurons, placed_number_neurons_in_subdomain);
    ASSERT_EQ(placed_number_neurons, positions.size());
    ASSERT_EQ(placed_number_neurons, types.size());
    ASSERT_EQ(placed_number_neurons, sfnpr.get_local_group_translator()->get_number_neurons_in_total());
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSizeMultipleSubdomains) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto number_subdomains = round_to_next_exponent(static_cast<std::size_t>(golden_number_ranks), 8);
    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + (number_subdomains * 50);
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    auto accumulated_placed_neurons = std::size_t{ 0 };

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        auto sfnpr = SubdomainFromNeuronPerRank{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnpr.initialize();
        sfnpr.initialize_groups_and_local_group_translator();

        const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
        accumulated_placed_neurons += placed_number_neurons;

        const auto& all_positions = sfnpr.get_neuron_positions_in_subdomains();
        const auto& all_types = sfnpr.get_neuron_types_in_subdomains();
        const auto all_placed_neurons_in_subdomains = sfnpr.get_number_neurons_in_subdomains();

        ASSERT_EQ(placed_number_neurons, all_placed_neurons_in_subdomains);
        ASSERT_EQ(placed_number_neurons, all_positions.size());
        ASSERT_EQ(placed_number_neurons, all_types.size());
        ASSERT_EQ(placed_number_neurons, sfnpr.get_local_group_translator()->get_number_neurons_in_total());
    }

    ASSERT_EQ(accumulated_placed_neurons, static_cast<std::size_t>(golden_number_ranks) * golden_number_neurons);
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSemanticSingleSubdomain) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 100;
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto sfnpr = SubdomainFromNeuronPerRank{ golden_number_neurons, golden_fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(golden_um_per_neuron), part };

    sfnpr.initialize();

    const auto placed_ratio_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

    const auto& positions = sfnpr.get_neuron_positions_in_subdomains();
    const auto& types = sfnpr.get_neuron_types_in_subdomains();

    const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(types);
    ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, 1.0 / static_cast<double>(golden_number_neurons));

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto neurons_per_dimension = pow(static_cast<double>(golden_number_neurons), 1. / 3.);
    const auto number_boxes = static_cast<size_t>(ceil(neurons_per_dimension));

    auto box_full = std::vector<bool>(number_boxes * number_boxes * number_boxes, false);

    for (const auto& position : positions) {
        ASSERT_TRUE(position.check_in_box(sim_box_min, sim_box_max));
        const auto cast_position = Vec3s{ position / static_cast<RelearnTypes::space_type>(golden_um_per_neuron) };

        const auto x = cast_position.get_x();
        const auto y = cast_position.get_y();
        const auto z = cast_position.get_z();

        ASSERT_LE(x, number_boxes);
        ASSERT_LE(y, number_boxes);
        ASSERT_LE(z, number_boxes);

        const auto flag = box_full[(z * number_boxes * number_boxes) + (y * number_boxes) + x];

        ASSERT_FALSE(flag);
        box_full[(z * number_boxes * number_boxes) + (y * number_boxes) + x] = true;
    }
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSemanticMultipleSubdomains) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto number_subdomains = round_to_next_exponent(static_cast<std::size_t>(golden_number_ranks), 8);
    const auto golden_number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + (number_subdomains * 50);
    const auto golden_fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto golden_um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        auto sfnpr = SubdomainFromNeuronPerRank{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnpr.initialize();
        sfnpr.initialize_groups_and_local_group_translator();

        const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
        const auto placed_ratio_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

        const auto& all_types = sfnpr.get_neuron_types_in_subdomains();

        const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(all_types);
        ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, static_cast<double>(number_subdomains) / static_cast<double>(golden_number_neurons)) << golden_number_neurons;

        const auto& positions = sfnpr.get_neuron_positions_in_subdomains();
        const auto& types = sfnpr.get_neuron_types_in_subdomains();

        ASSERT_EQ(placed_number_neurons, golden_number_neurons);
        ASSERT_EQ(positions.size(), golden_number_neurons);
        ASSERT_EQ(types.size(), golden_number_neurons);
        ASSERT_EQ(sfnpr.get_local_group_translator()->get_number_neurons_in_total(), golden_number_neurons);
    }
}

TEST_F(NeuronAssignmentTest, testFileLoadSingleSubdomain) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto positions = std::vector<RelearnTypes::position_type>{};
    auto types = std::vector<SignalType>{};

    const auto path = std::filesystem::path{ "./neurons0.tmp" };

    NeuronsFactory::generate_random_neuron_positions_and_signals(positions, types, mt, path);

    const auto number_neurons = positions.size();

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto sff = MultipleSubdomainsFromFile{ path, {}, part };

    sff.initialize();

    const auto& loaded_positions = sff.get_neuron_positions_in_subdomains();
    const auto& loaded_types = sff.get_neuron_types_in_subdomains();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& curr_pos = positions[neuron_id];
        const auto& curr_loaded_pos = loaded_positions[neuron_id];

        ASSERT_NEAR(curr_pos.get_x(), curr_loaded_pos.get_x(), eps);
        ASSERT_NEAR(curr_pos.get_y(), curr_loaded_pos.get_y(), eps);
        ASSERT_NEAR(curr_pos.get_z(), curr_loaded_pos.get_z(), eps);

        const auto& curr_type = types[neuron_id];
        const auto& curr_loaded_type = loaded_types[neuron_id];

        ASSERT_EQ(curr_type, curr_loaded_type);
    }
    std::filesystem::remove(path);
}

TEST_F(NeuronAssignmentTest, testFileLoadNetworkSingleSubdomain) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto positions = std::vector<RelearnTypes::position_type>{};
    auto types = std::vector<SignalType>{};

    const auto path = std::filesystem::path{ "./neurons1.tmp" };

    NeuronsFactory::generate_random_neuron_positions_and_signals(positions, types, mt, path);

    const auto number_neurons = positions.size();

    const auto& synapses = SynapsesFactory::generate_local_synapses(number_neurons, mt);

    write_synapses_to_file(synapses, ".");

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto sff = MultipleSubdomainsFromFile{ path, ".", part };

    sff.initialize();

    const auto loader = sff.get_synapse_loader();

    const auto& [static_synapses, plastic_synapses] = loader->load_synapses(std::make_unique<Essentials>());

    const auto& [local_synapses, in_synapses, out_synapses] = plastic_synapses;

    ASSERT_TRUE(in_synapses.empty());
    ASSERT_TRUE(out_synapses.empty());

    auto synapse_map = std::map<std::pair<NeuronID, NeuronID>, RelearnTypes::plastic_synapse_weight>{};

    for (const auto& [target, source, weight] : local_synapses) {
        synapse_map[{ target, source }] += weight;
    }

    for (const auto& [target, source, weight] : synapses) {
        synapse_map[{ target, source }] -= weight;
    }

    for (const auto& [_, weight] : synapse_map) {
        ASSERT_NEAR(weight, 0.0, eps);
    }
    std::filesystem::remove(path);
}

TEST_F(NeuronAssignmentTest, testFileGivenInputONCE) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto orig_sl = [number_neurons, this]() {
        const auto fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
        const auto um_per_neuron = RandomFactory::get_random_double(RelearnTypes::space_type{ 1 }, RelearnTypes::space_type{ 100 }, mt);

        const auto part = std::make_shared<Partition>(1, mpiPP::MPIInfo::get_my_rank());
        part->set_total_number_neurons(number_neurons);
        SubdomainFromNeuronDensity sfnd{ number_neurons, fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(um_per_neuron), part };
        sfnd.initialize();
        sfnd.write_neuron_positions_and_signals_to_file("rank_0_positions.txt");

        return sfnd.get_synapse_loader();
    }();
    [[maybe_unused]] const auto& [_, orig_plastic_synapses] = orig_sl->load_synapses(std::make_unique<Essentials>());
    const auto& [orig_local_synapses, orig_in_synapses, orig_out_synapses] = orig_plastic_synapses;
    auto path_to_neurons = std::filesystem::current_path() / "rank_0_positions.txt";

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    auto sff = MultipleSubdomainsFromFile{ path_to_neurons, std::nullopt, part };

    sff.initialize();

    const auto& positions = sff.get_neuron_positions_in_subdomains();

    ASSERT_EQ(positions.size(), number_neurons);

    const auto sl = sff.get_synapse_loader();

    const auto& [static_synapses, plastic_synapses] = sl->load_synapses(std::make_unique<Essentials>());

    const auto& [local_synapses, in_synapses, out_synapses] = plastic_synapses;

    ASSERT_TRUE(in_synapses.empty());
    ASSERT_TRUE(out_synapses.empty());
    ASSERT_EQ(in_synapses, orig_in_synapses);
    ASSERT_EQ(out_synapses, orig_out_synapses);
    ASSERT_EQ(local_synapses, orig_local_synapses);
}

TEST_F(NeuronAssignmentTest, testMultipleFilesEmptyPositionPath) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        auto partition = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW_NO_PRINT(MultipleSubdomainsFromFile msff(std::filesystem::path(""), {}, partition);, RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testMultipleFilesNonExistentPositionPath) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        auto partition = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW_NO_PRINT(MultipleSubdomainsFromFile msff(std::filesystem::path("./asfhasdfb�aslidhsdjfnasd"), {}, partition);, RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testMultipleFilesNonExistentFiles) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    namespace fs = std::filesystem;

    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto directory = fs::path("./temp_dir0");

    if (fs::exists(directory)) {
        for (const auto& path : directory) {
            if (path.string()[0] == '.') {
                continue;
            }
            fs::remove_all(path);
        }
    } else {
        fs::create_directory(directory);
    }

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        auto partition = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW_NO_PRINT(MultipleSubdomainsFromFile msff(directory, {}, partition);, RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testMultipleFilesEmptyFiles) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    namespace fs = std::filesystem;

    const auto golden_number_ranks = MPIRankFactory::get_adjusted_random_number_ranks(mt);
    const auto directory = fs::path("./temp_dir1");

    if (fs::exists(directory)) {
        for (const auto& path : directory) {
            if (path.string()[0] == '.') {
                continue;
            }

            fs::remove_all(path);
        }
    } else {
        fs::create_directory(directory);
    }

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        auto position_path = directory / ("rank_" + std::to_string(rank.get_rank()) + "_positions.txt");
        auto out_file = std::ofstream{ position_path };
        out_file.flush();
    }

    for (const auto rank : mpiPP::MPIRankRange::range(golden_number_ranks)) {
        auto partition = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW_NO_PRINT(MultipleSubdomainsFromFile msff(directory, {}, partition);, RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testInitializeLocalGroupTranslatorFromFile) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path_positions = std::filesystem::path{ "./positions0.tmp" };

    auto positions = std::vector<RelearnTypes::position_type>{};
    auto types = std::vector<SignalType>{};

    NeuronsFactory::generate_random_neuron_positions_and_signals(positions, types, mt, path_positions);

    const auto number_neurons = positions.size();

    const auto lgt_tmp = LocalGroupTranslatorFactory::get_randomized_group_translator(number_neurons, mt);

    const auto path = std::filesystem::path{ "./groups0.tmp" };

    NeuronIO::write_neuron_groups(path, lgt_tmp);

    const auto fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part1 = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    const auto part2 = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    const auto part3 = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());

    auto sfnd = SubdomainFromNeuronDensity{ number_neurons, fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(um_per_neuron), part1 };
    auto sfnpr = SubdomainFromNeuronPerRank{ number_neurons, fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(um_per_neuron), part2 };
    auto sff = MultipleSubdomainsFromFile{ path_positions, std::nullopt, part3 };

    const auto assignments = std::vector<NeuronToSubdomainAssignment*>{ &sfnd, &sfnpr, &sff };

    for (const auto& assignment : assignments) {
        assignment->initialize();
        assignment->initialize_groups_and_local_group_translator(path);

        const auto& translator = assignment->get_local_group_translator();

        const auto& golden_neuron_id_to_group_ids_unordered = lgt_tmp->get_neuron_ids_to_group_ids_unordered();
        const auto& golden_group_id_to_group_name = lgt_tmp->get_all_group_names();

        const auto& neuron_id_to_group_ids_unordered = translator->get_neuron_ids_to_group_ids_unordered();
        const auto& group_id_to_group_name = translator->get_all_group_names();

        ASSERT_EQ(golden_neuron_id_to_group_ids_unordered, neuron_id_to_group_ids_unordered); // only test with unordered group ids because order of group ids in regular group ids can be different
        ASSERT_EQ(golden_group_id_to_group_name, group_id_to_group_name);
    }

    std::filesystem::remove(path_positions);
    std::filesystem::remove(path);
}

TEST_F(NeuronAssignmentTest, testInitializeLocalGroupTranslatorDefault) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path_positions = std::filesystem::path{ "./positions1.tmp" };

    auto positions = std::vector<RelearnTypes::position_type>{};
    auto types = std::vector<SignalType>{};

    NeuronsFactory::generate_random_neuron_positions_and_signals(positions, types, mt, path_positions);

    const auto number_neurons = positions.size();

    const auto fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part1 = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    const auto part2 = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    const auto part3 = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());

    auto sfnd = SubdomainFromNeuronDensity{ number_neurons, fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(um_per_neuron), part1 };
    auto sfnpr = SubdomainFromNeuronPerRank{ number_neurons, fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(um_per_neuron), part2 };
    auto sff = MultipleSubdomainsFromFile{ path_positions, std::nullopt, part3 };

    const auto assignments = std::vector<NeuronToSubdomainAssignment*>{ &sfnd, &sfnpr, &sff };

    for (const auto& assignment : assignments) {
        assignment->initialize();
        assignment->initialize_groups_and_local_group_translator();

        const auto& translator = assignment->get_local_group_translator();

        ASSERT_EQ(number_neurons, translator->get_number_neurons_in_total());

        const auto& neuron_id_to_group_ids = translator->get_neuron_ids_to_group_ids();
        const auto& neuron_id_to_group_ids_unordered = translator->get_neuron_ids_to_group_ids_unordered();
        const auto& group_id_to_group_name = translator->get_all_group_names();

        ASSERT_EQ(group_id_to_group_name, RelearnTypes::group_names{ std::string{ Constants::default_group_name } });

        for (auto neuron_id = 0UL; neuron_id < neuron_id_to_group_ids.size(); ++neuron_id) {
            const auto& group_ids = neuron_id_to_group_ids[neuron_id];
            const auto& group_ids_unordered = neuron_id_to_group_ids_unordered[neuron_id];

            ASSERT_EQ(group_ids, RelearnTypes::group_ids{ Constants::default_group_id });
            ASSERT_EQ(group_ids_unordered, RelearnTypes::group_ids_unordered{ Constants::default_group_id });
        }
    }

    std::filesystem::remove(path_positions);
}

TEST_F(NeuronAssignmentTest, testWriteNeuronGroupsToFile) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path_positions = std::filesystem::path{ "./positions2.tmp" };
    const auto path_groups_in = std::filesystem::path{ "./groups1.tmp" };

    const auto paths = NeuronOptFilePaths{ path_positions, path_groups_in };

    auto positions = std::vector<RelearnTypes::position_type>{};
    auto types = std::vector<SignalType>{};
    auto golden_neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    auto golden_group_id_to_group_name = RelearnTypes::group_names{};

    NeuronsFactory::generate_random_neurons(positions, golden_neuron_id_to_group_ids, golden_group_id_to_group_name, types, mt, paths);

    const auto& golden_neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(golden_neuron_id_to_group_ids);

    const auto path_groups_out = std::filesystem::path{ "./groups2.tmp" };

    const auto number_neurons = positions.size();

    const auto fraction_excitatory_neurons = RandomFactory::get_random_percentage<RelearnTypes::percentage_type>(mt);
    const auto um_per_neuron = RandomFactory::get_random_percentage<RelearnTypes::space_type>(mt) * 100;

    const auto part1 = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    const auto part2 = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());
    const auto part3 = std::make_shared<Partition>(1, mpiPP::MPIRank::root_rank());

    auto sfnd = SubdomainFromNeuronDensity{ number_neurons, fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(um_per_neuron), part1 };
    auto sfnpr = SubdomainFromNeuronPerRank{ number_neurons, fraction_excitatory_neurons, static_cast<RelearnTypes::space_type>(um_per_neuron), part2 };
    auto sff = MultipleSubdomainsFromFile{ path_positions, std::nullopt, part3 };

    const auto assignments = std::vector<NeuronToSubdomainAssignment*>{ &sfnd, &sfnpr, &sff };

    for (const auto& assignment : assignments) {
        assignment->initialize();
        assignment->initialize_groups_and_local_group_translator(path_groups_in);

        assignment->write_neuron_groups_to_file(path_groups_out);

        const auto& [neuron_id_to_group_ids, group_id_to_group_name] = NeuronIO::read_neuron_groups(path_groups_out, number_neurons);

        const auto& neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(neuron_id_to_group_ids);

        ASSERT_EQ(golden_neuron_id_to_group_ids_unordered, neuron_id_to_group_ids_unordered); // only test with unordered group ids because order of group ids in regular group ids can be different
        ASSERT_EQ(golden_group_id_to_group_name, group_id_to_group_name);
    }

    std::filesystem::remove(path_positions);
    std::filesystem::remove(path_groups_in);
    std::filesystem::remove(path_groups_out);
}
