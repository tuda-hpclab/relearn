/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_io.h"

#include "Config.h"
#include "RelearnTest.hpp"

#include "io/NeuronIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "neurons/enums/SynapticElementType.h"
#include "sim/LoadedNeuron.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronFilePaths.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"
#include "util/shuffle/shuffle.h"

#include "factory/local_group_translator/local_group_translator_factory.h"
#include "factory/mpi_rank/mpi_rank_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/neuron_types/neuron_types_factory.h"
#include "factory/neurons/neurons_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"
#include "factory/synapses/synapses_factory.h"

#include <cpp-utility/Cast.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <range/v3/algorithm/contains.hpp>
#include <range/v3/algorithm/count.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/indices.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

TEST_F(IOTest, testNeuronIOWritePositionsSignalsComponentwiseSizeExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    const auto correct_ids = std::vector<NeuronID>{ number_neurons };
    const auto correct_positions = std::vector<RelearnTypes::position_type>{ number_neurons };
    const auto correct_signal_types = std::vector<SignalType>{ number_neurons };

    const auto faulty_ids = std::vector<NeuronID>{ number_neurons + 1 };
    const auto faulty_positions = std::vector<RelearnTypes::position_type>{ number_neurons + 1 };
    const auto faulty_signal_types = std::vector<SignalType>{ number_neurons + 1 };

    const auto path = std::filesystem::path{ "./neurons0.tmp" };

    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals_componentwise(faulty_ids, correct_positions, correct_signal_types, path), RelearnException);
    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals_componentwise(correct_ids, faulty_positions, correct_signal_types, path), RelearnException);
    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals_componentwise(correct_ids, correct_positions, faulty_signal_types, path), RelearnException);
    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals_componentwise(correct_ids, faulty_positions, faulty_signal_types, path), RelearnException);
    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals_componentwise(faulty_ids, correct_positions, faulty_signal_types, path), RelearnException);
    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals_componentwise(faulty_ids, faulty_positions, correct_signal_types, path), RelearnException);
    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals_componentwise(faulty_ids, faulty_positions, faulty_signal_types, path), RelearnException);

    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOWritePositionsSignalsComponentwiseFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position(mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto bad_path = std::filesystem::path{ "" };

    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, bad_path), RelearnException);
    std::filesystem::remove(bad_path);
}

TEST_F(IOTest, testNeuronIOWritePositionsSignalsComponentwise) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position(mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto path = std::filesystem::path{ "./neurons1.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path));
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsComponentwise) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto path = std::filesystem::path{ "./neurons2.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);

    const auto& [read_ids, read_positions, read_signal_types, additional_infos]
        = NeuronIO::read_neuron_positions_and_signals_componentwise(path);

    ASSERT_EQ(preliminary_ids, read_ids);
    ASSERT_EQ(preliminary_signal_types, read_signal_types);

    ASSERT_EQ(preliminary_position.size(), read_positions.size());

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& diff = preliminary_position[neuron_id] - read_positions[neuron_id];
        const auto norm = diff.calculate_2_norm();

        // NeuronIO writes a position with as many digits as space_type carries, so it comes back accurate
        // relative to its magnitude and not to an absolute epsilon.
        ASSERT_NEAR(0.0, norm, tolerance_for<RelearnTypes::space_type>(preliminary_position[neuron_id].calculate_2_norm()));
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    const auto number_excitatory = ranges::count(preliminary_signal_types, SignalType::Excitatory);
    const auto number_inhibitory = ranges::count(preliminary_signal_types, SignalType::Inhibitory);

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    auto minimum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    auto maximum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    for (const auto& position : read_positions) {
        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);
    }

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsComponentwiseFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto bad_path = std::filesystem::path{ "" };
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals_componentwise(bad_path), RelearnException);
    std::filesystem::remove(bad_path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsComponentwiseIDException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto idx1 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 1, mt);
    auto idx2 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 2, mt);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(preliminary_ids[idx1], preliminary_ids[idx2]);

    const auto path = std::filesystem::path{ "./neurons3.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals_componentwise(path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsComponentwisePositionXException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto idx1 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_x(-preliminary_position[idx1].get_x());

    const auto path = std::filesystem::path{ "./neurons4.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals_componentwise(path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsComponentwisePositionYException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto idx1 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_y(-preliminary_position[idx1].get_y());

    const auto path = std::filesystem::path{ "./neurons5.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals_componentwise(path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionZException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto idx1 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_z(-preliminary_position[idx1].get_z());

    const auto path = std::filesystem::path{ "./neurons6.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals_componentwise(path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOWritePositionsSignals1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position(mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));

        preliminary_neurons.emplace_back(preliminary_position[neuron_id], preliminary_ids[neuron_id], preliminary_signal_types[neuron_id]);
    }

    const auto path = std::filesystem::path{ "./neurons7.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neuron_positions_and_signals(preliminary_neurons, path, nullptr));

    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOWritePositionsSignals2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));

        preliminary_neurons.emplace_back(preliminary_position[neuron_id], preliminary_ids[neuron_id], preliminary_signal_types[neuron_id]);
    }

    const auto path = std::filesystem::path{ "./neurons8.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neuron_positions_and_signals(preliminary_neurons, path, nullptr));

    const auto& [read_ids, read_positions, read_signal_types, additional_infos]
        = NeuronIO::read_neuron_positions_and_signals_componentwise(path);

    ASSERT_EQ(preliminary_ids, read_ids);
    ASSERT_EQ(preliminary_signal_types, read_signal_types);

    ASSERT_EQ(preliminary_position.size(), read_positions.size());

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& diff = preliminary_position[neuron_id] - read_positions[neuron_id];
        const auto norm = diff.calculate_2_norm();

        // NeuronIO writes a position with as many digits as space_type carries, so it comes back accurate
        // relative to its magnitude and not to an absolute epsilon.
        ASSERT_NEAR(0.0, norm, tolerance_for<RelearnTypes::space_type>(preliminary_position[neuron_id].calculate_2_norm()));
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    const auto number_excitatory = ranges::count(preliminary_signal_types, SignalType::Excitatory);
    const auto number_inhibitory = ranges::count(preliminary_signal_types, SignalType::Inhibitory);

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    auto minimum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    auto maximum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    for (const auto& position : read_positions) {
        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);
    }

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);

    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOWritePositionsSignalsFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position(mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));

        preliminary_neurons.emplace_back(preliminary_position[neuron_id], preliminary_ids[neuron_id], preliminary_signal_types[neuron_id]);
    }

    const auto bad_path = std::filesystem::path{ "" };

    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals(preliminary_neurons, bad_path, nullptr), RelearnException);
    std::filesystem::remove(bad_path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignals) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_neurons.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt), NeuronID{ false, neuron_id }, NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto path = std::filesystem::path{ "./neurons9.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neuron_positions_and_signals(preliminary_neurons, path, nullptr));

    const auto& [read_neurons, additional_infos, _]
        = NeuronIO::read_neuron_positions_and_signals(path);

    ASSERT_EQ(read_neurons.size(), preliminary_neurons.size());

    auto minimum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    auto maximum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    auto number_excitatory = 0;
    auto number_inhibitory = 0;

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& preliminary_neuron = preliminary_neurons[neuron_id];
        const auto& read_neuron = read_neurons[neuron_id];

        const auto& diff = preliminary_neuron.pos - read_neuron.pos;
        const auto norm = diff.calculate_2_norm();

        // NeuronIO writes a position with as many digits as space_type carries, so it comes back accurate
        // relative to its magnitude and not to an absolute epsilon.
        ASSERT_NEAR(0.0, norm, tolerance_for<RelearnTypes::space_type>(preliminary_neuron.pos.calculate_2_norm()));

        ASSERT_EQ(read_neuron.id, preliminary_neuron.id);
        ASSERT_EQ(read_neuron.signal_type, preliminary_neuron.signal_type);

        minimum.calculate_componentwise_minimum(read_neuron.pos);
        maximum.calculate_componentwise_maximum(read_neuron.pos);

        if (read_neuron.signal_type == SignalType::Excitatory) {
            number_excitatory++;
        } else {
            number_inhibitory++;
        }
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);

    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto bad_path = std::filesystem::path{ "" };
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals(bad_path), RelearnException);
    std::filesystem::remove(bad_path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsIDException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 3;

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto idx1 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 1, mt);
    auto idx2 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 2, mt);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(preliminary_ids[idx1], preliminary_ids[idx2]);

    const auto path = std::filesystem::path{ "./neurons10.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals(path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsPositionXException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto idx1 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_x(-preliminary_position[idx1].get_x());

    const auto path = std::filesystem::path{ "./neurons11.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals(path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsPositionYException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto idx1 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_y(-preliminary_position[idx1].get_y());

    const auto path = std::filesystem::path{ "./neurons12.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals(path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadPositionsSignalsPositionZException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto idx1 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 1, mt);

    preliminary_position[idx1].set_z(-preliminary_position[idx1].get_z());

    const auto path = std::filesystem::path{ "./neurons13.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_positions_and_signals(path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadNeuronGroups) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    auto group_id_to_group_name = RelearnTypes::group_names{};

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    const auto path = std::filesystem::path{ "./groups1.tmp" };

    NeuronsFactory::generate_random_neuron_groups(neuron_id_to_group_ids, group_id_to_group_name, number_neurons, mt, path);

    const auto& neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(neuron_id_to_group_ids);
    const auto& neuron_id_to_group_names_unordered = NeuronsFactory::get_neuron_id_vs_group_names_unordered(neuron_id_to_group_ids_unordered, group_id_to_group_name);

    const auto& [read_neuron_id_to_group_ids, read_group_id_to_group_name] = NeuronIO::read_neuron_groups(path, number_neurons);

    const auto& read_neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(read_neuron_id_to_group_ids);
    const auto& read_neuron_id_to_group_names_unordered = NeuronsFactory::get_neuron_id_vs_group_names_unordered(read_neuron_id_to_group_ids_unordered, read_group_id_to_group_name);

    ASSERT_EQ(neuron_id_to_group_names_unordered, read_neuron_id_to_group_names_unordered);

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& actual_group_names = neuron_id_to_group_names_unordered[neuron_id];
        const auto& read_group_names = read_neuron_id_to_group_names_unordered[neuron_id];
        ASSERT_TRUE(ranges::contains(read_group_names, std::string{ Constants::default_group_name }));
        for (const auto& read_group_name : read_group_names) {
            ASSERT_TRUE(ranges::contains(actual_group_names, read_group_name));
        }
    }

    ASSERT_EQ(read_group_id_to_group_name[Constants::default_group_id], std::string{ Constants::default_group_name });
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadNeuronGroupsDeterministic) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = 7;

    const auto path = std::filesystem::path{ "./groups2.tmp" };

    auto ofs = std::ofstream(path, std::ios::binary | std::ios::out);

    ofs << "group_1 1 2 3\n";
    ofs << "group_2 2 3 4\n";
    ofs << "group_3 4 5 6\n";
    ofs << "group_4 1 5\n";
    ofs << "x 4\n";

    ofs.close();

    const auto& [neuron_id_to_group_ids, group_id_to_group_name] = NeuronIO::read_neuron_groups(path, number_neurons);

    const auto golden_groups = RelearnTypes::group_names{ std::string{ Constants::default_group_name }, "group_1", "group_2", "group_3", "group_4", "x" };

    ASSERT_EQ(group_id_to_group_name, golden_groups);

    const auto default_group_id = Constants::default_group_id;

    const auto golden_neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{
        { default_group_id, 1, 4 },    // 1
        { default_group_id, 1, 2 },    // 2
        { default_group_id, 1, 2 },    // 3
        { default_group_id, 2, 3, 5 }, // 4
        { default_group_id, 3, 4 },    // 5
        { default_group_id, 3 },       // 6
        { default_group_id }           // 7
    };

    ASSERT_EQ(neuron_id_to_group_ids, golden_neuron_id_to_group_ids);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadNeuronGroupsExceptions) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto bad_path = std::filesystem::path{ "" };

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_groups(bad_path, 100);, RelearnException);

    auto neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{};
    auto group_id_to_group_name = RelearnTypes::group_names{};

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    NeuronsFactory::generate_random_neuron_groups(neuron_id_to_group_ids, group_id_to_group_name, number_neurons, mt);

    auto one_neuron_too_much = neuron_id_to_group_ids;
    one_neuron_too_much.push_back({ Constants::default_group_id, 1 });

    const auto path = std::filesystem::path{ "./groups3.tmp" };

    const auto lgt = std::make_shared<LocalGroupTranslator>(group_id_to_group_name, one_neuron_too_much);

    NeuronIO::write_neuron_groups(path, lgt);

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_groups(path, number_neurons);, RelearnException);

    auto neither_neuron_id_nor_group_name = std::ofstream(path, std::ios::binary | std::ios::out);

    neither_neuron_id_nor_group_name << "1_1 1\n";

    neither_neuron_id_nor_group_name.close();

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_groups(path, number_neurons);, RelearnException);

    auto group_name_not_valid = std::ofstream(path, std::ios::binary | std::ios::out);

    group_name_not_valid << "1 1_1\n";

    group_name_not_valid.close();

    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_neuron_groups(path, number_neurons);, RelearnException);
    std::filesystem::remove(bad_path);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOWriteNeuronGroups) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);

    const auto& golden_neuron_id_to_group_ids_unordered = translator->get_neuron_ids_to_group_ids_unordered();

    const auto& golden_group_id_to_group_name = translator->get_all_group_names();

    const auto number_neurons = translator->get_number_neurons_in_total();

    const auto path = std::filesystem::path{ "./groups4.tmp" };

    NeuronIO::write_neuron_groups(path, translator);

    const auto& [read_neuron_id_to_group_ids, read_group_id_to_group_name] = NeuronIO::read_neuron_groups(path, number_neurons);

    const auto read_neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(read_neuron_id_to_group_ids);

    ASSERT_EQ(golden_neuron_id_to_group_ids_unordered, read_neuron_id_to_group_ids_unordered);
    ASSERT_EQ(golden_group_id_to_group_name, read_group_id_to_group_name);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOWriteNeuronGroupsDeterministic) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto default_group_id = Constants::default_group_id;
    const auto default_group_name = std::string{ Constants::default_group_name };

    const auto golden_neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{
        { default_group_id, 1, 4 },    // 1
        { default_group_id, 1, 2 },    // 2
        { default_group_id, 1, 2 },    // 3
        { default_group_id, 2, 3, 5 }, // 4
        { default_group_id, 3, 4 },    // 5
        { default_group_id, 3 },       // 6
        { default_group_id }           // 7
    };

    const auto golden_group_id_to_group_name = RelearnTypes::group_names{
        default_group_name,
        "group_1",
        "group_2",
        "group_3",
        "group_4",
        "x"
    };

    const auto number_neurons = 7;

    const auto translator = std::make_shared<LocalGroupTranslator>(golden_group_id_to_group_name, golden_neuron_id_to_group_ids);

    const auto path = std::filesystem::path{ "./groups5.tmp" };

    NeuronIO::write_neuron_groups(path, translator);

    const auto golden_neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(golden_neuron_id_to_group_ids);

    const auto& [read_neuron_id_to_group_ids, read_group_id_to_group_name] = NeuronIO::read_neuron_groups(path, number_neurons);

    const auto read_neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(read_neuron_id_to_group_ids);

    ASSERT_EQ(golden_neuron_id_to_group_ids_unordered, read_neuron_id_to_group_ids_unordered);
    ASSERT_EQ(golden_group_id_to_group_name, read_group_id_to_group_name);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOWriteNeuronGroupsEmptyPath) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);

    const auto bad_path = std::filesystem::path{ "" };

    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_groups(bad_path, translator), RelearnException);
    std::filesystem::remove(bad_path);
}

TEST_F(IOTest, testNeuronIOWriteNeuronGroupsSpecificNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);

    const auto& golden_neuron_id_to_group_ids_unordered = translator->get_neuron_ids_to_group_ids_unordered();

    const auto& golden_group_id_to_group_name = translator->get_all_group_names();

    const auto number_neurons = translator->get_number_neurons_in_total();

    auto all_ids = NeuronIDRange::range(number_neurons) | ranges::to_vector;

    const auto& ids = RandomFactory::sample(all_ids, mt);

    const auto path = std::filesystem::path{ "./groups6.tmp" };

    NeuronIO::write_neuron_groups_of_specific_neurons(ids, path, translator);

    const auto ids_set = ids | ranges::to<std::unordered_set>;

    const auto& [read_neuron_id_to_group_ids, read_group_id_to_group_name] = NeuronIO::read_neuron_groups(path, number_neurons);

    const auto read_neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(read_neuron_id_to_group_ids);

    for (auto neuron_id = 0UL; neuron_id < read_neuron_id_to_group_ids.size(); ++neuron_id) {
        const auto& read_group_ids_unordered = read_neuron_id_to_group_ids_unordered[neuron_id];
        if (!ranges::contains(ids_set, NeuronID(neuron_id))) {
            ASSERT_EQ(read_group_ids_unordered, RelearnTypes::group_ids_unordered{ Constants::default_group_id }); // if id was not in chosen ids, it should only have the default group (filled in by NeuronIO::read_neuron_groups)
            continue;
        }
        ASSERT_EQ(read_group_ids_unordered, golden_neuron_id_to_group_ids_unordered[neuron_id]); // if id is in chosen ids it should have been written and read normally
    }
    ASSERT_EQ(read_group_id_to_group_name, golden_group_id_to_group_name);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOWriteNeuronGroupsSpecificNeuronsEmptyPath) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);

    const auto number_neurons = translator->get_number_neurons_in_total();

    const auto ids = NeuronIDRange::range(number_neurons) | ranges::to_vector;

    const auto bad_path = std::filesystem::path{ "" };

    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_groups_of_specific_neurons(ids, bad_path, translator), RelearnException);
    std::filesystem::remove(bad_path);
}

TEST_F(IOTest, testNeuronIOWriteGroupNames) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);

    auto sstream = std::stringstream{};

    NeuronIO::write_group_names(sstream, translator);

    auto golden_group_id = 0UL;
    for (const auto& line : ranges::getlines(sstream)) {
        if (line.starts_with('#')) {
            continue;
        }

        auto group_id = RelearnTypes::group_id{};
        auto group_name = RelearnTypes::group_name{};
        auto number_neurons_in_group = RelearnTypes::number_neurons_type{};

        auto ss_line = std::stringstream(line);

        const auto success = (ss_line >> group_id) && (ss_line >> group_name) && (ss_line >> number_neurons_in_group);

        ASSERT_TRUE(success);

        const auto& golden_group_name = translator->get_group_name_for_group_id(golden_group_id);
        const auto& golden_number_neurons_in_group = translator->get_number_neurons_in_group(golden_group_id);

        ASSERT_EQ(golden_group_id, group_id);
        ASSERT_EQ(golden_group_name, group_name);
        ASSERT_EQ(golden_number_neurons_in_group, number_neurons_in_group);

        golden_group_id++;
    }
}

TEST_F(IOTest, testNeuronIOWriteGroupNameToFileName) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(mt);

    auto sstream = std::stringstream{};

    NeuronIO::write_group_name_to_file_name(sstream, translator);

    auto currently_observed_group_id = 0UL;
    for (const auto& line : ranges::getlines(sstream)) {
        if (line.starts_with('#')) {
            continue;
        }

        auto group_name = RelearnTypes::group_name{};
        auto file_name = std::string{};

        auto ss_line = std::stringstream(line);

        const auto success = (ss_line >> group_name) && (ss_line >> file_name);

        ASSERT_TRUE(success);

        const auto& golden_group_name = translator->get_group_name_for_group_id(currently_observed_group_id);
        const auto golden_file_name = fmt::format("0_group_{}.csv", currently_observed_group_id);

        ASSERT_EQ(golden_group_name, group_name);
        ASSERT_EQ(golden_file_name, file_name);

        currently_observed_group_id++;
    }
}

TEST_F(IOTest, testNeuronIOReadNeuronsAllInformation) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_neurons.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt), NeuronID{ false, neuron_id }, NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(number_neurons, mt);

    const auto& group_id_to_group_name = translator->get_all_group_names();

    const auto& neuron_id_to_group_ids_unordered = translator->get_neuron_ids_to_group_ids_unordered();
    const auto& neuron_id_to_group_names_unordered = NeuronsFactory::get_neuron_id_vs_group_names_unordered(neuron_id_to_group_ids_unordered, group_id_to_group_name);

    const auto path_positions_signals = std::filesystem::path{ "./neurons14.tmp" };
    const auto path_groups = std::filesystem::path{ "./groups7.tmp" };

    const auto paths = NeuronFilePaths{ path_positions_signals, path_groups };

    ASSERT_NO_THROW(NeuronIO::write_neurons(preliminary_neurons, paths, translator));

    const auto& [read_neurons, read_neuron_id_to_group_ids, read_group_id_to_group_name, additional_infos, _]
        = NeuronIO::read_neurons_all_information(paths);

    ASSERT_EQ(read_neurons.size(), preliminary_neurons.size());

    auto minimum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    auto maximum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    auto number_excitatory = 0;
    auto number_inhibitory = 0;

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& preliminary_neuron = preliminary_neurons[neuron_id];
        const auto& read_neuron = read_neurons[neuron_id];

        const auto& diff = preliminary_neuron.pos - read_neuron.pos;
        const auto norm = diff.calculate_2_norm();

        // NeuronIO writes a position with as many digits as space_type carries, so it comes back accurate
        // relative to its magnitude and not to an absolute epsilon.
        ASSERT_NEAR(0.0, norm, tolerance_for<RelearnTypes::space_type>(preliminary_neuron.pos.calculate_2_norm()));

        ASSERT_EQ(read_neuron.id, preliminary_neuron.id);
        ASSERT_EQ(read_neuron.signal_type, preliminary_neuron.signal_type);

        minimum.calculate_componentwise_minimum(read_neuron.pos);
        maximum.calculate_componentwise_maximum(read_neuron.pos);

        if (read_neuron.signal_type == SignalType::Excitatory) {
            number_excitatory++;
        } else {
            number_inhibitory++;
        }
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);

    const auto& read_neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(read_neuron_id_to_group_ids);
    const auto& read_neuron_id_to_group_names_unordered = NeuronsFactory::get_neuron_id_vs_group_names_unordered(read_neuron_id_to_group_ids_unordered, read_group_id_to_group_name);

    ASSERT_EQ(neuron_id_to_group_names_unordered, read_neuron_id_to_group_names_unordered);

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& actual_group_names = neuron_id_to_group_names_unordered[neuron_id];
        const auto& read_group_names = read_neuron_id_to_group_names_unordered[neuron_id];
        ASSERT_TRUE(ranges::contains(read_group_names, std::string{ Constants::default_group_name }));
        for (const auto& read_group_name : read_group_names) {
            ASSERT_TRUE(ranges::contains(actual_group_names, read_group_name));
        }
    }

    ASSERT_EQ(read_group_id_to_group_name[Constants::default_group_id], std::string{ Constants::default_group_name });

    std::filesystem::remove(path_positions_signals);
    std::filesystem::remove(path_groups);
}

TEST_F(IOTest, testNeuronIOReadNeuronsAllInformationComponentwise) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(number_neurons, mt);

    const auto& group_id_to_group_name = translator->get_all_group_names();

    const auto& neuron_id_to_group_ids_unordered = translator->get_neuron_ids_to_group_ids_unordered();
    const auto& neuron_id_to_group_names_unordered = NeuronsFactory::get_neuron_id_vs_group_names_unordered(neuron_id_to_group_ids_unordered, group_id_to_group_name);

    const auto path_positions_signals = std::filesystem::path{ "./neurons15.tmp" };
    const auto path_groups = std::filesystem::path{ "./groups8.tmp" };

    const auto paths = NeuronFilePaths{ path_positions_signals, path_groups };

    ASSERT_NO_THROW(NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, translator, preliminary_signal_types, paths));

    const auto& [read_ids, read_positions, read_neuron_id_to_group_ids, read_group_id_to_group_name, read_signal_types, additional_infos]
        = NeuronIO::read_neurons_all_information_componentwise(paths);

    ASSERT_EQ(preliminary_ids, read_ids);
    ASSERT_EQ(preliminary_signal_types, read_signal_types);

    ASSERT_EQ(preliminary_position.size(), read_positions.size());

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& diff = preliminary_position[neuron_id] - read_positions[neuron_id];
        const auto norm = diff.calculate_2_norm();

        // NeuronIO writes a position with as many digits as space_type carries, so it comes back accurate
        // relative to its magnitude and not to an absolute epsilon.
        ASSERT_NEAR(0.0, norm, tolerance_for<RelearnTypes::space_type>(preliminary_position[neuron_id].calculate_2_norm()));
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    const auto number_excitatory = ranges::count(preliminary_signal_types, SignalType::Excitatory);
    const auto number_inhibitory = ranges::count(preliminary_signal_types, SignalType::Inhibitory);

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    auto minimum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    auto maximum = RelearnTypes::position_type(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    for (const auto& position : read_positions) {
        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);
    }

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);

    const auto& read_neuron_id_to_group_ids_unordered = NeuronsFactory::get_neuron_id_to_group_ids_unordered(read_neuron_id_to_group_ids);
    const auto& read_neuron_id_to_group_names_unordered = NeuronsFactory::get_neuron_id_vs_group_names_unordered(read_neuron_id_to_group_ids_unordered, read_group_id_to_group_name);

    ASSERT_EQ(neuron_id_to_group_names_unordered, read_neuron_id_to_group_names_unordered);

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& actual_group_names = neuron_id_to_group_names_unordered[neuron_id];
        const auto& read_group_names = read_neuron_id_to_group_names_unordered[neuron_id];
        ASSERT_TRUE(ranges::contains(read_group_names, std::string{ Constants::default_group_name }));
        for (const auto& read_group_name : read_group_names) {
            ASSERT_TRUE(ranges::contains(actual_group_names, read_group_name));
        }
    }

    ASSERT_EQ(read_group_id_to_group_name[Constants::default_group_id], std::string{ Constants::default_group_name });

    std::filesystem::remove(path_positions_signals);
    std::filesystem::remove(path_groups);
}

TEST_F(IOTest, testNeuronIOWriteNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));

        preliminary_neurons.emplace_back(preliminary_position[neuron_id], preliminary_ids[neuron_id], preliminary_signal_types[neuron_id]);
    }

    const auto path_positions_signals = std::filesystem::path{ "./neurons16.tmp" };
    const auto path_groups = std::filesystem::path{ "./groups9.tmp" };

    const auto paths = NeuronFilePaths{ path_positions_signals, path_groups };

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(number_neurons, mt);

    ASSERT_NO_THROW(NeuronIO::write_neurons(preliminary_neurons, paths, translator));

    std::filesystem::remove(path_positions_signals);
    std::filesystem::remove(path_groups);
}

TEST_F(IOTest, testNeuronIOWriteNeuronsComponentwise) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));

        preliminary_neurons.emplace_back(preliminary_position[neuron_id], preliminary_ids[neuron_id], preliminary_signal_types[neuron_id]);
    }

    const auto path_positions_signals = std::filesystem::path{ "./neurons17.tmp" };
    const auto path_groups = std::filesystem::path{ "./groups10.tmp" };

    const auto paths = NeuronFilePaths{ path_positions_signals, path_groups };

    const auto translator = LocalGroupTranslatorFactory::get_randomized_group_translator(number_neurons, mt);

    ASSERT_NO_THROW(NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, translator, preliminary_signal_types, paths));

    std::filesystem::remove(path_positions_signals);
    std::filesystem::remove(path_groups);
}

TEST_F(IOTest, testNeuronIOReadIDs) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto path = std::filesystem::path{ "./neurons18.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);

    const auto& read_ids = NeuronIO::read_neuron_ids(path);

    ASSERT_EQ(read_ids, preliminary_ids);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadIDsEmpty1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = RandomFactory::get_random_integer<RelearnTypes::number_neurons_type>(2, NeuronIdFactory::upper_bound_num_neurons, mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto idx1 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 1, mt);
    auto idx2 = RandomFactory::get_random_integer<std::size_t>(0, number_neurons - 2, mt);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(preliminary_ids[idx1], preliminary_ids[idx2]);

    const auto path = std::filesystem::path{ "./neurons19.tmp" };

    NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path);

    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadIDsEmpty2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = SimulationFactory::get_maximum_position();

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        preliminary_ids.emplace_back(false, neuron_id);
        preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
        preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    preliminary_ids.emplace_back(false, number_neurons + 1);
    preliminary_position.emplace_back(SimulationFactory::get_random_position_in_box(min_pos, max_pos, mt));
    preliminary_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));

    const auto path = std::filesystem::path{ "./neurons20.tmp" };

    ASSERT_THROW_NO_PRINT(NeuronIO::write_neuron_positions_and_signals_componentwise(preliminary_ids, preliminary_position, preliminary_signal_types, path), RelearnException);
    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadIDsFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path = std::filesystem::path{ "" };
    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadCommentsFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path = std::filesystem::path{ "" };
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_comments(path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadComments1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path = std::filesystem::path{ "./comments0.tmp" };

    {
        auto out_file = std::ofstream{ path };
        out_file << "# 1\n# 2\n# #\n#";
    }

    const auto& comments = NeuronIO::read_comments(path);

    ASSERT_EQ(comments.size(), 4);
    ASSERT_EQ(comments[0], std::string("# 1"));
    ASSERT_EQ(comments[1], std::string("# 2"));
    ASSERT_EQ(comments[2], std::string("# #"));
    ASSERT_EQ(comments[3], std::string("#"));
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadComments2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path = std::filesystem::path{ "./comments1.tmp" };

    {
        auto out_file = std::ofstream{ path };
        out_file << "Hallo\n# 1\n# 2\n# #\n#";
    }

    const auto& comments = NeuronIO::read_comments(path);

    ASSERT_TRUE(comments.empty());
    std::filesystem::remove(path);
}

TEST_F(IOTest, testNeuronIOReadComments3) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path = std::filesystem::path{ "./comments2.tmp" };

    {
        auto out_file = std::ofstream{ path };
        for (auto i = 0; i < 10; i++) {
            out_file << "# 1\n";
            out_file << "Hallo\n";
        }
    }

    const auto& comments = NeuronIO::read_comments(path);

    ASSERT_EQ(comments.size(), 1);
    ASSERT_EQ(comments[0], std::string("# 1"));
    std::filesystem::remove(path);
}

TEST_F(IOTest, testReadInSynapsesFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path = std::filesystem::path{ "" };
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_in_synapses(path, 1, mpiPP::MPIRank(1), 2), RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testReadInSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt);
    const auto my_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);

    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_local_synapses_static = StaticLocalSynapses{};
    auto preliminary_distant_synapses_static = StaticDistantInSynapses{};
    auto preliminary_local_synapses_plastic = PlasticLocalSynapses{};
    auto preliminary_distant_synapses_plastic = PlasticDistantInSynapses{};

    const auto path = std::filesystem::path{ "./in_network0.tmp" };
    auto ofstream = std::ofstream(path);

    for ([[maybe_unused]] const auto synapse_id : ranges::views::indices(number_synapses)) {
        const auto& source_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto& target_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);

        const auto source_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);

        const auto plastic_weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);
        const auto static_weight = SynapsesFactory::get_random_static_synapse_weight(mt);

        const bool plastic = RandomFactory::get_random_bool(mt);
        const char flag = plastic ? '1' : '0';

        if (source_rank == my_rank) {
            if (plastic) {
                preliminary_local_synapses_plastic.emplace_back(target_id, source_id, plastic_weight);
            } else {
                preliminary_local_synapses_static.emplace_back(target_id, source_id, static_weight);
            }
        } else {
            if (plastic) {
                preliminary_distant_synapses_plastic.emplace_back(target_id, RankNeuronId(source_rank, source_id), plastic_weight);
            } else {
                preliminary_distant_synapses_static.emplace_back(target_id, RankNeuronId(source_rank, source_id), static_weight);
            }
        }

        if (plastic) {
            ofstream << my_rank.get_rank() << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                     << source_rank.get_rank() << ' ' << (source_id.get_neuron_id() + 1) << ' ' << plastic_weight << '\t' << flag << '\n';
        } else {
            ofstream << my_rank.get_rank() << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                     << source_rank.get_rank() << ' ' << (source_id.get_neuron_id() + 1) << ' ' << static_weight << '\t' << flag << '\n';
        }
    }

    ofstream.flush();
    ofstream.close();

    auto [synapses_static, synapses_plastic] = NeuronIO::read_in_synapses(path, number_neurons, my_rank, number_ranks);
    auto [read_local_synapses_plastic, read_distant_synapses_plastic] = synapses_plastic;
    auto [read_local_synapses_static, read_distant_synapses_static] = synapses_static;

    std::ranges::sort(preliminary_local_synapses_static);
    std::ranges::sort(preliminary_distant_synapses_static);
    std::ranges::sort(preliminary_local_synapses_plastic);
    std::ranges::sort(preliminary_distant_synapses_plastic);

    std::ranges::sort(read_local_synapses_static);
    std::ranges::sort(read_distant_synapses_static);
    std::ranges::sort(read_local_synapses_plastic);
    std::ranges::sort(read_distant_synapses_plastic);

    ASSERT_EQ(preliminary_local_synapses_static.size(), read_local_synapses_static.size());
    ASSERT_EQ(preliminary_distant_synapses_static.size(), read_distant_synapses_static.size());
    ASSERT_EQ(preliminary_local_synapses_plastic.size(), read_local_synapses_plastic.size());
    ASSERT_EQ(preliminary_distant_synapses_plastic.size(), read_distant_synapses_plastic.size());

    for (auto i = 0U; i < preliminary_local_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_distant_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_local_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_distant_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }
    std::filesystem::remove(path);
}

TEST_F(IOTest, testReadOutSynapsesFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path = std::filesystem::path{ "" };
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIO::read_out_synapses(path, 1, mpiPP::MPIRank(1), 2), RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testReadOutSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt);
    const auto my_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);

    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_local_synapses_static = StaticLocalSynapses{};
    auto preliminary_distant_synapses_static = StaticDistantOutSynapses{};
    auto preliminary_local_synapses_plastic = PlasticLocalSynapses{};
    auto preliminary_distant_synapses_plastic = PlasticDistantOutSynapses{};

    const auto path = std::filesystem::path{ "./out_network0.tmp" };
    auto ofstream = std::ofstream(path);

    for ([[maybe_unused]] const auto synapse_id : ranges::views::indices(number_synapses)) {
        const auto& source_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto& target_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);

        const auto target_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);

        const auto plastic_weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);
        const auto static_weight = SynapsesFactory::get_random_static_synapse_weight(mt);

        const bool plastic = RandomFactory::get_random_bool(mt);
        const char flag = plastic ? '1' : '0';

        if (target_rank == my_rank) {
            if (plastic) {
                preliminary_local_synapses_plastic.emplace_back(target_id, source_id, plastic_weight);
            } else {
                preliminary_local_synapses_static.emplace_back(target_id, source_id, static_weight);
            }
        } else {
            if (plastic) {
                preliminary_distant_synapses_plastic.emplace_back(RankNeuronId(target_rank, target_id), source_id, plastic_weight);
            } else {
                preliminary_distant_synapses_static.emplace_back(RankNeuronId(target_rank, target_id), source_id, static_weight);
            }
        }

        if (plastic) {
            ofstream << target_rank.get_rank() << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                     << my_rank.get_rank() << ' ' << (source_id.get_neuron_id() + 1) << ' ' << plastic_weight << '\t' << flag << '\n';
        } else {
            ofstream << target_rank.get_rank() << ' ' << (target_id.get_neuron_id() + 1) << '\t'
                     << my_rank.get_rank() << ' ' << (source_id.get_neuron_id() + 1) << ' ' << static_weight << '\t' << flag << '\n';
        }
    }

    ofstream.flush();
    ofstream.close();

    auto [synapses_static, synapses_plastic] = NeuronIO::read_out_synapses(path, number_neurons, my_rank, number_ranks);
    auto [read_local_synapses_plastic, read_distant_synapses_plastic] = synapses_plastic;
    auto [read_local_synapses_static, read_distant_synapses_static] = synapses_static;

    std::ranges::sort(preliminary_local_synapses_static);
    std::ranges::sort(preliminary_distant_synapses_static);
    std::ranges::sort(preliminary_local_synapses_plastic);
    std::ranges::sort(preliminary_distant_synapses_plastic);

    std::ranges::sort(read_local_synapses_static);
    std::ranges::sort(read_distant_synapses_static);
    std::ranges::sort(read_local_synapses_plastic);
    std::ranges::sort(read_distant_synapses_plastic);

    ASSERT_EQ(preliminary_local_synapses_static.size(), read_local_synapses_static.size());
    ASSERT_EQ(preliminary_distant_synapses_static.size(), read_distant_synapses_static.size());
    ASSERT_EQ(preliminary_local_synapses_plastic.size(), read_local_synapses_plastic.size());
    ASSERT_EQ(preliminary_distant_synapses_plastic.size(), read_distant_synapses_plastic.size());

    for (auto i = 0U; i < preliminary_local_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_distant_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_local_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_distant_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }
    std::filesystem::remove(path);
}

TEST_F(IOTest, testWriteInSynapsesFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path = std::filesystem::path{ "" };
    ASSERT_THROW_NO_PRINT(NeuronIO::write_in_synapses({}, {}, {}, {}, mpiPP::MPIRank(0), 0, path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testWriteInSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt);
    const auto my_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);

    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_local_synapses_static = StaticLocalSynapses{};
    auto preliminary_distant_synapses_static = StaticDistantInSynapses{};
    auto preliminary_local_synapses_plastic = PlasticLocalSynapses{};
    auto preliminary_distant_synapses_plastic = PlasticDistantInSynapses{};

    const auto path = std::filesystem::path{ "./in_network1.tmp" };

    for ([[maybe_unused]] const auto synapse_id : ranges::views::indices(number_synapses)) {
        const auto& source_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto& target_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);

        const auto source_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);

        const auto plastic_weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);
        const auto static_weight = SynapsesFactory::get_random_static_synapse_weight(mt);

        const bool plastic = RandomFactory::get_random_bool(mt);

        if (source_rank == my_rank) {
            if (plastic) {
                preliminary_local_synapses_plastic.emplace_back(target_id, source_id, plastic_weight);
            } else {
                preliminary_local_synapses_static.emplace_back(target_id, source_id, static_weight);
            }
        } else {
            if (plastic) {
                preliminary_distant_synapses_plastic.emplace_back(target_id, RankNeuronId(source_rank, source_id), plastic_weight);
            } else {
                preliminary_distant_synapses_static.emplace_back(target_id, RankNeuronId(source_rank, source_id), static_weight);
            }
        }
    }

    NeuronIO::write_in_synapses(preliminary_local_synapses_static, preliminary_distant_synapses_static, preliminary_local_synapses_plastic, preliminary_distant_synapses_plastic, my_rank, number_neurons, path);

    auto [synapses_static, synapses_plastic] = NeuronIO::read_in_synapses(path, number_neurons, my_rank, number_ranks);
    auto [read_local_synapses_plastic, read_distant_synapses_plastic] = synapses_plastic;
    auto [read_local_synapses_static, read_distant_synapses_static] = synapses_static;

    std::ranges::sort(preliminary_local_synapses_static);
    std::ranges::sort(preliminary_distant_synapses_static);
    std::ranges::sort(preliminary_local_synapses_plastic);
    std::ranges::sort(preliminary_distant_synapses_plastic);

    std::ranges::sort(read_local_synapses_static);
    std::ranges::sort(read_distant_synapses_static);
    std::ranges::sort(read_local_synapses_plastic);
    std::ranges::sort(read_distant_synapses_plastic);

    ASSERT_EQ(preliminary_local_synapses_static.size(), read_local_synapses_static.size());
    ASSERT_EQ(preliminary_distant_synapses_static.size(), read_distant_synapses_static.size());
    ASSERT_EQ(preliminary_local_synapses_plastic.size(), read_local_synapses_plastic.size());
    ASSERT_EQ(preliminary_distant_synapses_plastic.size(), read_distant_synapses_plastic.size());

    for (auto i = 0U; i < preliminary_local_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_distant_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_local_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_distant_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }
    std::filesystem::remove(path);
}

TEST_F(IOTest, testWriteOutSynapsesFileNotFound) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto path = std::filesystem::path{ "" };
    ASSERT_THROW_NO_PRINT(NeuronIO::write_out_synapses({}, {}, {}, {}, mpiPP::MPIRank(0), 0, path);, RelearnException);
    std::filesystem::remove(path);
}

TEST_F(IOTest, testWriteOutSynapses) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_ranks = MPIRankFactory::get_random_number_ranks(mt);
    const auto my_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);

    const auto number_synapses = SynapsesFactory::get_random_number_synapses(mt);
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);

    auto preliminary_local_synapses_static = StaticLocalSynapses{};
    auto preliminary_distant_synapses_static = StaticDistantOutSynapses{};
    auto preliminary_local_synapses_plastic = PlasticLocalSynapses{};
    auto preliminary_distant_synapses_plastic = PlasticDistantOutSynapses{};

    const auto path = std::filesystem::path{ "./out_network1.tmp" };

    for ([[maybe_unused]] const auto synapse_id : ranges::views::indices(number_synapses)) {
        const auto& source_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);
        const auto& target_id = NeuronIdFactory::get_random_neuron_id(number_neurons, mt);

        const auto target_rank = MPIRankFactory::get_random_mpi_rank(number_ranks, mt);

        const auto plastic_weight = SynapsesFactory::get_random_plastic_synapse_weight(mt);
        const auto static_weight = SynapsesFactory::get_random_static_synapse_weight(mt);

        const bool plastic = RandomFactory::get_random_bool(mt);

        if (target_rank == my_rank) {
            if (plastic) {
                preliminary_local_synapses_plastic.emplace_back(target_id, source_id, plastic_weight);
            } else {
                preliminary_local_synapses_static.emplace_back(target_id, source_id, static_weight);
            }
        } else {
            if (plastic) {
                preliminary_distant_synapses_plastic.emplace_back(RankNeuronId(target_rank, target_id), source_id, plastic_weight);
            } else {
                preliminary_distant_synapses_static.emplace_back(RankNeuronId(target_rank, target_id), source_id, static_weight);
            }
        }
    }

    NeuronIO::write_out_synapses(preliminary_local_synapses_static, preliminary_distant_synapses_static, preliminary_local_synapses_plastic, preliminary_distant_synapses_plastic, my_rank, number_neurons, path);

    auto [synapses_static, synapses_plastic] = NeuronIO::read_out_synapses(path, number_neurons, my_rank, number_ranks);
    auto [read_local_synapses_plastic, read_distant_synapses_plastic] = synapses_plastic;
    auto [read_local_synapses_static, read_distant_synapses_static] = synapses_static;

    std::ranges::sort(preliminary_local_synapses_static);
    std::ranges::sort(preliminary_distant_synapses_static);
    std::ranges::sort(preliminary_local_synapses_plastic);
    std::ranges::sort(preliminary_distant_synapses_plastic);

    std::ranges::sort(read_local_synapses_static);
    std::ranges::sort(read_distant_synapses_static);
    std::ranges::sort(read_local_synapses_plastic);
    std::ranges::sort(read_distant_synapses_plastic);

    ASSERT_EQ(preliminary_local_synapses_static.size(), read_local_synapses_static.size());
    ASSERT_EQ(preliminary_distant_synapses_static.size(), read_distant_synapses_static.size());
    ASSERT_EQ(preliminary_local_synapses_plastic.size(), read_local_synapses_plastic.size());
    ASSERT_EQ(preliminary_distant_synapses_plastic.size(), read_distant_synapses_plastic.size());

    for (auto i = 0U; i < preliminary_local_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_distant_synapses_static.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_static[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_static[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_local_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_local_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_local_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }

    for (auto i = 0U; i < preliminary_distant_synapses_plastic.size(); i++) {
        const auto& [p_1, p_2, p_weight] = preliminary_distant_synapses_plastic[i];
        const auto& [r_1, r_2, r_weight] = read_distant_synapses_plastic[i];

        ASSERT_NEAR(p_weight, r_weight, eps);
    }
    std::filesystem::remove(path);
}

TEST_F(IOTest, additionalPositionInformationTest) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto total_neurons = NeuronIdFactory::get_random_number_neurons(mt) + number_neurons;
    const auto num_subdomains = RandomFactory::get_random_integer(1U, 10U, mt);

    auto correct_ids = std::vector<NeuronID>{};
    auto correct_position = std::vector<RelearnTypes::position_type>{};
    auto correct_group_ids = std::vector<RelearnTypes::group_id>{};
    auto correct_group_names = std::vector<RelearnTypes::group_name>{};
    auto correct_signal_types = std::vector<SignalType>{};

    const auto sim_box = SimulationFactory::get_random_simulation_box_size(mt);
    auto subdomain_boxes = std::vector<RelearnTypes::bounding_box_type>{};
    for (auto i = 0U; i < num_subdomains; i++) {
        subdomain_boxes.push_back(SimulationFactory::get_random_simulation_box_size(mt));
    }

    for (auto i = 0U; i < number_neurons; i++) {
        correct_ids.emplace_back(i);
        correct_position.emplace_back(SimulationFactory::get_random_position_in_box(sim_box, mt));
        correct_group_names.emplace_back("group_" + std::to_string(i));
        correct_group_ids.emplace_back(i);
        correct_signal_types.emplace_back(NeuronTypesFactory::get_random_signal_type(mt));
    }

    const auto path = std::filesystem::current_path() / "neurons33.tmp";

    NeuronIO::write_neuron_positions_and_signals_componentwise(correct_ids, correct_position, correct_signal_types, path, total_neurons, sim_box, subdomain_boxes);

    const auto comments = NeuronIO::read_comments(path);
    const auto infos = NeuronIO::parse_additional_position_information(comments);

    ASSERT_EQ(infos.total_neurons, total_neurons);
    ASSERT_EQ(infos.local_neurons, number_neurons);
    // NeuronIO writes the boundaries with as many digits as space_type carries, so they come back accurate
    // relative to their magnitude and not to an absolute epsilon.
    const auto box_tolerance = [](const RelearnTypes::bounding_box_type& box) {
        const auto magnitude = std::max(box.get_minimum().calculate_2_norm(), box.get_maximum().calculate_2_norm());
        return utility::cast<RelearnTypes::space_type>(tolerance_for<RelearnTypes::space_type>(magnitude));
    };

    ASSERT_TRUE(sim_box.almost_equal(infos.sim_size, box_tolerance(sim_box)));
    ASSERT_EQ(subdomain_boxes.size(), infos.subdomain_sizes.size());

    for (auto i = 0U; i < subdomain_boxes.size(); i++) {
        const auto& bb1 = subdomain_boxes[i];
        const auto& bb2 = infos.subdomain_sizes[i];

        ASSERT_TRUE(bb1.almost_equal(bb2, box_tolerance(bb1)));
    }
    std::filesystem::remove(path);
}
