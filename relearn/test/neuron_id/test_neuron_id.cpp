/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_id.h"

#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"

#include "factory/random/random_factory.h"

#include <fmt/core.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <compare>
#include <cstdint>
#include <iostream>
#include <limits>

TEST_F(NeuronIDTest, testUninitialized) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id = NeuronID::uninitialized_id();

    ASSERT_FALSE(id.is_initialized());
    ASSERT_FALSE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_FALSE(id.is_actual_id());

    ASSERT_THROW_NO_PRINT(std::ignore = id.get_neuron_id(), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = id.get_rma_offset(), RelearnException);
}

TEST_F(NeuronIDTest, testVirtual) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id = NeuronID::virtual_id();

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_TRUE(id.is_virtual());
    ASSERT_FALSE(id.is_actual_id());

    ASSERT_THROW_NO_PRINT(std::ignore = id.get_neuron_id(), RelearnException);
    ASSERT_EQ(id.get_rma_offset(), 0);
    ASSERT_EQ(static_cast<std::uint64_t>(id), 0);
}

TEST_F(NeuronIDTest, testHijackedVirtual) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id = NeuronID::virtual_id(135);

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_TRUE(id.is_virtual());
    ASSERT_FALSE(id.is_actual_id());

    ASSERT_THROW_NO_PRINT(std::ignore = id.get_neuron_id(), RelearnException);
    ASSERT_EQ(id.get_rma_offset(), 135);
    ASSERT_EQ(static_cast<std::uint64_t>(id), 135);
}

TEST_F(NeuronIDTest, testConstructorDefault) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id = NeuronID{};

    ASSERT_FALSE(id.is_initialized());
    ASSERT_FALSE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_FALSE(id.is_actual_id());

    ASSERT_THROW_NO_PRINT(std::ignore = id.get_neuron_id(), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = id.get_rma_offset(), RelearnException);
}

TEST_F(NeuronIDTest, testConstructorOnlyID) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto test = [](const auto id_val) {
        const auto id = NeuronID{ id_val };

        ASSERT_TRUE(id.is_initialized());
        ASSERT_TRUE(static_cast<bool>(id));
        ASSERT_FALSE(id.is_virtual());
        ASSERT_TRUE(id.is_actual_id());

        ASSERT_THROW_NO_PRINT(std::ignore = id.get_rma_offset(), RelearnException);
        ASSERT_EQ(id.get_neuron_id(), id_val);
        ASSERT_EQ(static_cast<std::uint64_t>(id), id_val);
    };

    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 1 });
    test(std::uint64_t{ 65537 });
    test(std::uint64_t{ 102255410 });
    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 1 });
    test(std::uint64_t{ 0 });
}

TEST_F(NeuronIDTest, testConstructorLocal) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto test = [](const auto id_val) {
        const auto id = NeuronID{ false, id_val };

        ASSERT_TRUE(id.is_initialized());
        ASSERT_TRUE(static_cast<bool>(id));
        ASSERT_FALSE(id.is_virtual());
        ASSERT_TRUE(id.is_actual_id());

        ASSERT_THROW_NO_PRINT(std::ignore = id.get_rma_offset(), RelearnException);
        ASSERT_EQ(id.get_neuron_id(), id_val);
        ASSERT_EQ(static_cast<std::uint64_t>(id), id_val);
    };

    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 1 });
    test(std::uint64_t{ 65537 });
    test(std::uint64_t{ 102255410 });
    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 1 });
    test(std::uint64_t{ 0 });
}

TEST_F(NeuronIDTest, testConstructorVirtual) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto test = [](const auto id_val) {
        const auto id = NeuronID{ true, id_val };

        ASSERT_TRUE(id.is_initialized());
        ASSERT_TRUE(static_cast<bool>(id));
        ASSERT_TRUE(id.is_virtual());
        ASSERT_FALSE(id.is_actual_id());

        ASSERT_THROW_NO_PRINT(std::ignore = id.get_neuron_id(), RelearnException);
        ASSERT_EQ(id.get_rma_offset(), id_val);
        ASSERT_EQ(static_cast<std::uint64_t>(id), id_val);
    };

    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 1 });
    test(std::uint64_t{ 65537 });
    test(std::uint64_t{ 102255410 });
    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 1 });
    test(std::uint64_t{ 0 });
}

TEST_F(NeuronIDTest, testConstructorLimits) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    constexpr auto largest = NeuronID::limits::max;

    ASSERT_EQ(NeuronID{ largest }.get_neuron_id(), largest);
    ASSERT_EQ(NeuronID(false, largest).get_neuron_id(), largest);
    ASSERT_EQ(NeuronID::virtual_id(largest).get_rma_offset(), largest);

    // The id values are checked instead of being silently truncated to the 62 bits that the flags leave over
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronID{ largest + 1 }, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronID{ std::numeric_limits<NeuronID::value_type>::max() }, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronID{ -1 }, RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronID(true, largest + 1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronID(true, -1), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronID::virtual_id(largest + 1), RelearnException);
}

TEST_F(NeuronIDTest, testTaggedID) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto local = NeuronID{ 42 };
    const auto virtual_neuron = NeuronID::virtual_id(42);
    const auto uninitialized = NeuronID::uninitialized_id();

    // The wrapped tagged id round-trips through the constructor that takes it
    ASSERT_EQ(NeuronID{ local.get_id() }, local);
    ASSERT_EQ(NeuronID{ virtual_neuron.get_id() }, virtual_neuron);
    ASSERT_EQ(NeuronID{ uninitialized.get_id() }, uninitialized);

    ASSERT_TRUE(local.get_id().get_flag<NeuronIDTraits::initialized_flag>());
    ASSERT_FALSE(local.get_id().get_flag<NeuronIDTraits::virtual_flag>());
    ASSERT_EQ(local.get_id().get_value(), 42);

    ASSERT_TRUE(virtual_neuron.get_id().get_flag<NeuronIDTraits::initialized_flag>());
    ASSERT_TRUE(virtual_neuron.get_id().get_flag<NeuronIDTraits::virtual_flag>());
    ASSERT_EQ(virtual_neuron.get_id().get_value(), 42);

    ASSERT_FALSE(uninitialized.get_id().get_flag<NeuronIDTraits::initialized_flag>());
    ASSERT_FALSE(uninitialized.get_id().get_flag<NeuronIDTraits::virtual_flag>());

    // The tagged id prints with its name, which is what shows up in the messages of the exceptions it throws
    ASSERT_EQ(fmt::format("{}", local.get_id()), "NeuronID: 42");
    ASSERT_EQ(fmt::format("{}", uninitialized.get_id()), "NeuronID: uninitialized");
}

TEST_F(NeuronIDTest, testRange1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto test = [](const auto id_val_1, const auto id_val_2) {
        const auto range = NeuronIDRange::range(id_val_1, id_val_2);

        auto expected = id_val_1;
        for (const auto id : range) {
            const auto expected_id = NeuronID{ false, expected };
            ASSERT_EQ(id, expected_id);
            expected++;
        }

        ASSERT_EQ(expected, id_val_2);
    };

    test(std::uint64_t{ 0 }, std::uint64_t{ 0 });
    test(std::uint64_t{ 1 }, std::uint64_t{ 1 });
    test(std::uint64_t{ 0 }, std::uint64_t{ 1 });
    test(std::uint64_t{ 0 }, std::uint64_t{ 357 });
    test(std::uint64_t{ 157 }, std::uint64_t{ 357 });
    test(std::uint64_t{ 157 }, std::uint64_t{ 157 });
}

TEST_F(NeuronIDTest, testRange2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto test = [](const auto id_val_1, const auto id_val_2) {
        const auto range = NeuronIDRange::range(NeuronID{ false, id_val_1 }, NeuronID{ false, id_val_2 });

        auto expected = id_val_1;
        for (const auto id : range) {
            const auto expected_id = NeuronID{ false, expected };
            ASSERT_EQ(id, expected_id);
            expected++;
        }

        ASSERT_EQ(expected, id_val_2);
    };

    test(std::uint64_t{ 0 }, std::uint64_t{ 0 });
    test(std::uint64_t{ 1 }, std::uint64_t{ 1 });
    test(std::uint64_t{ 0 }, std::uint64_t{ 1 });
    test(std::uint64_t{ 0 }, std::uint64_t{ 357 });
    test(std::uint64_t{ 157 }, std::uint64_t{ 357 });
    test(std::uint64_t{ 157 }, std::uint64_t{ 157 });
}

TEST_F(NeuronIDTest, testRange3) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto test = [](const auto id_val) {
        const auto range = NeuronIDRange::range(std::uint64_t{ 0 }, id_val);

        auto expected = std::uint64_t{ 0 };
        for (const auto id : range) {
            const auto expected_id = NeuronID{ false, expected };
            ASSERT_EQ(id, expected_id);
            expected++;
        }

        ASSERT_EQ(expected, id_val);
    };

    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 1 });
    test(std::uint64_t{ 10 });
    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 357 });
    test(std::uint64_t{ 157 });
}

TEST_F(NeuronIDTest, testRange4) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto test = [](const auto id_val) {
        const auto range = NeuronIDRange::range(NeuronID{ false, 0 }, NeuronID{ false, id_val });

        auto expected = std::uint64_t{ 0 };
        for (const auto id : range) {
            const auto expected_id = NeuronID{ false, expected };
            ASSERT_EQ(id, expected_id);
            expected++;
        }

        ASSERT_EQ(expected, id_val);
    };

    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 1 });
    test(std::uint64_t{ 10 });
    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 357 });
    test(std::uint64_t{ 157 });
}

TEST_F(NeuronIDTest, testRangeId1) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto test = [](const auto id_val_1, const auto id_val_2) {
        const auto range = NeuronIDRange::range_id(id_val_1, id_val_2);

        auto expected = id_val_1;
        for (const auto id : range) {
            ASSERT_EQ(id, expected);
            expected++;
        }

        ASSERT_EQ(expected, id_val_2);
    };

    test(std::uint64_t{ 0 }, std::uint64_t{ 0 });
    test(std::uint64_t{ 1 }, std::uint64_t{ 1 });
    test(std::uint64_t{ 0 }, std::uint64_t{ 1 });
    test(std::uint64_t{ 0 }, std::uint64_t{ 357 });
    test(std::uint64_t{ 157 }, std::uint64_t{ 357 });
    test(std::uint64_t{ 157 }, std::uint64_t{ 157 });
}

TEST_F(NeuronIDTest, testRangeId2) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto test = [](const auto id_val) {
        const auto range = NeuronIDRange::range_id(std::uint64_t{ 0 }, id_val);

        auto expected = std::uint64_t{ 0 };
        for (const auto id : range) {
            ASSERT_EQ(id, expected);
            expected++;
        }

        ASSERT_EQ(expected, id_val);
    };

    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 1 });
    test(std::uint64_t{ 10 });
    test(std::uint64_t{ 0 });
    test(std::uint64_t{ 357 });
    test(std::uint64_t{ 157 });
}

TEST_F(NeuronIDTest, testRangeInvalid) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    // The bounds are checked instead of yielding a range that runs past the largest representable id
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIDRange::range(std::uint64_t{ 5 }, std::uint64_t{ 2 }), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIDRange::range_id(std::uint64_t{ 5 }, std::uint64_t{ 2 }), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIDRange::range(std::numeric_limits<NeuronID::value_type>::max()), RelearnException);
    ASSERT_THROW_NO_PRINT(std::ignore = NeuronIDRange::range_id(std::numeric_limits<NeuronID::value_type>::max()), RelearnException);

    // The largest range is the one that ends past the largest id
    ASSERT_NO_THROW(std::ignore = NeuronIDRange::range(NeuronID::limits::max, NeuronID::limits::max + 1));
}

TEST_F(NeuronIDTest, testComparisons1) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID{ 23 };
    const auto id2 = NeuronID{ 47 };

    ASSERT_EQ(id1 <=> id2, id1.get_neuron_id() <=> id2.get_neuron_id());

    ASSERT_EQ(NeuronID{}, NeuronID{});
    ASSERT_NE(id1, NeuronID{});
    ASSERT_NE(id2, NeuronID{});
    ASSERT_NE(NeuronID{}, id1);
    ASSERT_NE(NeuronID{}, id2);

    ASSERT_EQ(id1, id1);
    ASSERT_EQ(id1, NeuronID{ 23 });
    ASSERT_EQ(NeuronID{ 23 }, id1);
    ASSERT_EQ(NeuronID{ 23 }, NeuronID{ 23 });

    ASSERT_NE(id1, id2);
    ASSERT_NE(NeuronID{ 23 }, id2);
    ASSERT_NE(id1, NeuronID{ 47 });
    ASSERT_NE(NeuronID{ 23 }, NeuronID{ 47 });

    ASSERT_NE(id2, id1);
    ASSERT_NE(id2, NeuronID{ 23 });
    ASSERT_NE(NeuronID{ 47 }, id1);
    ASSERT_NE(NeuronID{ 47 }, NeuronID{ 23 });

    ASSERT_EQ(id2, id2);
    ASSERT_EQ(id2, NeuronID{ 47 });
    ASSERT_EQ(NeuronID{ 47 }, id2);
    ASSERT_EQ(NeuronID{ 47 }, NeuronID{ 47 });
}

TEST_F(NeuronIDTest, testComparisons2) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID(false, 23);
    const auto id2 = NeuronID(false, 47);

    ASSERT_EQ(id1 <=> id2, id1.get_neuron_id() <=> id2.get_neuron_id());
    ASSERT_NE(id1, NeuronID{});
    ASSERT_NE(id2, NeuronID{});
    ASSERT_NE(NeuronID{}, id1);
    ASSERT_NE(NeuronID{}, id2);

    ASSERT_EQ(id1, id1);
    ASSERT_EQ(id1, NeuronID(false, 23));
    ASSERT_EQ(NeuronID(false, 23), id1);
    ASSERT_EQ(NeuronID(false, 23), NeuronID(false, 23));

    ASSERT_NE(id1, id2);
    ASSERT_NE(NeuronID(false, 23), id2);
    ASSERT_NE(id1, NeuronID(false, 47));
    ASSERT_NE(NeuronID(false, 23), NeuronID(false, 47));

    ASSERT_NE(id2, id1);
    ASSERT_NE(id2, NeuronID(false, 23));
    ASSERT_NE(NeuronID(false, 47), id1);
    ASSERT_NE(NeuronID(false, 47), NeuronID(false, 23));

    ASSERT_EQ(id2, id2);
    ASSERT_EQ(id2, NeuronID(false, 47));
    ASSERT_EQ(NeuronID(false, 47), id2);
    ASSERT_EQ(NeuronID(false, 47), NeuronID(false, 47));
}

TEST_F(NeuronIDTest, testComparisons3) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID(false, 23);
    const auto id2 = NeuronID(true, 23);

    ASSERT_NE(id1 <=> id2, std::strong_ordering::equal);
    ASSERT_NE(id1 <=> id2, std::strong_ordering::equivalent);
    ASSERT_NE(id1, NeuronID{});
    ASSERT_NE(id2, NeuronID{});
    ASSERT_NE(NeuronID{}, id1);
    ASSERT_NE(NeuronID{}, id2);

    ASSERT_NE(id1, id2);
    ASSERT_NE(NeuronID(false, 23), id2);
    ASSERT_NE(id1, NeuronID(true, 23));
    ASSERT_NE(NeuronID(false, 23), NeuronID(true, 23));

    ASSERT_NE(id2, id1);
    ASSERT_NE(id2, NeuronID(false, 23));
    ASSERT_NE(NeuronID(true, 23), id1);
    ASSERT_NE(NeuronID(true, 23), NeuronID(false, 23));
}

TEST_F(NeuronIDTest, testHashValue) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID{ false, 18 };
    const auto id2 = NeuronID{ true, 18 };
    const auto id3 = NeuronID{};

    const auto hash_value_1 = id1.hash_value();
    const auto hash_value_2 = id2.hash_value();
    const auto hash_value_3 = id3.hash_value();

    ASSERT_NE(hash_value_1, hash_value_2);
    ASSERT_NE(hash_value_1, hash_value_3);
    ASSERT_NE(hash_value_2, hash_value_3);

    ASSERT_EQ(hash_value_1, std::hash<NeuronID>{}(id1));
    ASSERT_EQ(hash_value_2, std::hash<NeuronID>{}(id2));
    ASSERT_EQ(hash_value_3, std::hash<NeuronID>{}(id3));

    ASSERT_EQ(hash_value_1, std::hash<NeuronID>{}(NeuronID{ false, 18 }));
    ASSERT_EQ(hash_value_2, std::hash<NeuronID>{}(NeuronID{ true, 18 }));
    ASSERT_EQ(hash_value_3, std::hash<NeuronID>{}(NeuronID{}));

    ASSERT_EQ(hash_value_1, hash_value(id1));
    ASSERT_EQ(hash_value_2, hash_value(id2));
    ASSERT_EQ(hash_value_3, hash_value(id3));
}

TEST_F(NeuronIDTest, testPrint1) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID{ false, 89 };
    const auto id2 = NeuronID{ true, 89 };
    const auto id3 = NeuronID{};

    const auto str1 = fmt::format("{}", id1);
    const auto str2 = fmt::format("{}", id2);
    const auto str3 = fmt::format("{}", id3);

    ASSERT_EQ(str1, "89");
    ASSERT_EQ(str2, "10000000000000000089");
    ASSERT_EQ(str3, "18446744073709551615");
}

TEST_F(NeuronIDTest, testPrint2) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID{ false, 89 };
    const auto id2 = NeuronID{ true, 89 };
    const auto id3 = NeuronID{};

    const auto str1 = fmt::format("{:i}", id1);
    const auto str2 = fmt::format("{:i}", id2);
    const auto str3 = fmt::format("{:i}", id3);

    ASSERT_EQ(str1, "89");
    ASSERT_EQ(str2, "10000000000000000089");
    ASSERT_EQ(str3, "18446744073709551615");
}

TEST_F(NeuronIDTest, testPrint3) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID{ false, 89 };
    const auto id2 = NeuronID{ true, 89 };
    const auto id3 = NeuronID{};

    const auto str1 = fmt::format("{:s}", id1);
    const auto str2 = fmt::format("{:s}", id2);
    const auto str3 = fmt::format("{:s}", id3);

    ASSERT_EQ(str1, "10:89");
    ASSERT_EQ(str2, "11:10000000000000000089");
    ASSERT_EQ(str3, "00:18446744073709551615");
}

TEST_F(NeuronIDTest, testPrint4) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID{ false, 89 };
    const auto id2 = NeuronID{ true, 89 };
    const auto id3 = NeuronID{};

    const auto str1 = fmt::format("{:m}", id1);
    const auto str2 = fmt::format("{:m}", id2);
    const auto str3 = fmt::format("{:m}", id3);

    ASSERT_EQ(str1, "i1v0:89");
    ASSERT_EQ(str2, "i1v1:10000000000000000089");
    ASSERT_EQ(str3, "i0v0:18446744073709551615");
}

TEST_F(NeuronIDTest, testPrint5) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID{ false, 89 };
    const auto id2 = NeuronID{ true, 89 };
    const auto id3 = NeuronID{};

    const auto str1 = fmt::format("{:l}", id1);
    const auto str2 = fmt::format("{:l}", id2);
    const auto str3 = fmt::format("{:l}", id3);

    ASSERT_EQ(str1, "initialized: true, virtual: false, id: 89");
    ASSERT_EQ(str2, "initialized: true, virtual: true, id: 10000000000000000089");
    ASSERT_EQ(str3, "initialized: false, virtual: false, id: 18446744073709551615");
}

TEST_F(NeuronIDTest, testPrint6) { // NOLINT
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto id1 = NeuronID{ false, 89 };
    const auto id2 = NeuronID{ true, 89 };
    const auto id3 = NeuronID{};

    auto ss1 = std::stringstream{};
    auto ss2 = std::stringstream{};
    auto ss3 = std::stringstream{};

    ss1 << id1;
    ss2 << id2;
    ss3 << id3;

    const auto str1 = fmt::format("{}", id1);
    const auto str2 = fmt::format("{}", id2);
    const auto str3 = fmt::format("{}", id3);

    ASSERT_EQ(str1, ss1.str());
    ASSERT_EQ(str2, ss2.str());
    ASSERT_EQ(str3, ss3.str());
}
