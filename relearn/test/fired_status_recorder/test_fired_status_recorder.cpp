/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_fired_status_recorder.h"

#include "Config.h"

#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

TEST_F(FiredStatusRecorderTest, testConstructorNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_NO_THROW(std::ignore = FiredStatusRecorder());
}

TEST_F(FiredStatusRecorderTest, testInitNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsr = FiredStatusRecorder{};
    ASSERT_NO_THROW(fsr.init(107));
}

TEST_F(FiredStatusRecorderTest, testInitAndCreate) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsr = FiredStatusRecorder{};

    const auto number_neurons_init = 107;
    const auto number_neurons_create_1 = 23;
    const auto number_neurons_create_2 = 45;

    ASSERT_THROW_NO_PRINT(fsr.init(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.create_neurons(number_neurons_create_1), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.create_neurons(number_neurons_create_2), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.init(0), RelearnException);

    ASSERT_EQ(fsr.get_number_local_neurons(), 0);

    fsr.init(number_neurons_init);

    ASSERT_EQ(fsr.get_number_local_neurons(), number_neurons_init);

    ASSERT_THROW_NO_PRINT(fsr.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.init(0), RelearnException);

    ASSERT_EQ(fsr.get_number_local_neurons(), number_neurons_init);

    fsr.create_neurons(number_neurons_create_1);

    ASSERT_EQ(fsr.get_number_local_neurons(), number_neurons_init + number_neurons_create_1);

    ASSERT_THROW_NO_PRINT(fsr.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.init(0), RelearnException);

    ASSERT_EQ(fsr.get_number_local_neurons(), number_neurons_init + number_neurons_create_1);

    fsr.create_neurons(number_neurons_create_2);

    ASSERT_EQ(fsr.get_number_local_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);

    ASSERT_THROW_NO_PRINT(fsr.init(number_neurons_init), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.create_neurons(0), RelearnException);
    ASSERT_THROW_NO_PRINT(fsr.init(0), RelearnException);

    ASSERT_EQ(fsr.get_number_local_neurons(), number_neurons_init + number_neurons_create_1 + number_neurons_create_2);
}

TEST_F(FiredStatusRecorderTest, testSetFiredAll) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsr = FiredStatusRecorder{};

    const auto number_neurons_init = 107;
    fsr.init(number_neurons_init);

    const auto fired = fsr.get_fired();
    ASSERT_EQ(fired.size(), number_neurons_init);

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        ASSERT_FALSE(fsr.has_fired(neuron_id));
    }

    for (auto i = 0U; i < number_neurons_init; i++) {
        ASSERT_EQ(fired[i], FiredStatus::Inactive);
    }

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        fsr.set_fired(neuron_id, FiredStatus::Fired);
    }

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        ASSERT_TRUE(fsr.has_fired(neuron_id));
    }

    for (auto i = 0U; i < number_neurons_init; i++) {
        ASSERT_EQ(fired[i], FiredStatus::Fired);
    }

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        fsr.set_fired(neuron_id, FiredStatus::Inactive);
    }

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        ASSERT_FALSE(fsr.has_fired(neuron_id));
    }

    for (auto i = 0U; i < number_neurons_init; i++) {
        ASSERT_EQ(fired[i], FiredStatus::Inactive);
    }
}

TEST_F(FiredStatusRecorderTest, testSetFiredSome) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsr = FiredStatusRecorder{};

    const auto number_neurons_init = 47;
    fsr.init(number_neurons_init);

    const auto fired = fsr.get_fired();
    ASSERT_EQ(fired.size(), number_neurons_init);

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        ASSERT_FALSE(fsr.has_fired(neuron_id));
    }

    for (auto i = 0U; i < number_neurons_init; i++) {
        ASSERT_EQ(fired[i], FiredStatus::Inactive);
    }

    fsr.set_fired(NeuronID{ 3 }, FiredStatus::Fired);
    fsr.set_fired(NeuronID{ 11 }, FiredStatus::Fired);
    fsr.set_fired(NeuronID{ 24 }, FiredStatus::Fired);
    fsr.set_fired(NeuronID{ 42 }, FiredStatus::Fired);

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        const auto id = neuron_id.get_neuron_id();
        if (id == 3 || id == 11 || id == 24 || id == 42) {
            ASSERT_TRUE(fsr.has_fired(neuron_id));
        } else {
            ASSERT_FALSE(fsr.has_fired(neuron_id));
        }
    }

    for (auto i = 0U; i < number_neurons_init; i++) {
        if (i == 3 || i == 11 || i == 24 || i == 42) {
            ASSERT_EQ(fired[i], FiredStatus::Fired);
        } else {
            ASSERT_EQ(fired[i], FiredStatus::Inactive);
        }
    }
}

TEST_F(FiredStatusRecorderTest, testSetFiredThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsr = FiredStatusRecorder{};

    const auto number_neurons_init = 42;
    fsr.init(number_neurons_init);

    const auto fired = fsr.get_fired();

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init, number_neurons_init + 100)) {
        ASSERT_THROW_NO_PRINT(fsr.set_fired(neuron_id, FiredStatus::Fired), RelearnException);
        ASSERT_THROW_NO_PRINT(fsr.set_fired(neuron_id, FiredStatus::Inactive), RelearnException);
        ASSERT_THROW_NO_PRINT(std::ignore = fsr.has_fired(neuron_id), RelearnException);
    }

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        ASSERT_FALSE(fsr.has_fired(neuron_id));
    }

    for (auto i = 0U; i < number_neurons_init; i++) {
        ASSERT_EQ(fired[i], FiredStatus::Inactive);
    }
}
//
// TEST_F(FiredStatusRecorderTest, testFireHistoryNormal) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     if (Config::fire_history_reset_step < 4) {
//         std::cerr << "This test requires Config::fire_history_reset_step to be at least 4.\n";
//         return;
//     }
//
//     auto fsr = FiredStatusRecorder{};
//     const auto number_neurons_init = 7;
//     fsr.init(number_neurons_init);
//
//     const auto& fire_history_0 = fsr.get_fire_history(NeuronID{ 0 });
//     const auto& fire_history_1 = fsr.get_fire_history(NeuronID{ 1 });
//     const auto& fire_history_2 = fsr.get_fire_history(NeuronID{ 2 });
//     const auto& fire_history_3 = fsr.get_fire_history(NeuronID{ 3 });
//     const auto& fire_history_4 = fsr.get_fire_history(NeuronID{ 4 });
//     const auto& fire_history_5 = fsr.get_fire_history(NeuronID{ 5 });
//     const auto& fire_history_6 = fsr.get_fire_history(NeuronID{ 6 });
//
//     ASSERT_EQ(fsr.get_fire_history_size(), Config::fire_history_reset_step);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         ASSERT_EQ(fire_history.size(), Config::fire_history_reset_step);
//         for (auto i = 0U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
//
//     const auto fired_1 = std::vector<FiredStatus>{
//         FiredStatus::Fired,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//     };
//
//     const auto fired_2 = std::vector<FiredStatus>{
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//     };
//
//     const auto fired_3 = std::vector<FiredStatus>{
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//     };
//
//     const auto fired_4 = std::vector<FiredStatus>{
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Fired,
//         FiredStatus::Inactive,
//         FiredStatus::Fired,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//     };
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_1[neuron_id.get_neuron_id()]);
//     }
//
//     ASSERT_TRUE(fire_history_0[0]);
//     ASSERT_FALSE(fire_history_0[1]);
//     ASSERT_FALSE(fire_history_0[2]);
//     ASSERT_FALSE(fire_history_0[3]);
//
//     ASSERT_FALSE(fire_history_1[0]);
//     ASSERT_FALSE(fire_history_1[1]);
//     ASSERT_FALSE(fire_history_1[2]);
//     ASSERT_FALSE(fire_history_1[3]);
//
//     ASSERT_FALSE(fire_history_2[0]);
//     ASSERT_FALSE(fire_history_2[1]);
//     ASSERT_FALSE(fire_history_2[2]);
//     ASSERT_FALSE(fire_history_2[3]);
//
//     ASSERT_FALSE(fire_history_3[0]);
//     ASSERT_FALSE(fire_history_3[1]);
//     ASSERT_FALSE(fire_history_3[2]);
//     ASSERT_FALSE(fire_history_3[3]);
//
//     ASSERT_TRUE(fire_history_4[0]);
//     ASSERT_FALSE(fire_history_4[1]);
//     ASSERT_FALSE(fire_history_4[2]);
//     ASSERT_FALSE(fire_history_4[3]);
//
//     ASSERT_TRUE(fire_history_5[0]);
//     ASSERT_FALSE(fire_history_5[1]);
//     ASSERT_FALSE(fire_history_5[2]);
//     ASSERT_FALSE(fire_history_5[3]);
//
//     ASSERT_TRUE(fire_history_6[0]);
//     ASSERT_FALSE(fire_history_6[1]);
//     ASSERT_FALSE(fire_history_6[2]);
//     ASSERT_FALSE(fire_history_6[3]);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         for (auto i = 4U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_2[neuron_id.get_neuron_id()]);
//     }
//
//     ASSERT_TRUE(fire_history_0[0]);
//     ASSERT_TRUE(fire_history_0[1]);
//     ASSERT_FALSE(fire_history_0[2]);
//     ASSERT_FALSE(fire_history_0[3]);
//
//     ASSERT_TRUE(fire_history_1[0]);
//     ASSERT_FALSE(fire_history_1[1]);
//     ASSERT_FALSE(fire_history_1[2]);
//     ASSERT_FALSE(fire_history_1[3]);
//
//     ASSERT_TRUE(fire_history_2[0]);
//     ASSERT_FALSE(fire_history_2[1]);
//     ASSERT_FALSE(fire_history_2[2]);
//     ASSERT_FALSE(fire_history_2[3]);
//
//     ASSERT_TRUE(fire_history_3[0]);
//     ASSERT_FALSE(fire_history_3[1]);
//     ASSERT_FALSE(fire_history_3[2]);
//     ASSERT_FALSE(fire_history_3[3]);
//
//     ASSERT_TRUE(fire_history_4[0]);
//     ASSERT_TRUE(fire_history_4[1]);
//     ASSERT_FALSE(fire_history_4[2]);
//     ASSERT_FALSE(fire_history_4[3]);
//
//     ASSERT_TRUE(fire_history_5[0]);
//     ASSERT_TRUE(fire_history_5[1]);
//     ASSERT_FALSE(fire_history_5[2]);
//     ASSERT_FALSE(fire_history_5[3]);
//
//     ASSERT_TRUE(fire_history_6[0]);
//     ASSERT_TRUE(fire_history_6[1]);
//     ASSERT_FALSE(fire_history_6[2]);
//     ASSERT_FALSE(fire_history_6[3]);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         for (auto i = 4U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_3[neuron_id.get_neuron_id()]);
//     }
//
//     ASSERT_FALSE(fire_history_0[0]);
//     ASSERT_TRUE(fire_history_0[1]);
//     ASSERT_TRUE(fire_history_0[2]);
//     ASSERT_FALSE(fire_history_0[3]);
//
//     ASSERT_FALSE(fire_history_1[0]);
//     ASSERT_TRUE(fire_history_1[1]);
//     ASSERT_FALSE(fire_history_1[2]);
//     ASSERT_FALSE(fire_history_1[3]);
//
//     ASSERT_FALSE(fire_history_2[0]);
//     ASSERT_TRUE(fire_history_2[1]);
//     ASSERT_FALSE(fire_history_2[2]);
//     ASSERT_FALSE(fire_history_2[3]);
//
//     ASSERT_FALSE(fire_history_3[0]);
//     ASSERT_TRUE(fire_history_3[1]);
//     ASSERT_FALSE(fire_history_3[2]);
//     ASSERT_FALSE(fire_history_3[3]);
//
//     ASSERT_FALSE(fire_history_4[0]);
//     ASSERT_TRUE(fire_history_4[1]);
//     ASSERT_TRUE(fire_history_4[2]);
//     ASSERT_FALSE(fire_history_4[3]);
//
//     ASSERT_FALSE(fire_history_5[0]);
//     ASSERT_TRUE(fire_history_5[1]);
//     ASSERT_TRUE(fire_history_5[2]);
//     ASSERT_FALSE(fire_history_5[3]);
//
//     ASSERT_FALSE(fire_history_6[0]);
//     ASSERT_TRUE(fire_history_6[1]);
//     ASSERT_TRUE(fire_history_6[2]);
//     ASSERT_FALSE(fire_history_6[3]);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         for (auto i = 4U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_4[neuron_id.get_neuron_id()]);
//     }
//
//     ASSERT_FALSE(fire_history_0[0]);
//     ASSERT_FALSE(fire_history_0[1]);
//     ASSERT_TRUE(fire_history_0[2]);
//     ASSERT_TRUE(fire_history_0[3]);
//
//     ASSERT_FALSE(fire_history_1[0]);
//     ASSERT_FALSE(fire_history_1[1]);
//     ASSERT_TRUE(fire_history_1[2]);
//     ASSERT_FALSE(fire_history_1[3]);
//
//     ASSERT_TRUE(fire_history_2[0]);
//     ASSERT_FALSE(fire_history_2[1]);
//     ASSERT_TRUE(fire_history_2[2]);
//     ASSERT_FALSE(fire_history_2[3]);
//
//     ASSERT_FALSE(fire_history_3[0]);
//     ASSERT_FALSE(fire_history_3[1]);
//     ASSERT_TRUE(fire_history_3[2]);
//     ASSERT_FALSE(fire_history_3[3]);
//
//     ASSERT_TRUE(fire_history_4[0]);
//     ASSERT_FALSE(fire_history_4[1]);
//     ASSERT_TRUE(fire_history_4[2]);
//     ASSERT_TRUE(fire_history_4[3]);
//
//     ASSERT_FALSE(fire_history_5[0]);
//     ASSERT_FALSE(fire_history_5[1]);
//     ASSERT_TRUE(fire_history_5[2]);
//     ASSERT_TRUE(fire_history_5[3]);
//
//     ASSERT_FALSE(fire_history_6[0]);
//     ASSERT_FALSE(fire_history_6[1]);
//     ASSERT_TRUE(fire_history_6[2]);
//     ASSERT_TRUE(fire_history_6[3]);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         for (auto i = 4U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
// }
//
// TEST_F(FiredStatusRecorderTest, testFireHistoryCreate) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     if (Config::fire_history_reset_step < 4) {
//         std::cerr << "This test requires Config::fire_history_reset_step to be at least 4.\n";
//         return;
//     }
//
//     auto fsr = FiredStatusRecorder{};
//     const auto number_neurons_init = 7;
//     fsr.init(number_neurons_init);
//
//     const auto number_nerons_create_1 = 6;
//     const auto number_nerons_create_2 = 7;
//     const auto number_nerons_create_3 = 2;
//     const auto number_nerons_create_4 = 12;
//     const auto number_nerons_create_5 = 1;
//
//     const auto number_neurons_create = number_nerons_create_1 + number_nerons_create_2 + number_nerons_create_3 + number_nerons_create_4 + number_nerons_create_5;
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         ASSERT_EQ(fire_history.size(), Config::fire_history_reset_step);
//         for (auto i = 0U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
//
//     const auto fired_1 = std::vector<FiredStatus>{
//         FiredStatus::Fired,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//     };
//
//     const auto fired_2 = std::vector<FiredStatus>{
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//     };
//
//     const auto fired_3 = std::vector<FiredStatus>{
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//     };
//
//     const auto fired_4 = std::vector<FiredStatus>{
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Fired,
//         FiredStatus::Inactive,
//         FiredStatus::Fired,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//     };
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_1[neuron_id.get_neuron_id()]);
//     }
//
//     fsr.create_neurons(number_nerons_create_1);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_2[neuron_id.get_neuron_id()]);
//     }
//
//     fsr.create_neurons(number_nerons_create_2);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_3[neuron_id.get_neuron_id()]);
//     }
//
//     fsr.create_neurons(number_nerons_create_3);
//     fsr.create_neurons(number_nerons_create_4);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_4[neuron_id.get_neuron_id()]);
//     }
//
//     fsr.create_neurons(number_nerons_create_5);
//
//     const auto& fire_history_0 = fsr.get_fire_history(NeuronID{ 0 });
//     const auto& fire_history_1 = fsr.get_fire_history(NeuronID{ 1 });
//     const auto& fire_history_2 = fsr.get_fire_history(NeuronID{ 2 });
//     const auto& fire_history_3 = fsr.get_fire_history(NeuronID{ 3 });
//     const auto& fire_history_4 = fsr.get_fire_history(NeuronID{ 4 });
//     const auto& fire_history_5 = fsr.get_fire_history(NeuronID{ 5 });
//     const auto& fire_history_6 = fsr.get_fire_history(NeuronID{ 6 });
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         for (auto i = 4U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
//
//     ASSERT_FALSE(fire_history_0[0]);
//     ASSERT_FALSE(fire_history_0[1]);
//     ASSERT_TRUE(fire_history_0[2]);
//     ASSERT_TRUE(fire_history_0[3]);
//
//     ASSERT_FALSE(fire_history_1[0]);
//     ASSERT_FALSE(fire_history_1[1]);
//     ASSERT_TRUE(fire_history_1[2]);
//     ASSERT_FALSE(fire_history_1[3]);
//
//     ASSERT_TRUE(fire_history_2[0]);
//     ASSERT_FALSE(fire_history_2[1]);
//     ASSERT_TRUE(fire_history_2[2]);
//     ASSERT_FALSE(fire_history_2[3]);
//
//     ASSERT_FALSE(fire_history_3[0]);
//     ASSERT_FALSE(fire_history_3[1]);
//     ASSERT_TRUE(fire_history_3[2]);
//     ASSERT_FALSE(fire_history_3[3]);
//
//     ASSERT_TRUE(fire_history_4[0]);
//     ASSERT_FALSE(fire_history_4[1]);
//     ASSERT_TRUE(fire_history_4[2]);
//     ASSERT_TRUE(fire_history_4[3]);
//
//     ASSERT_FALSE(fire_history_5[0]);
//     ASSERT_FALSE(fire_history_5[1]);
//     ASSERT_TRUE(fire_history_5[2]);
//     ASSERT_TRUE(fire_history_5[3]);
//
//     ASSERT_FALSE(fire_history_6[0]);
//     ASSERT_FALSE(fire_history_6[1]);
//     ASSERT_TRUE(fire_history_6[2]);
//     ASSERT_TRUE(fire_history_6[3]);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         for (auto i = 4U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init, number_neurons_init + number_neurons_create)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         ASSERT_EQ(fire_history.size(), Config::fire_history_reset_step);
//         for (auto i = 0U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
// }
//
// TEST_F(FiredStatusRecorderTest, testFireHistoryLocalRank) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     if (Config::fire_history_reset_step < 4) {
//         std::cerr << "This test requires Config::fire_history_reset_step to be at least 4.\n";
//         return;
//     }
//
//     auto fsr = FiredStatusRecorder{};
//     const auto number_neurons_init = 7;
//     fsr.init(number_neurons_init);
//
//     const auto fired_1 = std::vector<FiredStatus>{
//         FiredStatus::Fired,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//     };
//
//     const auto fired_2 = std::vector<FiredStatus>{
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//         FiredStatus::Fired,
//     };
//
//     const auto fired_3 = std::vector<FiredStatus>{
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//     };
//
//     const auto fired_4 = std::vector<FiredStatus>{
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//         FiredStatus::Fired,
//         FiredStatus::Inactive,
//         FiredStatus::Fired,
//         FiredStatus::Inactive,
//         FiredStatus::Inactive,
//     };
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_1[neuron_id.get_neuron_id()]);
//     }
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_2[neuron_id.get_neuron_id()]);
//     }
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_3[neuron_id.get_neuron_id()]);
//     }
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         fsr.set_fired(neuron_id, fired_4[neuron_id.get_neuron_id()]);
//     }
//
//     const auto& fire_history_0 = fsr.get_fire_history(RankNeuronId{ mpiPP::MPIRank{ 0 }, NeuronID{ 0 } });
//     const auto& fire_history_1 = fsr.get_fire_history(RankNeuronId{ mpiPP::MPIRank{ 0 }, NeuronID{ 1 } });
//     const auto& fire_history_2 = fsr.get_fire_history(RankNeuronId{ mpiPP::MPIRank{ 0 }, NeuronID{ 2 } });
//     const auto& fire_history_3 = fsr.get_fire_history(RankNeuronId{ mpiPP::MPIRank{ 0 }, NeuronID{ 3 } });
//     const auto& fire_history_4 = fsr.get_fire_history(RankNeuronId{ mpiPP::MPIRank{ 0 }, NeuronID{ 4 } });
//     const auto& fire_history_5 = fsr.get_fire_history(RankNeuronId{ mpiPP::MPIRank{ 0 }, NeuronID{ 5 } });
//     const auto& fire_history_6 = fsr.get_fire_history(RankNeuronId{ mpiPP::MPIRank{ 0 }, NeuronID{ 6 } });
//
//     ASSERT_FALSE(fire_history_0[0]);
//     ASSERT_FALSE(fire_history_0[1]);
//     ASSERT_TRUE(fire_history_0[2]);
//     ASSERT_TRUE(fire_history_0[3]);
//
//     ASSERT_FALSE(fire_history_1[0]);
//     ASSERT_FALSE(fire_history_1[1]);
//     ASSERT_TRUE(fire_history_1[2]);
//     ASSERT_FALSE(fire_history_1[3]);
//
//     ASSERT_TRUE(fire_history_2[0]);
//     ASSERT_FALSE(fire_history_2[1]);
//     ASSERT_TRUE(fire_history_2[2]);
//     ASSERT_FALSE(fire_history_2[3]);
//
//     ASSERT_FALSE(fire_history_3[0]);
//     ASSERT_FALSE(fire_history_3[1]);
//     ASSERT_TRUE(fire_history_3[2]);
//     ASSERT_FALSE(fire_history_3[3]);
//
//     ASSERT_TRUE(fire_history_4[0]);
//     ASSERT_FALSE(fire_history_4[1]);
//     ASSERT_TRUE(fire_history_4[2]);
//     ASSERT_TRUE(fire_history_4[3]);
//
//     ASSERT_FALSE(fire_history_5[0]);
//     ASSERT_FALSE(fire_history_5[1]);
//     ASSERT_TRUE(fire_history_5[2]);
//     ASSERT_TRUE(fire_history_5[3]);
//
//     ASSERT_FALSE(fire_history_6[0]);
//     ASSERT_FALSE(fire_history_6[1]);
//     ASSERT_TRUE(fire_history_6[2]);
//     ASSERT_TRUE(fire_history_6[3]);
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
//         const auto& fire_history = fsr.get_fire_history(neuron_id);
//         for (auto i = 4U; i < Config::fire_history_reset_step; i++) {
//             ASSERT_FALSE(fire_history[i]);
//         }
//     }
//
//     for (const auto neuron_id : NeuronIDRange::range(number_neurons_init, number_neurons_init + 100)) {
//         ASSERT_THROW_NO_PRINT(std::ignore = fsr.get_fire_history(RankNeuronId{ mpiPP::MPIRank{ 0 }, neuron_id }), RelearnException);
//     }
// }

TEST_F(FiredStatusRecorderTest, testFireRecord) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsr = FiredStatusRecorder{};
    const auto number_neurons_init = 7;
    fsr.init(number_neurons_init);

    const auto fired_1 = std::vector<FiredStatus>{
        FiredStatus::Fired,
        FiredStatus::Inactive,
        FiredStatus::Inactive,
        FiredStatus::Inactive,
        FiredStatus::Fired,
        FiredStatus::Fired,
        FiredStatus::Fired,
    };

    const auto fired_2 = std::vector<FiredStatus>{
        FiredStatus::Fired,
        FiredStatus::Fired,
        FiredStatus::Fired,
        FiredStatus::Fired,
        FiredStatus::Fired,
        FiredStatus::Fired,
        FiredStatus::Fired,
    };

    const auto fired_3 = std::vector<FiredStatus>{
        FiredStatus::Inactive,
        FiredStatus::Inactive,
        FiredStatus::Inactive,
        FiredStatus::Inactive,
        FiredStatus::Inactive,
        FiredStatus::Inactive,
        FiredStatus::Inactive,
    };

    const auto fired_4 = std::vector<FiredStatus>{
        FiredStatus::Inactive,
        FiredStatus::Inactive,
        FiredStatus::Fired,
        FiredStatus::Inactive,
        FiredStatus::Fired,
        FiredStatus::Inactive,
        FiredStatus::Inactive,
    };

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        fsr.set_fired(neuron_id, fired_1[neuron_id.get_neuron_id()]);
    }

    fsr.reset(FiredStatusRecorder::FireRecorderPeriod::GroupMonitor);

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        fsr.set_fired(neuron_id, fired_2[neuron_id.get_neuron_id()]);
    }

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        fsr.set_fired(neuron_id, fired_3[neuron_id.get_neuron_id()]);
    }

    fsr.reset(FiredStatusRecorder::FireRecorderPeriod::NeuronMonitor);

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        fsr.set_fired(neuron_id, fired_4[neuron_id.get_neuron_id()]);
    }

    const auto fired_record_group_monitor = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::GroupMonitor);
    ASSERT_EQ(fired_record_group_monitor.size(), 7);
    ASSERT_EQ(fired_record_group_monitor[0], 1);
    ASSERT_EQ(fired_record_group_monitor[1], 1);
    ASSERT_EQ(fired_record_group_monitor[2], 2);
    ASSERT_EQ(fired_record_group_monitor[3], 1);
    ASSERT_EQ(fired_record_group_monitor[4], 2);
    ASSERT_EQ(fired_record_group_monitor[5], 1);
    ASSERT_EQ(fired_record_group_monitor[6], 1);

    const auto fired_record_neuron_monitor = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::NeuronMonitor);
    ASSERT_EQ(fired_record_neuron_monitor.size(), 7);
    ASSERT_EQ(fired_record_neuron_monitor[0], 0);
    ASSERT_EQ(fired_record_neuron_monitor[1], 0);
    ASSERT_EQ(fired_record_neuron_monitor[2], 1);
    ASSERT_EQ(fired_record_neuron_monitor[3], 0);
    ASSERT_EQ(fired_record_neuron_monitor[4], 1);
    ASSERT_EQ(fired_record_neuron_monitor[5], 0);
    ASSERT_EQ(fired_record_neuron_monitor[6], 0);

    const auto fired_record_plasticity = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::Plasticity);
    ASSERT_EQ(fired_record_plasticity.size(), 7);
    ASSERT_EQ(fired_record_plasticity[0], 2);
    ASSERT_EQ(fired_record_plasticity[1], 1);
    ASSERT_EQ(fired_record_plasticity[2], 2);
    ASSERT_EQ(fired_record_plasticity[3], 1);
    ASSERT_EQ(fired_record_plasticity[4], 3);
    ASSERT_EQ(fired_record_plasticity[5], 2);
    ASSERT_EQ(fired_record_plasticity[6], 2);
}

// Regression test for a bug found and fixed this session: create_neurons() rebuilt the GPU-side
// per-period counter array (all_fired_recorders, laid out period-major as
// [period0: n neurons][period1: n neurons][period2: n neurons]) via a plain resize() to
// number_fire_recorders * creation_count (undersized -- should be * new_size, an out-of-bounds
// write once the kernel touched any neuron beyond creation_count) using offsets computed from the
// stale pre-resize neuron count. Even after fixing the size, a naive resize() only appends at the
// end, which would leave period 1's/2's *existing* data at their old, now-wrong offsets instead of
// migrating it to the new, wider ones. This test builds up distinct, independently-verifiable
// counts per period *before* growing the population, then checks every original neuron's counts
// survived at the right index in every period, and every newly added neuron starts at 0.
TEST_F(FiredStatusRecorderTest, testFireRecordSurvivesCreateNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsr = FiredStatusRecorder{};
    const auto number_neurons_init = 4;
    fsr.init(number_neurons_init);

    // Same construction style as testFireRecord above: fire a pattern, reset one period, fire
    // another pattern, reset a different period, fire a third pattern -- resets are what make the
    // three periods' final counts genuinely independent (without them, set_fired() bumps all
    // three together and they'd trivially match even with the aliasing bug this test guards
    // against).
    const auto fired_a = std::vector<FiredStatus>{ FiredStatus::Fired, FiredStatus::Fired, FiredStatus::Fired, FiredStatus::Fired };
    const auto fired_b = std::vector<FiredStatus>{ FiredStatus::Fired, FiredStatus::Fired, FiredStatus::Inactive, FiredStatus::Inactive };
    const auto fired_c = std::vector<FiredStatus>{ FiredStatus::Inactive, FiredStatus::Inactive, FiredStatus::Fired, FiredStatus::Inactive };

    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        fsr.set_fired(neuron_id, fired_a[neuron_id.get_neuron_id()]);
    }
    fsr.reset(FiredStatusRecorder::FireRecorderPeriod::GroupMonitor);
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        fsr.set_fired(neuron_id, fired_b[neuron_id.get_neuron_id()]);
    }
    fsr.reset(FiredStatusRecorder::FireRecorderPeriod::NeuronMonitor);
    for (const auto neuron_id : NeuronIDRange::range(number_neurons_init)) {
        fsr.set_fired(neuron_id, fired_c[neuron_id.get_neuron_id()]);
    }

    // Hand-traced: after fired_a, all three periods are [1,1,1,1]. reset(GroupMonitor) -> Group
    // [0,0,0,0]. fired_b fires neurons 0,1 again -> Group [1,1,0,0], Neuron [2,2,1,1], Plasticity
    // [2,2,1,1]. reset(NeuronMonitor) -> Neuron [0,0,0,0]. fired_c fires neuron 2 again -> Group
    // [1,1,1,0], Neuron [0,0,1,0], Plasticity [2,2,2,1].
    const auto expected_area_before = std::vector<unsigned int>{ 1, 1, 1, 0 };
    const auto expected_neuron_before = std::vector<unsigned int>{ 0, 0, 1, 0 };
    const auto expected_plasticity_before = std::vector<unsigned int>{ 2, 2, 2, 1 };

    const auto area_before = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::GroupMonitor);
    const auto neuron_before = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::NeuronMonitor);
    const auto plasticity_before = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::Plasticity);
    for (auto id = std::size_t{ 0 }; id < static_cast<std::size_t>(number_neurons_init); ++id) {
        ASSERT_EQ(area_before[id], expected_area_before[id]) << "neuron " << id << " before create_neurons (GroupMonitor)";
        ASSERT_EQ(neuron_before[id], expected_neuron_before[id]) << "neuron " << id << " before create_neurons (NeuronMonitor)";
        ASSERT_EQ(plasticity_before[id], expected_plasticity_before[id]) << "neuron " << id << " before create_neurons (Plasticity)";
    }

    constexpr auto creation_count = 3;
    fsr.create_neurons(creation_count);
    const auto number_neurons_after = number_neurons_init + creation_count;

    const auto area_after = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::GroupMonitor);
    const auto neuron_after = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::NeuronMonitor);
    const auto plasticity_after = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::Plasticity);
    ASSERT_EQ(area_after.size(), number_neurons_after);
    ASSERT_EQ(neuron_after.size(), number_neurons_after);
    ASSERT_EQ(plasticity_after.size(), number_neurons_after);

    for (auto id = std::size_t{ 0 }; id < static_cast<std::size_t>(number_neurons_init); ++id) {
        ASSERT_EQ(area_after[id], expected_area_before[id]) << "neuron " << id << " after create_neurons (GroupMonitor) -- old data misplaced";
        ASSERT_EQ(neuron_after[id], expected_neuron_before[id]) << "neuron " << id << " after create_neurons (NeuronMonitor) -- old data misplaced";
        ASSERT_EQ(plasticity_after[id], expected_plasticity_before[id]) << "neuron " << id << " after create_neurons (Plasticity) -- old data misplaced";
    }
    for (auto id = static_cast<std::size_t>(number_neurons_init); id < static_cast<std::size_t>(number_neurons_after); ++id) {
        ASSERT_EQ(area_after[id], 0U) << "newly added neuron " << id << " (GroupMonitor)";
        ASSERT_EQ(neuron_after[id], 0U) << "newly added neuron " << id << " (NeuronMonitor)";
        ASSERT_EQ(plasticity_after[id], 0U) << "newly added neuron " << id << " (Plasticity)";
    }

    // The newly added neurons must be independently usable too, not just zeroed.
    fsr.set_fired(NeuronID{ static_cast<NeuronID::value_type>(number_neurons_init) }, FiredStatus::Fired);
    const auto area_final = fsr.get_fired_recorder(FiredStatusRecorder::FireRecorderPeriod::GroupMonitor);
    ASSERT_EQ(area_final[static_cast<std::size_t>(number_neurons_init)], 1U);
    for (auto id = std::size_t{ 0 }; id < static_cast<std::size_t>(number_neurons_init); ++id) {
        ASSERT_EQ(area_final[id], expected_area_before[id]) << "neuron " << id << ": firing a new neuron corrupted an old one's count";
    }
}

TEST_F(FiredStatusRecorderTest, testFootprintNoThrow) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto fsr = FiredStatusRecorder{};

    const auto number_neurons_init = 42;
    fsr.init(number_neurons_init);

    const auto footprint = std::make_unique<utility::MemoryFootprint>(100);

    ASSERT_NO_THROW(fsr.record_memory_footprint(footprint));
}
