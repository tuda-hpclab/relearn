/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_neuron_monitor.h"

#include "neurons/helper/NeuronMonitor.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

static_assert(sizeof(Parameter) <= 8, "Parameter is too large to fit into registers");

TEST_F(NeuronMonitorTest, testRegisterParameter) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto nm = NeuronMonitor{};

    nm.set_output_path("./", true);

    ASSERT_NO_THROW(nm.register_paramter("test1", [](RelearnTypes::number_neurons_type) { return 0.0F; }, []() { }, []() { }));
    ASSERT_NO_THROW(nm.register_paramter("test2", [](RelearnTypes::number_neurons_type) { return 0; }, []() { }, []() { }));
    ASSERT_NO_THROW(nm.register_paramter("test3", [](RelearnTypes::number_neurons_type) { return 0U; }, []() { }, []() { }));
    ASSERT_NO_THROW(nm.register_paramter("test4", [](RelearnTypes::number_neurons_type) { return false; }, []() { }, []() { }));
}

TEST_F(NeuronMonitorTest, testRegisterNeuron) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto nm = NeuronMonitor{};

    nm.set_output_path("./", true);

    ASSERT_NO_THROW(nm.register_neuron(0));
    ASSERT_NO_THROW(nm.register_neuron(1));
    ASSERT_NO_THROW(nm.register_neuron(2048));
}

TEST_F(NeuronMonitorTest, testCallbacks) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto nm = NeuronMonitor{};

    nm.set_output_path("./", true);

    auto test1_calls = std::unordered_set<RelearnTypes::number_neurons_type>{};
    auto test2_calls = std::unordered_set<RelearnTypes::number_neurons_type>{};
    auto test3_calls = std::unordered_set<RelearnTypes::number_neurons_type>{};
    auto test4_calls = std::unordered_set<RelearnTypes::number_neurons_type>{};

    nm.register_paramter("test1", [&test1_calls](RelearnTypes::number_neurons_type neuron_id) {
                             test1_calls.emplace(neuron_id);
                             return 0.0F; }, []() { }, []() { });

    nm.record_data(100);

    ASSERT_TRUE(test1_calls.empty());
    ASSERT_TRUE(test2_calls.empty());
    ASSERT_TRUE(test3_calls.empty());
    ASSERT_TRUE(test4_calls.empty());

    nm.register_paramter("test2", [&test2_calls](RelearnTypes::number_neurons_type neuron_id) {
                             test2_calls.emplace(neuron_id);
                             return 0; }, []() { }, []() { });

    nm.record_data(101);

    ASSERT_TRUE(test1_calls.empty());
    ASSERT_TRUE(test2_calls.empty());
    ASSERT_TRUE(test3_calls.empty());
    ASSERT_TRUE(test4_calls.empty());

    nm.register_neuron(25);

    nm.record_data(102);

    ASSERT_EQ(1, test1_calls.size());
    ASSERT_EQ(1, test2_calls.size());

    ASSERT_TRUE(test1_calls.contains(25));
    ASSERT_TRUE(test2_calls.contains(25));

    ASSERT_TRUE(test3_calls.empty());
    ASSERT_TRUE(test4_calls.empty());

    test1_calls.clear();
    test2_calls.clear();

    nm.register_paramter("test3", [&test3_calls](RelearnTypes::number_neurons_type neuron_id) {
                             test3_calls.emplace(neuron_id);
                             return 0U; }, []() { }, []() { });

    nm.register_paramter("test4", [&test4_calls](RelearnTypes::number_neurons_type neuron_id) {
                             test4_calls.emplace(neuron_id);
                             return false; }, []() { }, []() { });

    nm.register_neuron(30);
    nm.register_neuron(32);

    nm.record_data(103);

    ASSERT_EQ(3, test1_calls.size());
    ASSERT_EQ(3, test2_calls.size());
    ASSERT_EQ(3, test3_calls.size());
    ASSERT_EQ(3, test4_calls.size());

    ASSERT_TRUE(test1_calls.contains(25));
    ASSERT_TRUE(test2_calls.contains(25));
    ASSERT_TRUE(test3_calls.contains(25));
    ASSERT_TRUE(test4_calls.contains(25));

    ASSERT_TRUE(test1_calls.contains(30));
    ASSERT_TRUE(test2_calls.contains(30));
    ASSERT_TRUE(test3_calls.contains(30));
    ASSERT_TRUE(test4_calls.contains(30));

    ASSERT_TRUE(test1_calls.contains(32));
    ASSERT_TRUE(test2_calls.contains(32));
    ASSERT_TRUE(test3_calls.contains(32));
    ASSERT_TRUE(test4_calls.contains(32));
}

TEST_F(NeuronMonitorTest, testFlushOutput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto nm = NeuronMonitor{};

    nm.set_output_path("./t1", true);

    nm.register_paramter("test1", [](RelearnTypes::number_neurons_type neuron_id) { return static_cast<float>(neuron_id) + 0.024F; }, []() { }, []() { });
    nm.register_paramter("test2", [](RelearnTypes::number_neurons_type neuron_id) { return -static_cast<int>(neuron_id); }, []() { }, []() { });
    nm.register_paramter("test3", [](RelearnTypes::number_neurons_type neuron_id) { return static_cast<unsigned int>(neuron_id); }, []() { }, []() { });
    nm.register_paramter("test4", [](RelearnTypes::number_neurons_type /*neuron_id*/) { return false; }, []() { }, []() { });

    nm.register_neuron(25);
    nm.register_neuron(30);
    nm.register_neuron(32);

    nm.record_data(100);
    nm.record_data(101);
    nm.record_data(102);

    nm.flush_current_contents();

    auto path = std::filesystem::path("./t1/neuron_monitors");

    ASSERT_TRUE(std::filesystem::exists(path));

    const auto check_file_25_content = [](const std::vector<std::string>& lines) {
        const auto expected_lines = std::array<std::string, 4>{
            "# Step;test1;test2;test3;test4",
            "100;25.024;-25;25;0",
            "101;25.024;-25;25;0",
            "102;25.024;-25;25;0"
        };

        for (auto i = 0U; i < 4U; i++) {
            ASSERT_EQ(lines[i], expected_lines[i]);
        }
    };

    const auto check_file_30_content = [](const std::vector<std::string>& lines) {
        const auto expected_lines = std::array<std::string, 4>{
            "# Step;test1;test2;test3;test4",
            "100;30.024;-30;30;0",
            "101;30.024;-30;30;0",
            "102;30.024;-30;30;0"
        };

        for (auto i = 0U; i < 4U; i++) {
            ASSERT_EQ(lines[i], expected_lines[i]);
        }
    };

    const auto check_file_32_content = [](const std::vector<std::string>& lines) {
        const auto expected_lines = std::array<std::string, 4>{
            "# Step;test1;test2;test3;test4",
            "100;32.024;-32;32;0",
            "101;32.024;-32;32;0",
            "102;32.024;-32;32;0"
        };

        for (auto i = 0U; i < 4U; i++) {
            ASSERT_EQ(lines[i], expected_lines[i]);
        }
    };

    const auto check_file_content = [&path, &check_file_25_content, &check_file_30_content, &check_file_32_content](const RelearnTypes::number_neurons_type neuron_id) {
        const auto file_path = path / (std::string("0_") + std::to_string(neuron_id + 1) + ".csv");
        ASSERT_TRUE(std::filesystem::exists(file_path));

        auto file = std::ifstream{ file_path };

        auto lines = std::vector<std::string>{};
        lines.reserve(4);

        for (auto line = std::string{}; std::getline(file, line);) {
            lines.emplace_back(std::move(line));
        }

        ASSERT_EQ(lines.size(), 4);

        if (neuron_id == 25) {
            check_file_25_content(lines);
        } else if (neuron_id == 30) {
            check_file_30_content(lines);
        } else if (neuron_id == 32) {
            check_file_32_content(lines);
        } else {
            FAIL() << "Unexpected neuron id: " << neuron_id;
        }
    };

    check_file_content(25);
    check_file_content(30);
    check_file_content(32);
}

TEST_F(NeuronMonitorTest, testMultipleFlushOutput) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    auto nm = NeuronMonitor{};

    nm.set_output_path("./t2", true);

    nm.register_paramter("test1", [](RelearnTypes::number_neurons_type neuron_id) { return static_cast<float>(neuron_id) + 0.024F; }, []() { }, []() { });
    nm.register_paramter("test2", [](RelearnTypes::number_neurons_type neuron_id) { return -static_cast<int>(neuron_id); }, []() { }, []() { });
    nm.register_paramter("test3", [](RelearnTypes::number_neurons_type neuron_id) { return static_cast<unsigned int>(neuron_id); }, []() { }, []() { });
    nm.register_paramter("test4", [](RelearnTypes::number_neurons_type /*neuron_id*/) { return false; }, []() { }, []() { });

    nm.register_neuron(25);
    nm.register_neuron(30);
    nm.register_neuron(32);

    nm.flush_current_contents();
    nm.record_data(100);
    nm.flush_current_contents();
    nm.record_data(101);
    nm.flush_current_contents();
    nm.record_data(102);
    nm.flush_current_contents();

    auto path = std::filesystem::path("./t2/neuron_monitors");

    ASSERT_TRUE(std::filesystem::exists(path));

    const auto check_file_25_content = [](const std::vector<std::string>& lines) {
        const auto expected_lines = std::array<std::string, 4>{
            "# Step;test1;test2;test3;test4",
            "100;25.024;-25;25;0",
            "101;25.024;-25;25;0",
            "102;25.024;-25;25;0"
        };

        for (auto i = 0U; i < 4U; i++) {
            ASSERT_EQ(lines[i], expected_lines[i]);
        }
    };

    const auto check_file_30_content = [](const std::vector<std::string>& lines) {
        const auto expected_lines = std::array<std::string, 4>{
            "# Step;test1;test2;test3;test4",
            "100;30.024;-30;30;0",
            "101;30.024;-30;30;0",
            "102;30.024;-30;30;0"
        };

        for (auto i = 0U; i < 4U; i++) {
            ASSERT_EQ(lines[i], expected_lines[i]);
        }
    };

    const auto check_file_32_content = [](const std::vector<std::string>& lines) {
        const auto expected_lines = std::array<std::string, 4>{
            "# Step;test1;test2;test3;test4",
            "100;32.024;-32;32;0",
            "101;32.024;-32;32;0",
            "102;32.024;-32;32;0"
        };

        for (auto i = 0U; i < 4U; i++) {
            ASSERT_EQ(lines[i], expected_lines[i]);
        }
    };

    const auto check_file_content = [&path, &check_file_25_content, &check_file_30_content, &check_file_32_content](const RelearnTypes::number_neurons_type neuron_id) {
        const auto file_path = path / (std::string("0_") + std::to_string(neuron_id + 1) + ".csv");
        ASSERT_TRUE(std::filesystem::exists(file_path));

        auto file = std::ifstream{ file_path };

        auto lines = std::vector<std::string>{};
        lines.reserve(4);

        for (auto line = std::string{}; std::getline(file, line);) {
            lines.emplace_back(std::move(line));
        }

        ASSERT_EQ(lines.size(), 4);

        if (neuron_id == 25) {
            check_file_25_content(lines);
        } else if (neuron_id == 30) {
            check_file_30_content(lines);
        } else if (neuron_id == 32) {
            check_file_32_content(lines);
        } else {
            FAIL() << "Unexpected neuron id: " << neuron_id;
        }
    };

    check_file_content(25);
    check_file_content(30);
    check_file_content(32);
}
