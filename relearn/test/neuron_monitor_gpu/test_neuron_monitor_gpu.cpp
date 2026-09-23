/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#ifdef RELEARN_CUDA_ENABLED

#include "test_neuron_monitor_gpu.h"

#include "Config.h"
#include "RelearnTest.hpp"

#include "cuda/memory/LazySyncedArray.h"
#include "neurons/calcium/CalciumCalculator.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/firing/FiredStatusRecorder.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/input/CombinedActivityInput.h"
#include "neurons/input/ConstantActivityInput.h"
#include "neurons/input/NormalActivityInput.h"
#include "neurons/input/SynapticEquallyWeightedActivityInput.h"
#include "neurons/models/aeif/AEIFModel.h"
#include "neurons/models/aeif/Parameters.h"
#include "neurons/models/fitzhughnagumo/FitzHughNagumoModel.h"
#include "neurons/models/fitzhughnagumo/Parameters.h"
#include "neurons/models/izhikevich/IzhikevichModel.h"
#include "neurons/models/izhikevich/Parameters.h"
#include "neurons/models/poisson/Parameters.h"
#include "neurons/models/poisson/PoissonModel.h"
#include "neurons/synaptic_elements/Axons.h"
#include "neurons/synaptic_elements/Dendrites.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"

#include "factory/activity_input/activity_input_factory.h"
#include "factory/extra_info/extra_info_factory.h"
#include "factory/fired_status_communicator/fired_status_communicator_factory.h"
#include "factory/network_graph/network_graph_factory.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class NeuronMonitorGPUTest : public RelearnTest { };

namespace {

// Reads back the CSV lines flush_current_contents() wrote for one monitored neuron, skipping the
// header line, matching the pattern in test_neuron_monitor.cpp's testFlushOutput.
std::vector<std::string> read_data_lines(const std::filesystem::path& output_path, const RelearnTypes::number_neurons_type neuron_id) {
    const auto file_path = output_path / "neuron_monitors" / (std::string("0_") + std::to_string(neuron_id + 1) + ".csv");
    EXPECT_TRUE(std::filesystem::exists(file_path));

    auto file = std::ifstream{ file_path };
    auto lines = std::vector<std::string>{};
    for (auto line = std::string{}; std::getline(file, line);) {
        lines.emplace_back(std::move(line));
    }

    EXPECT_GE(lines.size(), 2U);
    lines.erase(lines.begin()); // drop the "# Step;..." header
    return lines;
}

// NeuronMonitor::flush_current_contents writes each recorded value with a plain `outfile << value`
// (std::visit over the Parameter variant) -- not std::to_string, which uses fixed 6-decimal
// notation and would produce e.g. "0.500000" instead of the "0.5" the CSV actually contains.
// Every value Axons/Dendrites register is a float (see the utility::cast<float> in both
// register_neuron_monitor implementations), so expected strings must go through the same
// default-formatted ostream to match byte-for-byte.
std::string format_like_ostream(float value) {
    auto oss = std::ostringstream{};
    oss << value;
    return oss.str();
}

// Reference Euler integration replicating update_current_calcium_kernel's documented recurrence
// exactly (see source/cuda/calcium/Calcium.cu and test_calcium_gpu.cpp, which verifies the kernel
// itself against this same formula).
RelearnTypes::calcium_type expected_after_h_steps(RelearnTypes::calcium_type c0, unsigned int h,
                                                  RelearnTypes::calcium_type tau_C, RelearnTypes::calcium_type beta, bool fired) {
    const auto scale = RelearnTypes::calcium_type{ 1 } / static_cast<RelearnTypes::calcium_type>(h);
    const auto tau_C_inverse = -RelearnTypes::calcium_type{ 1 } / tau_C;
    auto c = c0;
    for (auto i = 0U; i < h; ++i) {
        if (fired) {
            c += scale * (c * tau_C_inverse + beta);
        } else {
            c += scale * (c * tau_C_inverse);
        }
    }
    return c;
}

std::vector<std::string> split(const std::string& line, char delim) {
    auto parts = std::vector<std::string>{};
    auto start = std::size_t{ 0 };
    while (true) {
        const auto pos = line.find(delim, start);
        if (pos == std::string::npos) {
            parts.push_back(line.substr(start));
            break;
        }
        parts.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return parts;
}

// Reads back one monitored neuron's CSV rows as name->value maps, keyed by column name (from
// the "# Step;name1;name2;..." header) rather than a hardcoded position -- robust against the
// exact registration order of register_neuron_monitor's callbacks.
std::vector<std::unordered_map<std::string, std::string>> read_csv_rows_by_column(
    const std::filesystem::path& output_path, const RelearnTypes::number_neurons_type neuron_id) {
    const auto file_path = output_path / "neuron_monitors" / (std::string("0_") + std::to_string(neuron_id + 1) + ".csv");
    EXPECT_TRUE(std::filesystem::exists(file_path));

    auto file = std::ifstream{ file_path };
    auto header_line = std::string{};
    EXPECT_TRUE(static_cast<bool>(std::getline(file, header_line)));

    auto column_names = split(header_line, ';');
    EXPECT_FALSE(column_names.empty());
    column_names.erase(column_names.begin()); // drop "# Step"

    auto rows = std::vector<std::unordered_map<std::string, std::string>>{};
    for (auto line = std::string{}; std::getline(file, line);) {
        auto fields = split(line, ';');
        EXPECT_FALSE(fields.empty());
        fields.erase(fields.begin()); // drop the step number column
        EXPECT_EQ(fields.size(), column_names.size());

        auto& row = rows.emplace_back();
        for (std::size_t i = 0; i < column_names.size() && i < fields.size(); ++i) {
            row[column_names[i]] = fields[i];
        }
    }
    return rows;
}

} // namespace

// Axons has a single ElementBase; writing values directly through its device pointers (as a real
// synapse-creation kernel would) and checking register_neuron_monitor's callbacks see them --
// via NeuronMonitor::record_data + flush_current_contents, exactly the CSV path a real run takes
// -- exercises the LazySyncedArray host-mirror sync that a stale-cache bug would break silently.
TEST_F(NeuronMonitorGPUTest, testAxonsMonitorReflectsDeviceWrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 4 };

    auto axons = Axons{};
    axons.init(number_neurons); // defaults every neuron's signal type to Excitatory

    device_write_synaptic_elements(axons.get_cuda_handle(), /*grown_offset=*/0.5F, /*connected_offset=*/3U);

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_axons");
    nm.set_output_path(output_path, true);

    axons.register_neuron_monitor(nm);

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < number_neurons; ++neuron_id) {
        nm.register_neuron(neuron_id);
    }

    nm.record_data(0);
    nm.flush_current_contents();

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < number_neurons; ++neuron_id) {
        const auto lines = read_data_lines(output_path, neuron_id);
        ASSERT_EQ(lines.size(), 1U);

        const auto expected_grown = 2.0F * static_cast<float>(neuron_id) + 0.5F;
        const auto expected_connected = static_cast<float>(neuron_id + 3U);

        // Column order matches Axons::register_neuron_monitor's registration order: exc. grown,
        // inh. grown, exc. connected, inh. connected. Every neuron is Excitatory here, so the
        // inh. columns must read back exactly 0.
        const auto expected = "0;" + format_like_ostream(expected_grown) + ";0;" + format_like_ostream(expected_connected) + ";0";
        ASSERT_EQ(lines[0], expected);
    }
}

// Dendrites hold two independent ElementBase instances (excitatory/inhibitory) behind one
// SignalType-dispatched API. Writing distinguishable values to each and checking the monitor
// reports the right value under the right column catches a mixed-up handle (e.g.
// get_cuda_handle(Excitatory) accidentally touching the inhibitory buffer).
TEST_F(NeuronMonitorGPUTest, testDendritesMonitorDistinguishesSignalTypes) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 3 };

    auto dendrites = Dendrites{};
    dendrites.init(number_neurons);

    device_write_synaptic_elements(dendrites.get_cuda_handle(SignalType::Excitatory), /*grown_offset=*/0.5F, /*connected_offset=*/10U);

    device_write_synaptic_elements(dendrites.get_cuda_handle(SignalType::Inhibitory), /*grown_offset=*/100.5F, /*connected_offset=*/200U);

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_dendrites");
    nm.set_output_path(output_path, true);

    dendrites.register_neuron_monitor(nm);

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < number_neurons; ++neuron_id) {
        nm.register_neuron(neuron_id);
    }

    nm.record_data(0);
    nm.flush_current_contents();

    for (auto neuron_id = RelearnTypes::number_neurons_type{ 0 }; neuron_id < number_neurons; ++neuron_id) {
        const auto lines = read_data_lines(output_path, neuron_id);
        ASSERT_EQ(lines.size(), 1U);

        const auto expected_exc_grown = 2.0F * static_cast<float>(neuron_id) + 0.5F;
        const auto expected_inh_grown = 2.0F * static_cast<float>(neuron_id) + 100.5F;
        const auto expected_exc_connected = static_cast<float>(neuron_id + 10U);
        const auto expected_inh_connected = static_cast<float>(neuron_id + 200U);

        // Column order matches Dendrites::register_neuron_monitor: exc. grown, inh. grown,
        // exc. connected, inh. connected.
        const auto expected = "0;" + format_like_ostream(expected_exc_grown) + ";" + format_like_ostream(expected_inh_grown)
                              + ";" + format_like_ostream(expected_exc_connected) + ";" + format_like_ostream(expected_inh_connected);
        ASSERT_EQ(lines[0], expected);
    }
}

// Drives CalciumCalculator's real GPU update path (update_calcium -> update_current_calcium_gpu,
// the same kernel test_calcium_gpu.cpp verifies numerically) instead of writing to its device
// buffer directly -- CalciumCalculator has no public raw-pointer/notify seam like
// Axons/Dendrites, so this exercises the LazySyncedArray sync through the actual production
// entry point. A silently-broken device->host sync would leave "Calcium"/"Calcium Difference"
// reading the init()-time value (0) forever regardless of update_calcium calls.
TEST_F(NeuronMonitorGPUTest, testCalciumCalculatorMonitorReflectsDeviceWrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 4 };
    constexpr auto tau_C = RelearnTypes::calcium_type{ 5.0 };
    constexpr auto beta = RelearnTypes::calcium_type{ 0.5 };
    constexpr auto h = 10U;
    constexpr auto target = RelearnTypes::calcium_type{ 2.0 };

    auto cc = CalciumCalculator{};
    cc.set_beta(beta);
    cc.set_tau_C(tau_C);
    cc.set_h(h);
    cc.set_initial_calcium_calculator([](mpiPP::MPIRank, NeuronID::value_type) { return RelearnTypes::calcium_type{ 0.0 }; });
    cc.set_target_calcium_calculator([](mpiPP::MPIRank, NeuronID::value_type) { return target; });

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    cc.set_extra_infos(extra_infos);
    cc.init(number_neurons);

    // Alternate fired/not-fired per neuron so the two code paths inside the kernel both run.
    auto h_fired = std::vector<FiredStatus>(number_neurons);
    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        h_fired[id] = (id % 2 == 0) ? FiredStatus::Fired : FiredStatus::Inactive;
    }
    auto d_fired = LazySyncedArray<FiredStatus>(std::move(h_fired));

    cc.update_calcium(1, d_fired.get_device_ptr_const());

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_calcium");
    nm.set_output_path(output_path, true);

    cc.register_neuron_monitor(nm);

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        nm.register_neuron(id);
    }

    nm.record_data(0);
    nm.flush_current_contents();

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        const auto rows = read_csv_rows_by_column(output_path, id);
        ASSERT_EQ(rows.size(), 1U);

        const auto fired = (id % 2 == 0);
        const auto expected_calcium = expected_after_h_steps(RelearnTypes::calcium_type{ 0.0 }, h, tau_C, beta, fired);

        ASSERT_EQ(rows[0].at("Calcium"), format_like_ostream(static_cast<float>(expected_calcium)));
        ASSERT_EQ(rows[0].at("Target Calcium"), format_like_ostream(static_cast<float>(target)));
        ASSERT_EQ(rows[0].at("Calcium Difference"), format_like_ostream(static_cast<float>(target) - static_cast<float>(expected_calcium)));
    }
}

// Drives a real AEIFModel through its full GPU update_electrical_activity() pipeline (activity
// input -> update_activity() kernel -> fired-status commit/exchange) instead of writing to any
// device buffer directly -- there is no lower-level seam for NeuronModel's "x"/model-specific "w"
// or FiredStatusRecorder's "Hz", all three of which are LazySyncedArray-backed and only ever
// updated by the model's own GPU kernels. Checks two independent things:
//  1. "x"/"w" move away from their all-zero init() state -- if the LazySyncedArray sync from
//     device were broken, these would read exactly 0 forever regardless of how many update steps
//     ran.
//  2. The "Hz" monitor value (via FiredStatusRecorder::get_fired_recorder(), which -- see
//     FiredStatusRecorder.cpp -- had to be fixed this session to read the GPU-updated counter
//     array instead of a host-only one that the GPU kernel never touches) matches an
//     independently-tracked fired count obtained via NeuronModel::has_fired() (itself backed by
//     a separate LazySyncedArray<FiredStatus>), regardless of whether any neuron actually spikes.
TEST_F(NeuronMonitorGPUTest, testAEIFModelMonitorReflectsDeviceState) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_default_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(50.0);

    const auto h = models::AEIFModel::default_h;
    const auto parameters = models::aeif::Parameters<RelearnTypes::activity_type>{};
    auto model = models::AEIFModel(h, activity_input, fired_status_comm, parameters);

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 5 };
    auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons, mpiPP::MPIRank::root_rank());
    fired_status_comm->set_network_graph(network_graph);

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    model.set_extra_infos(extra_infos);
    model.init(number_neurons);

    auto fired_counts = std::vector<unsigned int>(number_neurons, 0U);
    constexpr auto num_steps = RelearnTypes::step_type{ 30 };
    for (auto step = RelearnTypes::step_type{ 1 }; step <= num_steps; ++step) {
        model.update_electrical_activity(step);
        for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
            if (model.has_fired(NeuronID{ id })) {
                ++fired_counts[id];
            }
        }
    }

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        EXPECT_NE(model.get_x(NeuronID{ id }), NeuronModel::activity_type{ 0.0 })
            << "neuron " << id << ": x never moved away from its init() default -- device sync likely broken";
    }

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_aeif");
    nm.set_output_path(output_path, true);

    model.register_neuron_monitor(nm);

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        nm.register_neuron(id);
    }

    nm.record_data(num_steps);
    nm.flush_current_contents();

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        const auto rows = read_csv_rows_by_column(output_path, id);
        ASSERT_EQ(rows.size(), 1U);

        ASSERT_NE(rows[0].at("x"), format_like_ostream(0.0F))
            << "neuron " << id << ": monitor's \"x\" reads the init() default despite direct getter showing otherwise";
        ASSERT_NE(rows[0].at("w"), format_like_ostream(0.0F))
            << "neuron " << id << ": monitor's \"w\" reads the init() default -- AEIF-specific device state not syncing";

        const auto expected_hz = utility::cast<float>(static_cast<double>(fired_counts[id]) / Config::neuron_monitor_log_step * 1000.0);
        ASSERT_EQ(rows[0].at("Hz"), format_like_ostream(expected_hz))
            << "neuron " << id << ": Hz (via get_fired_recorder) disagrees with an independently-tracked has_fired() count "
            << "-- the GPU-kernel-only firing path isn't reaching the monitor";
    }
}

// Same shape as testAEIFModelMonitorReflectsDeviceState, for FitzHughNagumoModel's "w". Unlike
// AEIFModel (which was missing its w.device_was_modified() call -- found and fixed this session),
// FitzHughNagumoModel already calls it correctly; this test is the regression guard for that.
TEST_F(NeuronMonitorGPUTest, testFitzHughNagumoModelMonitorReflectsDeviceState) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_default_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(50.0);

    const auto h = models::FitzHughNagumoModel::default_h;
    const auto parameters = models::fitzhughnagumo::Parameters<RelearnTypes::activity_type>{};
    auto model = models::FitzHughNagumoModel(h, activity_input, fired_status_comm, parameters);

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 5 };
    auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons, mpiPP::MPIRank::root_rank());
    fired_status_comm->set_network_graph(network_graph);

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    model.set_extra_infos(extra_infos);
    model.init(number_neurons);

    auto fired_counts = std::vector<unsigned int>(number_neurons, 0U);
    constexpr auto num_steps = RelearnTypes::step_type{ 30 };
    for (auto step = RelearnTypes::step_type{ 1 }; step <= num_steps; ++step) {
        model.update_electrical_activity(step);
        for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
            if (model.has_fired(NeuronID{ id })) {
                ++fired_counts[id];
            }
        }
    }

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        EXPECT_NE(model.get_x(NeuronID{ id }), NeuronModel::activity_type{ 0.0 })
            << "neuron " << id << ": x never moved away from its init() default -- device sync likely broken";
    }

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_fhn");
    nm.set_output_path(output_path, true);

    model.register_neuron_monitor(nm);

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        nm.register_neuron(id);
    }

    nm.record_data(num_steps);
    nm.flush_current_contents();

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        const auto rows = read_csv_rows_by_column(output_path, id);
        ASSERT_EQ(rows.size(), 1U);

        ASSERT_NE(rows[0].at("x"), format_like_ostream(0.0F))
            << "neuron " << id << ": monitor's \"x\" reads the init() default despite direct getter showing otherwise";
        ASSERT_NE(rows[0].at("w"), format_like_ostream(0.0F))
            << "neuron " << id << ": monitor's \"w\" reads the init() default -- FitzHughNagumo-specific device state not syncing";

        const auto expected_hz = utility::cast<float>(static_cast<double>(fired_counts[id]) / Config::neuron_monitor_log_step * 1000.0);
        ASSERT_EQ(rows[0].at("Hz"), format_like_ostream(expected_hz))
            << "neuron " << id << ": Hz disagrees with an independently-tracked has_fired() count";
    }
}

// Same shape again, for IzhikevichModel's "u".
TEST_F(NeuronMonitorGPUTest, testIzhikevichModelMonitorReflectsDeviceState) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_default_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(50.0);

    const auto h = models::IzhikevichModel::default_h;
    const auto parameters = models::izhikevich::Parameters<RelearnTypes::activity_type>{};
    auto model = models::IzhikevichModel(h, activity_input, fired_status_comm, parameters);

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 5 };
    auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons, mpiPP::MPIRank::root_rank());
    fired_status_comm->set_network_graph(network_graph);

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    model.set_extra_infos(extra_infos);
    model.init(number_neurons);

    auto fired_counts = std::vector<unsigned int>(number_neurons, 0U);
    constexpr auto num_steps = RelearnTypes::step_type{ 30 };
    for (auto step = RelearnTypes::step_type{ 1 }; step <= num_steps; ++step) {
        model.update_electrical_activity(step);
        for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
            if (model.has_fired(NeuronID{ id })) {
                ++fired_counts[id];
            }
        }
    }

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        EXPECT_NE(model.get_x(NeuronID{ id }), NeuronModel::activity_type{ 0.0 })
            << "neuron " << id << ": x never moved away from its init() default -- device sync likely broken";
    }

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_izhikevich");
    nm.set_output_path(output_path, true);

    model.register_neuron_monitor(nm);

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        nm.register_neuron(id);
    }

    nm.record_data(num_steps);
    nm.flush_current_contents();

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        const auto rows = read_csv_rows_by_column(output_path, id);
        ASSERT_EQ(rows.size(), 1U);

        ASSERT_NE(rows[0].at("x"), format_like_ostream(0.0F))
            << "neuron " << id << ": monitor's \"x\" reads the init() default despite direct getter showing otherwise";
        ASSERT_NE(rows[0].at("u"), format_like_ostream(0.0F))
            << "neuron " << id << ": monitor's \"u\" reads the init() default -- Izhikevich-specific device state not syncing";

        const auto expected_hz = utility::cast<float>(static_cast<double>(fired_counts[id]) / Config::neuron_monitor_log_step * 1000.0);
        ASSERT_EQ(rows[0].at("Hz"), format_like_ostream(expected_hz))
            << "neuron " << id << ": Hz disagrees with an independently-tracked has_fired() count";
    }
}

// PoissonModel's "r" (refractory_time) only ever changes once a neuron actually fires (it's a
// post-spike countdown, not a continuously-integrated ODE variable like AEIF/FHN/Izhikevich's
// adaptation term) -- so unlike the other three models, this test must force a real spike rather
// than just relying on "enough update steps eventually move it". A huge constant input pushes
// PoissonModel's x_val past any possible random threshold (drawn from [0,1)) on the very first
// integration sub-step, guaranteeing FiredStatus::Fired -- and therefore refractory_time
// becoming default_refractory_period (4) -- on step 1, deterministically.
TEST_F(NeuronMonitorGPUTest, testPoissonModelMonitorReflectsDeviceState) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_default_communicator(1);
    auto activity_input = ActivityInputFactory::construct_constant_activity(1.0e6);

    const auto h = models::PoissonModel::default_h;
    const auto parameters = models::poisson::Parameters<RelearnTypes::activity_type, unsigned int>{};
    auto model = models::PoissonModel(h, activity_input, fired_status_comm, parameters);

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 5 };
    auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons, mpiPP::MPIRank::root_rank());
    fired_status_comm->set_network_graph(network_graph);

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    model.set_extra_infos(extra_infos);
    model.init(number_neurons);

    auto fired_counts = std::vector<unsigned int>(number_neurons, 0U);
    // Only 2 steps: step 1 forces the spike (refractory_time -> 4), step 2 decrements it to 3 --
    // well short of default_refractory_period, so it stays observably nonzero either way.
    constexpr auto num_steps = RelearnTypes::step_type{ 2 };
    for (auto step = RelearnTypes::step_type{ 1 }; step <= num_steps; ++step) {
        model.update_electrical_activity(step);
        for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
            if (model.has_fired(NeuronID{ id })) {
                ++fired_counts[id];
            }
        }
    }

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        ASSERT_EQ(fired_counts[id], 1U) << "neuron " << id << ": expected exactly one forced spike on step 1";
    }

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_poisson");
    nm.set_output_path(output_path, true);

    model.register_neuron_monitor(nm);

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        nm.register_neuron(id);
    }

    nm.record_data(num_steps);
    nm.flush_current_contents();

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        const auto rows = read_csv_rows_by_column(output_path, id);
        ASSERT_EQ(rows.size(), 1U);

        ASSERT_NE(rows[0].at("r"), std::to_string(0U))
            << "neuron " << id << ": monitor's \"r\" (refractory_time) reads 0 despite a forced spike -- device sync likely broken";

        const auto expected_hz = utility::cast<float>(static_cast<double>(fired_counts[id]) / Config::neuron_monitor_log_step * 1000.0);
        ASSERT_EQ(rows[0].at("Hz"), format_like_ostream(expected_hz))
            << "neuron " << id << ": Hz disagrees with an independently-tracked has_fired() count";
    }
}

// ConstantActivityInput has no device state at all (the registered value is just the constant
// itself) -- included for completeness of "every CUDA-supported ActivityInput", not because
// there's a sync mechanism to break.
TEST_F(NeuronMonitorGPUTest, testConstantActivityInputMonitor) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    constexpr auto constant_input = 3.5;
    auto activity_input = ConstantActivityInput(1, static_cast<ActivityInput::activity_type>(constant_input));

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 3 };
    activity_input.init(number_neurons);

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_constant");
    nm.set_output_path(output_path, true);
    activity_input.register_neuron_monitor(nm);

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        nm.register_neuron(id);
    }
    nm.record_data(0);
    nm.flush_current_contents();

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        const auto rows = read_csv_rows_by_column(output_path, id);
        ASSERT_EQ(rows.size(), 1U);
        ASSERT_EQ(rows[0].at("Constant input"), format_like_ostream(utility::cast<float>(constant_input)));
    }
}

// NormalActivityInput's GPU update_input_range writes directly into the base ActivityInput's
// _input (unlike SynapticActivityInput's subclasses -- see below), and its monitor callback reads
// it back via get_input_internal() correctly. This is a plain regression guard, not a bug hunt.
TEST_F(NeuronMonitorGPUTest, testNormalActivityInputMonitorReflectsDeviceWrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    auto activity_input = NormalActivityInput(1, 0.0, 1.0);

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 5 };
    activity_input.init(number_neurons);

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    activity_input.set_extra_infos(extra_infos);

    activity_input.update_input(1);

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_normal");
    nm.set_output_path(output_path, true);
    activity_input.register_neuron_monitor(nm);

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        nm.register_neuron(id);
    }
    nm.record_data(1);
    nm.flush_current_contents();

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        const auto rows = read_csv_rows_by_column(output_path, id);
        ASSERT_EQ(rows.size(), 1U);

        const auto expected = utility::cast<float>(activity_input.get_input(NeuronID{ id }));
        ASSERT_EQ(rows[0].at("Normal Input"), format_like_ostream(expected))
            << "neuron " << id << ": monitor's \"Normal Input\" disagrees with get_input() -- device sync likely broken";
    }
}

// SynapticActivityInput::register_neuron_monitor used to read the base class's _input field
// directly -- but on the GPU path _input is never populated (SynapticActivityInput overrides
// get_input()/get_input(id) to combine d_input_local/d_input_distant instead, exactly because of
// this), so "Synaptic Input" silently read 0 forever in every GPU build. Found and fixed this
// session. This test drives a real all-to-all network graph with every neuron firing (mirroring
// test_synaptic_equally_weighted_activity_input.cpp's testUpdateInputFullNetworkGraph, which
// already validates get_input() itself) and checks the *monitor* value against the same expected
// sum -- the thing no existing test checked.
TEST_F(NeuronMonitorGPUTest, testSynapticEquallyWeightedActivityInputMonitorReflectsDeviceWrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 6 };
    auto network_graph = NetworkGraphFactory::construct_all_to_all_network_graph(number_neurons, mpiPP::MPIRank::root_rank());

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);

    auto fired_status_recorder = std::make_shared<FiredStatusRecorder>();
    fired_status_recorder->init(number_neurons);

    auto fired_status_comm = FiredStatusCommunicatorFactory::construct_default_communicator(1);
    fired_status_comm->init(number_neurons);
    fired_status_comm->set_network_graph(network_graph);
    fired_status_comm->set_fired_status_recorder(fired_status_recorder);

    auto activity_input = SynapticEquallyWeightedActivityInput(1, fired_status_comm, 1.0);
    activity_input.init(number_neurons);
    activity_input.set_network_graph(network_graph);
    activity_input.set_extra_infos(extra_infos);

    auto expected_input = std::vector<double>(number_neurons, 0.0);
    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto& [plastic_edges, _1] = network_graph->get_local_in_edges(neuron_id);
        fired_status_recorder->set_fired(neuron_id, FiredStatus::Fired);
        for (const auto& [_2, weight] : plastic_edges) {
            expected_input[neuron_id] += weight;
        }
    }

    activity_input.update_input(102);

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_synaptic");
    nm.set_output_path(output_path, true);
    activity_input.register_neuron_monitor(nm);

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        nm.register_neuron(id);
    }
    nm.record_data(102);
    nm.flush_current_contents();

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        const auto rows = read_csv_rows_by_column(output_path, id);
        ASSERT_EQ(rows.size(), 1U);

        const auto expected = utility::cast<float>(expected_input[id]);
        EXPECT_NE(expected, 0.0F) << "neuron " << id << ": test setup produced a zero expected input, assertion below would be vacuous";
        ASSERT_EQ(rows[0].at("Synaptic Input"), format_like_ostream(expected))
            << "neuron " << id << ": monitor's \"Synaptic Input\" reads _input (never populated on GPU) instead of get_input()";
    }
}

// CombinedActivityInput sums its sub-inputs' device pointers via sub_input_ptrs (a LazySyncedArray
// this session -- previously a hand-rolled cudaMalloc/cudaFree/cudaMemcpyAsync trio). Two
// ConstantActivityInputs with distinct values makes the expected sum unambiguous.
TEST_F(NeuronMonitorGPUTest, testCombinedActivityInputMonitorReflectsDeviceWrites) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        return;
    }

    constexpr auto input_1 = 2.0;
    constexpr auto input_2 = 5.5;
    auto sub_1 = std::make_shared<ConstantActivityInput>(1, static_cast<ActivityInput::activity_type>(input_1));
    auto sub_2 = std::make_shared<ConstantActivityInput>(1, static_cast<ActivityInput::activity_type>(input_2));

    auto combined = CombinedActivityInput(1, std::vector<std::shared_ptr<ActivityInput>>{ sub_1, sub_2 });

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 4 };
    combined.init(number_neurons);

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    combined.set_extra_infos(extra_infos);

    combined.update_input(1);

    auto nm = NeuronMonitor{};
    const auto output_path = std::filesystem::path("./test_neuron_monitor_gpu_combined");
    nm.set_output_path(output_path, true);
    combined.register_neuron_monitor(nm);

    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        nm.register_neuron(id);
    }
    nm.record_data(1);
    nm.flush_current_contents();

    const auto expected = utility::cast<float>(input_1 + input_2);
    for (auto id = RelearnTypes::number_neurons_type{ 0 }; id < number_neurons; ++id) {
        const auto rows = read_csv_rows_by_column(output_path, id);
        ASSERT_EQ(rows.size(), 1U);
        ASSERT_EQ(rows[0].at("Combined input"), format_like_ostream(expected))
            << "neuron " << id << ": monitor's \"Combined input\" disagrees with the sum of both sub-inputs' constants";
    }
}

#endif // RELEARN_CUDA_ENABLED
