/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Timers.h"

#include "Config.h"
#include "Types.h"

#include "io/LogFiles.h"
#include "sim/Essentials.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"
#include "mpi-wrapper/MPIReductions.h"

#include <boost/json/object.hpp>
#include <boost/json/serialize.hpp>
#include <boost/predef.h>
#if BOOST_OS_WINDOWS
#include <boost/json/src.hpp>
#endif
#include <fmt/core.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <iomanip>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

std::string Timers::wall_clock_time() {
    auto now = std::chrono::system_clock::now();

    // Workaround as some compiler still do not support all C++20 features for timezones...
    // Convert to time_t for use with localtime
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    // Convert to local time
    std::tm time_info{};
#ifdef _WIN32
    localtime_s(&time_info, &now_time);
#else
    localtime_r(&now_time, &time_info);
#endif

    // Format the time with std::put_time at runtime
    std::ostringstream oss;
    oss << std::put_time(&time_info, "%a %b %d %H:%M:%S %Y");
    return oss.str();
}

void Timers::print_human_readable(const std::unique_ptr<Essentials>& essentials) {
    if (mpiPP::MPIRank::root_rank() != mpiPP::MPIInfo::get_my_rank()) {
        return;
    }

    auto console_output = std::stringstream{};

    // Helper function to print a single timer
    auto print_timer = [&console_output](const std::string& timer_name,
                                         const TimerRegion timer,
                                         const std::uint16_t hierarchy_level) {
        const auto timer_index = get_timer_index(timer);
        const auto min_time = timers_min.at(timer_index);
        const auto avg_time = timers_sum.at(timer_index);
        const auto max_time = timers_max.at(timer_index);

        if (max_time == 0.0) {
            // Do not print the timer if there is nothing timed
            return;
        }

        auto str = timer_name;
        str.insert(0, static_cast<std::size_t>(2) * hierarchy_level, ' ');

        console_output << std::left << std::setw(Constants::print_string_width) << str << ": " << std::right
                       << std::setw(Constants::print_width) << std::fixed
                       << std::setprecision(Constants::print_precision) << min_time << " | "
                       << std::setw(Constants::print_width) << std::fixed
                       << std::setprecision(Constants::print_precision) << avg_time << " | "
                       << std::setw(Constants::print_width) << std::fixed
                       << std::setprecision(Constants::print_precision) << max_time << '\n';
    };

    // Divide second entry of (min, sum, max), i.e., sum, by the number of ranks
    // so that sum becomes average
    for (auto i = index_type{ 0 }; i < NUMBER_TIMERS; i++) {
        // NOLINTNEXTLINE
        timers_sum[i] /= mpiPP::MPIInfo::get_number_ranks();
    }

    // Set precision for aligned double output
    console_output.precision(Constants::print_precision);

    console_output << "\n======== TIMERS GLOBAL OVER ALL RANKS ========\n";
    console_output << std::setw(Constants::print_string_width + 2) << std::right << "("
                   << std::setw(Constants::print_width) << "min"
                   << " | "
                   << std::setw(Constants::print_width) << " avg"
                   << " | "
                   << std::setw(Constants::print_width) << " max"
                   << ") sec.\n";
    console_output << "TIMERS: main()\n";

    // Helper function to walk through the timer hierarchy
    std::function<void(TimerHierarchy, std::uint16_t)> walk;
    walk = [&walk, print_timer](const auto& n, const std::uint16_t hierarchy_depth) {
        print_timer(n.timer_name, n.timer_region, hierarchy_depth);
        for (const auto& child : n.children) {
            walk(child, hierarchy_depth + 1);
        }
    };

    walk(root_timer, 0);

    console_output << "\n\n";

    LogFiles::write_to_file(LogFiles::EventType::Timers, true, console_output.str());

    const auto average_simulation_time = timers_sum.at(get_timer_index(TimerRegion::SIMULATION_LOOP));
    essentials->insert("Simulation-Time-Seconds", average_simulation_time);
}

void Timers::print_extrap(RelearnTypes::step_type total_steps, RelearnTypes::number_neurons_type number_neurons_per_rank) {
    if (mpiPP::MPIRank::root_rank() != mpiPP::MPIInfo::get_my_rank()) {
        return;
    }

    auto console_output = std::stringstream{};

    console_output << "PARAMETER " << "Steps" << "\n";
    console_output << "PARAMETER " << "Neurons-per-rank" << "\n";
    console_output << "POINTS " << "(" << total_steps << " " << number_neurons_per_rank << ") " << "\n\n";

    // Helper function to print a single timer
    auto print_timer = [&console_output](const std::string& timer_name,
                                         const TimerRegion timer) {
        const auto timer_index = get_timer_index(timer);
        const auto min_time = timers_min.at(timer_index);
        const auto avg_time = timers_sum.at(timer_index);
        const auto max_time = timers_max.at(timer_index);

        console_output << "REGION " << timer_name << "\n"
                       << "METRIC time-min\n"
                       << "DATA " << std::fixed << std::setprecision(Constants::print_precision)
                       << min_time << '\n'
                       << "METRIC time-avg\n"
                       << "DATA " << std::fixed << std::setprecision(Constants::print_precision)
                       << avg_time << '\n'
                       << "METRIC time-max\n"
                       << "DATA " << std::fixed << std::setprecision(Constants::print_precision)
                       << max_time << "\n\n";
    };

    // Helper function to walk through the timer hierarchy
    std::function<void(TimerHierarchy)> walk;
    walk = [&walk, print_timer](const auto& n) {
        print_timer(n.timer_name, n.timer_region);
        for (const auto& child : n.children) {
            walk(child);
        }
    };

    walk(root_timer);

    LogFiles::write_to_file(LogFiles::EventType::TimersExtraP, false, console_output.str());
}

void Timers::print_json() {
    if (mpiPP::MPIRank::root_rank() != mpiPP::MPIInfo::get_my_rank()) {
        return;
    }

    // Helper function to recursively build JSON object
    std::function<boost::json::object(const TimerHierarchy&)> build_json;
    build_json = [&build_json](const auto& node) -> boost::json::object {
        const auto timer_index = get_timer_index(node.timer_region);
        const auto min_time = timers_min.at(timer_index);
        const auto avg_time = timers_sum.at(timer_index);
        const auto max_time = timers_max.at(timer_index);

        boost::json::object timer_json;
        timer_json["timer_name"] = node.timer_name;
        timer_json["min_time"] = min_time;
        timer_json["avg_time"] = avg_time;
        timer_json["max_time"] = max_time;

        // Recursively process child nodes
        auto children = boost::json::array{};
        for (const auto& child : node.children) {
            auto child_json = build_json(child);
            if (!child_json.empty()) {
                children.push_back(child_json);
            }
        }
        if (!children.empty()) {
            timer_json["children"] = std::move(children);
        }

        return timer_json;
    };

    const auto root_json = build_json(root_timer);

    LogFiles::write_raw_string_to_file(LogFiles::EventType::TimersJson, false, boost::json::serialize(root_json));
}

void Timers::collect_timer_data() {
    auto timers_local = std::array<double, NUMBER_TIMERS>{};
    auto local_timer_output = std::stringstream{};

    for (auto i = 0U; i < NUMBER_TIMERS; ++i) {
        const auto timer = static_cast<TimerRegion>(i);
        const auto elapsed = get_elapsed(timer);

        local_timer_output << elapsed.count() << '\n';

        const auto counted = elapsed.count();
        const auto seconds = static_cast<double>(counted) * 1e-9;

        // NOLINTNEXTLINE
        timers_local[i] = seconds;
    }

    LogFiles::write_to_file(LogFiles::EventType::TimersLocal, false, local_timer_output.str());

    Timers::timers_min = mpiPP::MPIReductions::reduce_componentwise_min(timers_local);
    Timers::timers_sum = mpiPP::MPIReductions::reduce_componentwise_sum(timers_local);
    Timers::timers_max = mpiPP::MPIReductions::reduce_componentwise_max(timers_local);
}
