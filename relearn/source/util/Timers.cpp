/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Timers.h"

#include "Config.h"

#include "io/LogFiles.h"
#include "sim/Essentials.h"
#include "types/BasicTypes.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPITypes.h>
#include <mpi-wrapper/reductions/MPIComponentwiseReductions.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wnonnull"
#include <boost/json/object.hpp>
#include <boost/json/serialize.hpp>
#include <boost/predef.h>

#include <mpi-wrapper/collectives/MPIGather.h>
#include <mpi-wrapper/collectives/MPIGatherV.h>

#include <string>
#pragma GCC diagnostic pop
#if BOOST_OS_WINDOWS
#include <boost/json/src.hpp>
#endif
#include <fmt/core.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iomanip>
#include <ios>
#include <numeric>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

std::string Timers::wall_clock_time() {
    auto now = std::chrono::system_clock::now();

    // Workaround as some compiler still do not support all C++20 features for timezones...
    // Convert to time_t for use with localtime
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

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

void Timers::insert(TimerHierarchy& node, const std::vector<std::uint64_t>& stack, const profile_accumulator& timer, const std::size_t offset) {
    //    const auto timer_id = stack[offset];
    //    RelearnException::check( get_timer_index(node.timer_region) == timer_id, "Node not matching");
    if (offset == stack.size()) {
        return;
    }
    const auto next_id = stack[offset];

    for (auto& child : node.children) {
        if (get_timer_index(child.timer_region) == next_id) {
            // Timer exists in tree
            insert(child, stack, timer, offset + 1);
            return;
        }
    }
    // Timer not in the tree yet.

    RelearnException::check(next_id < NUMBER_TIMERS, "Timers::insert: Next id too large");
    const auto& timer_name = TimerNames[next_id];
    auto new_node = TimerHierarchy{ static_cast<TimerRegion>(next_id), timer_name, {}, timer.time }; // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange) - range checked above
    node.children.emplace_back(new_node);
    insert(node.children[node.children.size() - 1], stack, timer, offset + 1);
}

std::optional<TimerHierarchy> Timers::to_timer_tree(const std::vector<std::vector<std::uint64_t>>& timer_stacks) {

    TimerHierarchy dummy{ TimerRegion::DUMMY, "Dummy", {}, 0 };

    for (const auto& stack : timer_stacks) {
        insert(dummy, stack, accumulators_[stack], 0);
    }
    RelearnException::check(dummy.children.size() == 1, "Timers::to_timer_tree: There are {} children of the dummy node", dummy.children.size());

    std::function<void(TimerHierarchy&)> propagate_data = [&propagate_data](TimerHierarchy& node) {
        auto sum_time = 0ULL;
        for (auto& child : node.children) {
            sum_time += child.time_ns;
            propagate_data(child);
        }
        node.time_childs_ns = sum_time;
    };
    if (dummy.children.empty()) {
        return std::nullopt;
    }
    propagate_data(dummy.children[0]);

    return dummy.children[0];
}

void Timers::print_human_readable(std::stringstream& console_output, const std::vector<std::uint64_t>& _timers_min, const std::vector<std::uint64_t>& _timers_max, const std::vector<std::uint64_t>& _timers_sum, const std::vector<std::uint64_t>& _timers_children) {
    // Helper function to print a single timer
    auto print_timer = [&console_output, &_timers_min, &_timers_max, &_timers_sum, &_timers_children](const std::string& timer_name,
                                                                                                      const std::uint16_t hierarchy_level, const std::size_t flat_idx, const std::uint64_t childs_sum, const std::uint64_t total_time, const auto number_ranks) {
        const auto min_time = static_cast<double>(_timers_min.at(flat_idx)) / 1e9;
        const auto avg_time = static_cast<double>(_timers_sum.at(flat_idx)) / 1e9 / number_ranks;
        const auto max_time = static_cast<double>(_timers_max.at(flat_idx)) / 1e9;
        const auto child_sum_time = static_cast<double>(_timers_children.at(flat_idx)) / 1e9;

        const auto childs_captured = child_sum_time / avg_time * 100;
        const auto absolute_percentage = avg_time / (static_cast<double>(total_time) / 1e9) * 100;

        auto str = timer_name;
        if (hierarchy_level > 0 && !str.empty()) {
            str.insert(0, std::size_t(2) * hierarchy_level, ' ');
        }

        console_output << std::left << std::setw(Constants::print_string_width) << str << ": " << std::right
                       << std::setw(Constants::print_width) << std::fixed
                       << std::setprecision(Constants::print_precision) << min_time << " | "
                       << std::setw(Constants::print_width) << std::fixed
                       << std::setprecision(Constants::print_precision) << avg_time << " | "
                       << std::setw(Constants::print_width) << std::fixed
                       << std::setprecision(Constants::print_precision) << max_time << " | "
                       << std::setw(Constants::print_width) << std::fixed
                       << std::setprecision(Constants::print_precision) << static_cast<double>(childs_sum) / 1e9 << " | "
                       << std::setw(Constants::print_width) << std::fixed
                       << std::setprecision(Constants::print_precision) << childs_captured << " | "
                       << std::setw(Constants::print_width) << std::fixed
                       << std::setprecision(Constants::print_precision) << absolute_percentage << '\n';
    };

    // Set precision for aligned double output
    console_output.precision(Constants::print_precision);

    console_output << "\n======== TIMERS GLOBAL OVER ALL RANKS ========\n";
    console_output << std::setw(Constants::print_string_width + 2) << std::right << "("
                   << std::setw(Constants::print_width) << "min"
                   << " | "
                   << std::setw(Constants::print_width) << " avg"
                   << " | "
                   << std::setw(Constants::print_width) << " max"
                   << " | "
                   << std::setw(Constants::print_width) << " avg-childs"
                   << ") sec."
                   << " | "
                   << std::setw(Constants::print_width) << " childs captured %"
                   << " | "
                   << std::setw(Constants::print_width) << " absolute-avg % "
                   << "\n";
    console_output << "TIMERS: main()\n";

    // Helper function to walk through the timer hierarchy
    const auto number_ranks = mpiPP::MPIInfo::get_number_ranks();
    std::function<std::size_t(TimerHierarchy, std::uint16_t, std::size_t, std::uint64_t)> walk;
    walk = [&walk, print_timer, number_ranks](const auto& n, const std::uint16_t hierarchy_depth, std::size_t i, const std::uint64_t total_time) {
        print_timer(n.timer_name, hierarchy_depth, i, n.time_childs_ns, total_time, number_ranks);
        for (const auto& child : n.children) {
            i++;
            i = walk(child, hierarchy_depth + 1, i, total_time);
        }
        return i;
    };
    RelearnException::check(root_tree.has_value(), "Timers::print_human_readable: root_tree has no value");
    const auto root_timer = root_tree.value(); // NOLINT(bugprone-unchecked-optional-access) - guarded by RelearnException::check above
    const auto total_time = root_timer.time_ns;
    walk(root_timer, 0, 0, total_time);

    console_output << "\n\n";
}

void Timers::print_human_readable() {
    if (mpiPP::MPIRank::root_rank() != mpiPP::MPIInfo::get_my_rank()) {
        return;
    }
    auto console_output = std::stringstream{};
    print_human_readable(console_output, timers_min, timers_max, timers_sum, timers_children);
    LogFiles::write_to_file(LogFiles::EventType::Timers, true, console_output.str());
}

void Timers::print_local_human_readable() {
    auto console_output = std::stringstream{};
    print_human_readable(console_output, timers_local, timers_local, timers_local, timers_children_local);
    LogFiles::write_to_file(LogFiles::EventType::TimersLocal, false, console_output.str());
}

void Timers::print_extrap(RelearnTypes::step_type total_steps, RelearnTypes::number_neurons_type number_neurons_per_rank) {
    if (mpiPP::MPIRank::root_rank() != mpiPP::MPIInfo::get_my_rank()) {
        return;
    }
    // TODO NOT WORKING WITH ranks > 1

    auto console_output = std::stringstream{};

    console_output << "PARAMETER " << "Steps" << "\n";
    console_output << "PARAMETER " << "Neurons-per-rank" << "\n";
    console_output << "POINTS " << "(" << total_steps << " " << number_neurons_per_rank << ") " << "\n\n";

    // Helper function to print a single timer
    auto print_timer = [&console_output](const std::string& timer_name,
                                         const std::uint64_t time_ns) {
        const auto min_time = time_ns;
        const auto avg_time = time_ns;
        const auto max_time = time_ns;

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

    // Helper function to walk through the timer hierarchy. Extra-P's flat REGION list has no
    // notion of nesting on its own, so each region is named with its full ancestor chain
    // (root+child+grandchild+...) to keep the parent/child structure visible in the exported file.
    std::function<void(TimerHierarchy, const std::string&)> walk;
    walk = [&walk, print_timer](const auto& n, const std::string& parent_path) {
        const auto full_name = parent_path.empty() ? n.timer_name : parent_path + "->" + n.timer_name;
        print_timer(full_name, n.time_ns);
        for (const auto& child : n.children) {
            walk(child, full_name);
        }
    };

    RelearnException::check(root_tree.has_value(), "Timers::print_extrap: root_tree has no value");
    const auto root_timer = root_tree.value(); // NOLINT(bugprone-unchecked-optional-access) - guarded by RelearnException::check above

    walk(root_timer, "");

    LogFiles::write_to_file(LogFiles::EventType::TimersExtraP, false, console_output.str());
}

void Timers::print_json() {
    if (mpiPP::MPIRank::root_rank() != mpiPP::MPIInfo::get_my_rank()) {
        return;
    }
    // TODO NOT WORKING WITH ranks > 1

    // Helper function to recursively build JSON object
    std::function<boost::json::object(const TimerHierarchy&)> build_json;
    build_json = [&build_json](const auto& node) -> boost::json::object {
        const auto min_time = node.time_ns;
        const auto avg_time = node.time_ns;
        const auto max_time = node.time_ns;

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

    RelearnException::check(root_tree.has_value(), "Timers::print_json: root_tree has no value");
    const auto root_json = build_json(root_tree.value()); // NOLINT(bugprone-unchecked-optional-access) - guarded by RelearnException::check above

    LogFiles::write_raw_string_to_file(LogFiles::EventType::TimersJson, false, boost::json::serialize(root_json));
}

template <typename T>
std::vector<std::vector<std::vector<T>>> gather_2d_vector(const std::vector<std::vector<T>>& local) {

    const auto size = mpiPP::MPIInfo::get_number_ranks();

    // 1. Flatten local data
    std::vector<T> flat;
    std::vector<int> row_sizes(local.size());

    for (size_t i = 0; i < local.size(); ++i) {
        row_sizes[i] = static_cast<int>(local[i].size());
        flat.insert(flat.end(), local[i].begin(), local[i].end());
    }

    const int local_rows = static_cast<int>(local.size());
    const int flat_size = static_cast<int>(flat.size());

    // 2. Gather metadata: number of rows per rank
    const auto all_rows = mpiPP::MPICollectives::gather(local_rows);

    // 3. Gather sizes of flattened data
    const auto all_flat_sizes = mpiPP::MPICollectives::gather(flat_size);

    // 4. Gather row_sizes (variable length → Gatherv)
    std::vector<int> row_displs(static_cast<std::size_t>(size));

    if (mpiPP::MPIInfo::is_root_rank()) {
        row_displs[0] = 0;
        for (auto i = std::size_t{ 1 }; i < static_cast<std::size_t>(mpiPP::MPIInfo::get_number_ranks()); ++i) {
            row_displs[i] = row_displs[i - 1] + all_rows[i - 1];
        }
    }

    const auto all_row_sizes = mpiPP::MPICollectives::gatherv(row_sizes, all_rows, row_displs, mpiPP::MPIRank::root_rank());

    // 5. Gather flat data
    std::vector<int> flat_displs(static_cast<std::size_t>(size));
    if (mpiPP::MPIInfo::is_root_rank()) {
        flat_displs[0] = 0;
        for (auto i = std::size_t{ 1 }; i < static_cast<std::size_t>(size); ++i) {
            flat_displs[i] = flat_displs[i - 1] + all_flat_sizes[i - 1];
        }
    }

    const auto all_flat = mpiPP::MPICollectives::gatherv(flat, all_flat_sizes, flat_displs, mpiPP::MPIRank::root_rank());

    // 6. Reconstruct on rank 0
    std::vector<std::vector<std::vector<T>>> result;
    if (mpiPP::MPIInfo::is_root_rank()) {
        result.resize(static_cast<std::size_t>(size));

        int flat_offset = 0;
        int row_offset = 0;

        for (auto r = std::size_t{ 0 }; r < static_cast<std::size_t>(size); ++r) {
            const int rows = all_rows[r];
            result[r].resize(static_cast<std::size_t>(rows));

            for (auto i = std::size_t{ 0 }; i < static_cast<std::size_t>(rows); ++i) {
                const int len = all_row_sizes[static_cast<std::size_t>(row_offset++)];
                result[r][i] = std::vector<T>(
                    all_flat.begin() + flat_offset,
                    all_flat.begin() + flat_offset + len);
                flat_offset += len;
            }
        }
    }

    return result; // leer auf != 0
}

template <typename T>
std::vector<std::vector<T>> broadcast_2d_vector(std::vector<std::vector<T>>& root_data) {

    const auto mpi_data_type = mpiPP::MPITypes::convert_type_to_mpi_type<T>();

    if (mpiPP::MPIInfo::is_root_rank()) {
        int number_rows = static_cast<int>(root_data.size());
        MPI_Bcast(&number_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> counts(root_data.size());
        for (auto i = 0U; i < root_data.size(); i++) {
            counts[i] = static_cast<int>(root_data[i].size());
        }

        MPI_Bcast(counts.data(), static_cast<int>(counts.size()), MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<T> flat_root_data{};
        for (const auto& row : root_data) {
            flat_root_data.insert(flat_root_data.end(), row.begin(), row.end());
        }

        const int flat_size = static_cast<int>(flat_root_data.size());
        MPI_Bcast(flat_root_data.data(), flat_size, mpi_data_type, 0, MPI_COMM_WORLD);
        return root_data;
    }
    int number_rows = 0;
    MPI_Bcast(&number_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> counts(static_cast<std::size_t>(number_rows));
    MPI_Bcast(counts.data(), static_cast<int>(counts.size()), MPI_INT, 0, MPI_COMM_WORLD);

    const int flat_size = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<T> flat_data(static_cast<std::size_t>(flat_size));
    MPI_Bcast(flat_data.data(), flat_size, mpi_data_type, 0, MPI_COMM_WORLD);

    std::vector<std::vector<T>> received_2d_data(static_cast<std::size_t>(number_rows));
    auto displ = 0;
    for (auto i = 0; i < number_rows; i++) {
        const auto row_size = counts[static_cast<std::size_t>(i)];
        received_2d_data[static_cast<std::size_t>(i)] = { flat_data.begin() + displ, flat_data.begin() + displ + row_size };
        displ += row_size;
    }
    return received_2d_data;
}

void Timers::collect_timer_data() {

    std::vector<std::vector<std::uint64_t>> timer_stacks{};
    for (const auto& [stack, timer] : accumulators_) {
        timer_stacks.push_back(stack);
    }
    std::sort(timer_stacks.begin(), timer_stacks.end());

    const auto all_timer_stacks = gather_2d_vector(timer_stacks);
    std::vector<std::vector<std::uint64_t>> flat_all_timer_stacks{};
    for (const auto& timer_stack_on_rank : all_timer_stacks) {
        flat_all_timer_stacks.insert(flat_all_timer_stacks.end(), timer_stack_on_rank.begin(), timer_stack_on_rank.end());
    }
    std::sort(flat_all_timer_stacks.begin(), flat_all_timer_stacks.end());
    const auto last_it = std::unique(flat_all_timer_stacks.begin(), flat_all_timer_stacks.end());
    flat_all_timer_stacks.erase(last_it, flat_all_timer_stacks.end());

    const auto global_timer_stack = broadcast_2d_vector(flat_all_timer_stacks);

    root_tree = to_timer_tree(global_timer_stack);

    std::vector<std::uint64_t> flat_timings{};
    std::vector<std::uint64_t> flat_childs{};
    std::function<void(TimerHierarchy)> walk;
    walk = [&walk, &flat_timings, &flat_childs](const auto& n) {
        flat_timings.emplace_back(n.time_ns);
        flat_childs.emplace_back(n.time_childs_ns);
        for (const auto& child : n.children) {
            walk(child);
        }
    };
    RelearnException::check(root_tree.has_value(), "Timers::gather: root_tree has no value after to_timer_tree");
    walk(root_tree.value());

    timers_local = flat_timings;
    timers_children_local = flat_childs;

    timers_min = mpiPP::MPIReductions::reduce_componentwise_min(timers_local);
    timers_sum = mpiPP::MPIReductions::reduce_componentwise_sum(timers_local);
    timers_max = mpiPP::MPIReductions::reduce_componentwise_max(timers_local);
    timers_children = mpiPP::MPIReductions::reduce_componentwise_max(timers_children_local);
}