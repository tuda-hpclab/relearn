/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Event.h"

#include "io/LogFiles.h"
#include "util/OMPHelper.h"

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <chrono>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

Event Event::create_duration_begin_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args) {
    const auto dur_since_epoch = std::chrono::system_clock::now().time_since_epoch();
    const auto dur_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dur_since_epoch).count();
    const auto dur_in_ms = static_cast<double>(dur_in_ns) / 1000000.0;

    const auto process_id = static_cast<std::uint64_t>(mpiPP::MPIInfo::get_my_rank().get_rank());
    const auto thread_id = static_cast<std::uint64_t>(omp_get_thread_num());

    return create_duration_begin_event(std::move(name), std::move(categories), dur_in_ms, process_id, thread_id, std::move(args));
}

void Event::create_and_print_duration_begin_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args, const bool flush) {
    if (LogFiles::get_log_status(LogFiles::EventType::Events)) {
        return;
    }

    const auto event = create_duration_begin_event(std::move(name), std::move(categories), std::move(args));
    LogFiles::add_event_to_trace(event);

    if (flush) {
        LogFiles::flush_file(LogFiles::EventType::Events);
    }
}

Event Event::create_duration_end_event() {
    const auto dur_since_epoch = std::chrono::system_clock::now().time_since_epoch();
    const auto dur_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dur_since_epoch).count();
    const auto dur_in_ms = static_cast<double>(dur_in_ns) / 1000000.0;

    const auto process_id = static_cast<std::uint64_t>(mpiPP::MPIInfo::get_my_rank().get_rank());
    const auto thread_id = static_cast<std::uint64_t>(omp_get_thread_num());

    return create_duration_end_event(dur_in_ms, process_id, thread_id);
}

void Event::create_and_print_duration_end_event(bool flush) {
    if (LogFiles::get_log_status(LogFiles::EventType::Events)) {
        return;
    }

    const auto event = create_duration_end_event();
    LogFiles::add_event_to_trace(event);

    if (flush) {
        LogFiles::flush_file(LogFiles::EventType::Events);
    }
}

Event Event::create_complete_event(std::string&& name, std::set<EventCategory>&& categories, const double duration, std::vector<std::pair<std::string, std::string>>&& args) {
    const auto dur_since_epoch = std::chrono::system_clock::now().time_since_epoch();
    const auto dur_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dur_since_epoch).count();
    const auto dur_in_ms = static_cast<double>(dur_in_ns) / 1000000.0;

    const auto process_id = static_cast<std::uint64_t>(mpiPP::MPIInfo::get_my_rank().get_rank());
    const auto thread_id = static_cast<std::uint64_t>(omp_get_thread_num());

    return create_complete_event(std::move(name), std::move(categories), duration, dur_in_ms, process_id, thread_id, std::move(args));
}

void Event::create_and_print_complete_event(std::string&& name, std::set<EventCategory>&& categories, double duration, std::vector<std::pair<std::string, std::string>>&& args, bool flush) {
    if (LogFiles::get_log_status(LogFiles::EventType::Events)) {
        return;
    }

    const auto event = create_complete_event(std::move(name), std::move(categories), duration, std::move(args));
    LogFiles::add_event_to_trace(event);

    if (flush) {
        LogFiles::flush_file(LogFiles::EventType::Events);
    }
}

Event Event::create_instant_event(std::string&& name, std::set<EventCategory>&& categories, const InstantEventScope scope, std::vector<std::pair<std::string, std::string>>&& args) {
    const auto dur_since_epoch = std::chrono::system_clock::now().time_since_epoch();
    const auto dur_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dur_since_epoch).count();
    const auto dur_in_ms = static_cast<double>(dur_in_ns) / 1000000.0;

    const auto process_id = static_cast<std::uint64_t>(mpiPP::MPIInfo::get_my_rank().get_rank());
    const auto thread_id = static_cast<std::uint64_t>(omp_get_thread_num());

    return create_instant_event(std::move(name), std::move(categories), scope, dur_in_ms, process_id, thread_id, std::move(args));
}

void Event::create_and_print_instant_event(std::string&& name, std::set<EventCategory>&& categories, InstantEventScope scope, std::vector<std::pair<std::string, std::string>>&& args, bool flush) {
    if (LogFiles::get_log_status(LogFiles::EventType::Events)) {
        return;
    }

    const auto event = create_instant_event(std::move(name), std::move(categories), scope, std::move(args));
    LogFiles::add_event_to_trace(event);

    if (flush) {
        LogFiles::flush_file(LogFiles::EventType::Events);
    }
}

Event Event::create_counter_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args) {
    const auto dur_since_epoch = std::chrono::system_clock::now().time_since_epoch();
    const auto dur_in_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(dur_since_epoch).count();
    const auto dur_in_ms = static_cast<double>(dur_in_ns) / 1000000.0;

    const auto process_id = static_cast<std::uint64_t>(mpiPP::MPIInfo::get_my_rank().get_rank());
    const auto thread_id = static_cast<std::uint64_t>(omp_get_thread_num());

    return create_counter_event(std::move(name), std::move(categories), dur_in_ms, process_id, thread_id, std::move(args));
}

void Event::create_and_print_counter_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args, bool flush) {
    if (LogFiles::get_log_status(LogFiles::EventType::Events)) {
        return;
    }

    const auto event = create_counter_event(std::move(name), std::move(categories), std::move(args));
    LogFiles::add_event_to_trace(event);

    if (flush) {
        LogFiles::flush_file(LogFiles::EventType::Events);
    }
}
