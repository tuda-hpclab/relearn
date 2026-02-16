#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h> // IWYU pragma: keep
#include <range/v3/algorithm/fold_left.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/transform.hpp>

#include <cstdint>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

 /**
  * Specifies categories an event can have.
  * These are custom an can be extended
  */
enum class EventCategory : std::uint8_t {
	async,
	calculation,
	mpi,
	sync,
};

/**
 * @brief Pretty-prints the event category to the chosen stream
 * @param out The stream to which to print the event category
 * @param event_category The event category to print
 * @return The argument out, now altered with the event category
 */
inline std::ostream& operator<<(std::ostream& out, const EventCategory event_category) {
	switch (event_category) {
	case EventCategory::async:
		return out << "async";
	case EventCategory::calculation:
		return out << "calculation";
	case EventCategory::mpi:
		return out << "mpi";
	case EventCategory::sync:
		return out << "sync";
	default:
		return out << "cat-unknown";
	}
}

template <>
struct fmt::formatter<EventCategory> : ostream_formatter {};

/**
 * Specifies the phase of an event. These are not custom and should not change. See:
 * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
 */
enum class EventPhase : std::uint8_t {
	DurationBegin,
	DurationEnd,
	Complete,
	Instant,
	Counter,
	AsyncNestableStart,
	AsyncNestableInstant,
	AsyncNestableEnd,
	FlowStart,
	FlowStep,
	FlowEnd,
	ObjectCreated,
	ObjectSnapshot,
	ObjectDestroyed,
	Metadata,
	MemoryDumpGlobal,
	MemoryDumpProcess,
	Mark,
	ClockSync,
	ContextBegin,
	ContextEnd
};

/**
 * @brief Pretty-prints the event phase to the chosen stream
 * @param out The stream to which to print the event phase
 * @param event_phase The event phase to print
 * @return The argument out, now altered with the event phase
 */
inline std::ostream& operator<<(std::ostream& out, const EventPhase event_phase) {
	switch (event_phase) {
	case EventPhase::DurationBegin:
		return out << 'B';
	case EventPhase::DurationEnd:
		return out << 'E';
	case EventPhase::Complete:
		return out << 'X';
	case EventPhase::Instant:
		return out << 'i';
	case EventPhase::Counter:
		return out << 'C';
	case EventPhase::AsyncNestableStart:
		return out << 'b';
	case EventPhase::AsyncNestableInstant:
		return out << 'n';
	case EventPhase::AsyncNestableEnd:
		return out << 'e';
	case EventPhase::FlowStart:
		return out << 's';
	case EventPhase::FlowStep:
		return out << 't';
	case EventPhase::FlowEnd:
		return out << 'f';
	case EventPhase::ObjectCreated:
		return out << 'N';
	case EventPhase::ObjectSnapshot:
		return out << 'O';
	case EventPhase::ObjectDestroyed:
		return out << 'D';
	case EventPhase::Metadata:
		return out << 'M';
	case EventPhase::MemoryDumpGlobal:
		return out << 'V';
	case EventPhase::MemoryDumpProcess:
		return out << 'v';
	case EventPhase::Mark:
		return out << 'R';
	case EventPhase::ClockSync:
		return out << 'c';
	case EventPhase::ContextBegin:
		return out << '(';
	case EventPhase::ContextEnd:
		return out << ')';
	default:
		return out << '?';
	}
}

template <>
struct fmt::formatter<EventPhase> : ostream_formatter {};

/**
 * An instant event can be of global, process, and thread level.
 * There is a forth level (default), which collapses to thread.
 */
enum class InstantEventScope : std::uint8_t {
	Global,
	Process,
	Thread
};

/**
 * @brief Pretty-prints the instant event scope to the chosen stream
 * @param out The stream to which to print the instant event scope
 * @param event_scope The instant event scope to print
 * @return The argument out, now altered with the instant event scope
 */
inline std::ostream& operator<<(std::ostream& out, const InstantEventScope event_scope) {
	switch (event_scope) {
	case InstantEventScope::Global:
		return out << 'g';
	case InstantEventScope::Process:
		return out << 'p';
	case InstantEventScope::Thread:
		return out << 't';
	default:
		return out << "?";
	}
}

template <>
struct fmt::formatter<InstantEventScope> : ostream_formatter {};

/**
 * Provides the possibility to create events in the style of the Google Trace Event Format:
 * https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
 */
class Event {
public:
	/**
	 * @brief Creates an event that signals the begin of some duration
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param tracing_clock The clock at the start of the event
	 * @param process_id The id of the process to which the event belongs
	 * @param thread_id The id of the thread to which the event belongs
	 * @param args The arguments for the event, can be empty
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_duration_begin_event(std::string&& name, std::set<EventCategory>&& categories,
		const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id, std::vector<std::pair<std::string, std::string>>&& args) {
		return { std::move(name), std::move(categories), EventPhase::DurationBegin, {}, tracing_clock, process_id, thread_id, std::move(args), {} };
	}

	/**
	 * @brief Creates an event that signals the begin of some duration with default arguments for process-id, thread-id, and tracing-clock
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param args The arguments for the event, can be empty
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_duration_begin_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args);

	/**
	 * @brief Creates an event that signals the begin of some duration with default arguments for process-id, thread-id, and tracing-clock.
	 *      Prints the event directly to the file (if event tracing is enabled)
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param args The arguments for the event, can be empty
	 * @param flush True if the file should be flushed
	 */
	static void create_and_print_duration_begin_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args, bool flush = false);

	/**
	 * @brief Creates an event that signals the end of some duration. Always ends the latest begun event
	 * @param tracing_clock The clock at the start of the event
	 * @param process_id The id of the process to which the event belongs
	 * @param thread_id The id of the thread to which the event belongs
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_duration_end_event(const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id) {
		return { {}, {}, EventPhase::DurationEnd, {}, tracing_clock, process_id, thread_id, {}, {} };
	}

	/**
	 * @brief Creates an event that signals the end of some duration with default arguments for process-id, thread-id, and tracing-clock. Always ends the latest begun event
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_duration_end_event();

	/**
	 * @brief Creates an event that signals the end of some duration with default arguments for process-id, thread-id, and tracing-clock. Always ends the latest begun event.
	 *      Prints the event directly to the file (if event tracing is enabled)
	 * @param flush True if the file should be flushed
	 */
	static void create_and_print_duration_end_event(bool flush = false);

	/**
	 * @brief Creates an event that signals the completion of some event (not a duration)
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param duration The duration of the event
	 * @param tracing_clock The clock at the start of the event
	 * @param process_id The id of the process to which the event belongs
	 * @param thread_id The id of the thread to which the event belongs
	 * @param args The arguments for the event, can be empty
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_complete_event(std::string&& name, std::set<EventCategory>&& categories, const double duration,
		const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id, std::vector<std::pair<std::string, std::string>>&& args) {
		return { std::move(name), std::move(categories), EventPhase::Complete, {}, tracing_clock, process_id, thread_id, std::move(args), duration };
	}

	/**
	 * @brief Creates an event that signals the completion of some event (not a duration event) with default arguments for process-id, thread-id, and tracing-clock.
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param duration The duration of the event
	 * @param args The arguments for the event, can be empty
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_complete_event(std::string&& name, std::set<EventCategory>&& categories, double duration, std::vector<std::pair<std::string, std::string>>&& args);

	/**
	 * @brief Creates an event that signals the completion of some event (not a duration event) with default arguments for process-id, thread-id, and tracing-clock.
	 *      Prints the event directly to the file (if event tracing is enabled)
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param duration The duration of the event
	 * @param args The arguments for the event, can be empty
	 * @param flush True if the file should be flushed
	 */
	static void create_and_print_complete_event(std::string&& name, std::set<EventCategory>&& categories, double duration, std::vector<std::pair<std::string, std::string>>&& args, bool flush = false);

	/**
	 * @brief Creates an event that signals same instant
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param scope The scope of the event, can be global, process, or thread
	 * @param tracing_clock The clock at the start of the event
	 * @param process_id The id of the process to which the event belongs
	 * @param thread_id The id of the thread to which the event belongs
	 * @param args The arguments for the event, can be empty
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_instant_event(std::string&& name, std::set<EventCategory>&& categories, const InstantEventScope scope,
		const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id, std::vector<std::pair<std::string, std::string>>&& args) {
		return { std::move(name), std::move(categories), EventPhase::Instant, scope, tracing_clock, process_id, thread_id, std::move(args), {} };
	}

	/**
	 * @brief Creates an event that signals same instant with default arguments for process-id, thread-id, and tracing-clock.
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param scope The scope of the event, can be global, process, or thread
	 * @param args The arguments for the event, can be empty
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_instant_event(std::string&& name, std::set<EventCategory>&& categories, InstantEventScope scope, std::vector<std::pair<std::string, std::string>>&& args);

	/**
	 * @brief Creates an event that signals same instant with default arguments for process-id, thread-id, and tracing-clock.
	 *      Prints the event directly to the file (if event tracing is enabled)
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param scope The scope of the event, can be global, process, or thread
	 * @param args The arguments for the event, can be empty
	 * @param flush True if the file should be flushed
	 */
	static void create_and_print_instant_event(std::string&& name, std::set<EventCategory>&& categories, InstantEventScope scope, std::vector<std::pair<std::string, std::string>>&& args, bool flush = false);

	/**
	 * @brief Creates an event that signals the change of some counter (values specified in args)
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param tracing_clock The clock at the start of the event
	 * @param process_id The id of the process to which the event belongs
	 * @param thread_id The id of the thread to which the event belongs
	 * @param args The arguments for the event, should not be empty
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_counter_event(std::string&& name, std::set<EventCategory>&& categories,
		const double tracing_clock, const std::uint64_t process_id, const std::uint64_t thread_id, std::vector<std::pair<std::string, std::string>>&& args) {
		return { std::move(name), std::move(categories), EventPhase::Counter, {}, tracing_clock, process_id, thread_id, std::move(args), {} };
	}

	/**
	 * @brief Creates an event that signals the change of some counter (values specified in args) with default arguments for process-id, thread-id, and tracing-clock.
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param args The arguments for the event, should not be empty
	 * @return The created object that can be printed using operator<<
	 */
	static Event create_counter_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args);

	/**
	 * @brief Creates an event that signals the change of some counter (values specified in args) with default arguments for process-id, thread-id, and tracing-clock.
	 *      Prints the event directly to the file (if event tracing is enabled)
	 * @param name The name of the event
	 * @param categories The categories for the event, can be empty
	 * @param args The arguments for the event, should not be empty
	 * @param flush True if the file should be flushed
	 */
	static void create_and_print_counter_event(std::string&& name, std::set<EventCategory>&& categories, std::vector<std::pair<std::string, std::string>>&& args, bool flush = false);

private:
	Event(std::optional<std::string>&& _name, std::set<EventCategory>&& _categories, const EventPhase _phase,
		std::optional<InstantEventScope>&& _scope, const double _tracing_clock, const std::uint64_t _process_id,
		const std::uint64_t _thread_id, std::vector<std::pair<std::string, std::string>>&& _arguments, std::optional<double>&& _duration)
		: name(std::move(_name))
		, categories(std::move(_categories))
		, phase(_phase)
		, scope(std::move(_scope))
		, tracing_clock(_tracing_clock)
		, process_id(_process_id)
		, thread_id(_thread_id)
		, arguments(std::move(_arguments))
		, duration(std::move(_duration)) {
	}

	std::optional<std::string> name{};
	std::set<EventCategory> categories{};
	EventPhase phase{};
	std::optional<InstantEventScope> scope{};
	double tracing_clock{};
	std::uint64_t process_id{};
	std::uint64_t thread_id{};
	std::vector<std::pair<std::string, std::string>> arguments{};
	std::optional<double> duration{};

	friend std::ostream& operator<<(std::ostream& out, const Event& event);
};

/**
 * @brief Prints the event in JSON format as one line to the stream. Does not add a line break
 * @param out The stream to which to print the instant event scope
 * @param event The event to print
 * @return The argument out, now altered with the event
 */
inline std::ostream& operator<<(std::ostream& out, const Event& event) {
	out << '{';

	if (event.name.has_value()) {
		out << fmt::format(R"("name": "{}", )", event.name.value());
	}

	out << fmt::format(R"("ph": "{}", "pid": {}, "tid": {}, "ts": {})",
		event.phase, event.process_id, event.thread_id, event.tracing_clock);

	if (event.duration.has_value()) {
		out << fmt::format(R"(, "dur": {})", event.duration.value());
	}

	if (event.scope.has_value()) {
		out << fmt::format(R"(, "s": "{}")", event.scope.value());
	}

	if (!event.categories.empty()) {
		out << fmt::format(R"(, "cat": "{:n}")", event.categories);
	}

	if (!event.arguments.empty()) {
		out << fmt::format(R"(, "args": {{{}}})",
			fmt::join(event.arguments
				| ranges::views::transform([](const auto& tup) {
					const auto& [first_name, first_value] = tup;
					return fmt::format(R"("{}": {})", first_name, first_value);
					}),
				", "));
	}

	out << '}';

	return out;
}

template <>
struct fmt::formatter<Event> : ostream_formatter {};
