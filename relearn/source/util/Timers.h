#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "types/BasicTypes.h"
#include "util/RelearnException.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef RELEARN_CUDA_ENABLED
#include <nvtx3/nvToolsExt.h>
#endif

class Essentials;

/**
 * This type allows type-safe specification of a specific timer
 */

constexpr std::size_t NUMBER_TIMERS = 160;

enum class TimerRegion : std::uint8_t {
#define X(name, str) name,
#include "timer_regions.def"
#undef X
};

static constexpr std::array<const char*, static_cast<size_t>(TimerRegion::DUMMY) + 1>
    TimerNames = {
#define X(name, str) str,
#include "timer_regions.def"
#undef X
    };

struct TimerHierarchy {
    /**
     * Struct that represent the hierarchy of the timers.
     * A root timer contains all other timers. Each timer can have children that further split its run time. The parent timer is at least the sum of all its child timers but can also be larger.
     */
    TimerRegion timer_region{};
    std::string timer_name;
    std::vector<TimerHierarchy> children;
    std::uint64_t time_ns{};
    std::uint64_t time_childs_ns{};
};

// clang-format off
// clang-format on

struct profile_accumulator {
    std::size_t count = 0;
    std::uint64_t time = 0.;
    std::chrono::steady_clock::time_point start_time;
    bool running = false;
};
using timer_stack = std::vector<std::size_t>;

struct TimerStackHash {
    std::size_t operator()(const timer_stack& v) const {
        std::size_t seed = v.size();
        for (const auto& i : v) {
            seed ^= std::hash<std::uint64_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

/**
 * This class is used to collect all sorts of different timers (see TimerRegion).
 * It provides an interface to start, stop, and print_human_readable the timers
 */
class Timers {
    using time_point = std::chrono::high_resolution_clock::time_point;
    using index_type = std::vector<time_point>::size_type;

public:
    /**
     * @brief Starts the respective timer
     * @param timer The timer to start
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void start(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::start: timer_id was {}", timer_id);
        current_timer_stack.push_back(timer_id);
        auto& cur_acc = accumulators_[current_timer_stack];
        RelearnException::check(!cur_acc.running, "Timers::start: you entered the timer twice ");
#ifdef RELEARN_CUDA_ENABLED
        nvtxRangePushA(TimerNames[timer_id]);
#endif

        cur_acc.start_time = std::chrono::steady_clock::now();
        cur_acc.running = true;
    }

    /**
     * @brief Stops the respective timer
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void stop(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::stop: timer_id was: {}", timer_id);

        RelearnException::check(
            current_timer_stack[current_timer_stack.size() - 1] == timer_id,
            "Timers::stop: without matching Timers::start: Trying to leave {} but currently in {}", TimerNames[timer_id],
            TimerNames[current_timer_stack[current_timer_stack.size() - 1]]);
        auto& cur_acc = accumulators_[current_timer_stack];

        // calculate the elapsed time before any other steps, to increase accuracy.
        const auto start = cur_acc.start_time;
        const auto stop = std::chrono::steady_clock::now();
        const auto delta_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        RelearnException::check(cur_acc.running, "Timers::stop: Timer {} is not running", timer_id, TimerNames[timer_id]);
        RelearnException::check(delta_ns >= 0, "Timers::stop: Delta is negative or 0 {} {}", delta_ns, TimerNames[timer_id]);
#ifdef RELEARN_CUDA_ENABLED
        nvtxRangePop();
#endif

        cur_acc.count++;
        cur_acc.time += static_cast<uint64_t>(delta_ns);
        cur_acc.running = false;

        current_timer_stack.erase(
            std::next(current_timer_stack.begin(), static_cast<std::int32_t>(current_timer_stack.size() - 1)));
    }

    static std::optional<TimerHierarchy> to_timer_tree(const std::vector<std::vector<std::uint64_t>>& timer_stacks);

    static void insert(TimerHierarchy& node, const std::vector<std::uint64_t>& stack, const profile_accumulator& timer,
                       std::size_t offset);

    /**
     * @brief Stops the respective timer and adds the elapsed time
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void stop_and_add(const TimerRegion timer) {
        stop(timer);
    }

    /**
     * @brief Captures the timer-hierarchy position for `timer` as if start(timer) were called
     *      right now, without touching the live timer stack. Used to attribute a duration that
     *      is measured out-of-band and only known later (e.g. a CUDA event's elapsed time) to
     *      the correct place in the timer hierarchy.
     * @param timer The timer region the later measurement belongs to
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    [[nodiscard]] static timer_stack capture_stack(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::capture_stack: timer_id was {}", timer_id);
        auto stack = current_timer_stack;
        stack.push_back(timer_id);
        return stack;
    }

    /**
     * @brief Adds a duration measured out-of-band (e.g. via a CUDA event) directly to the
     *      accumulator for `stack`. Does not interact with the live start()/stop() timer stack,
     *      so it is safe to call this well after the corresponding capture_stack() call, once the
     *      measurement (e.g. a CUDA event) has actually resolved.
     * @param stack A stack captured earlier via capture_stack()
     * @param delta_ns Elapsed time in nanoseconds to add
     */
    static void add_measurement(const timer_stack& stack, const std::uint64_t delta_ns) {
        auto& acc = accumulators_[stack];
        acc.count++;
        acc.time += delta_ns;
    }

    static void reset_all() {
        accumulators_.clear();
        current_timer_stack.clear();
    }

    /**
     * @brief Prints all timers with min, max, and sum across all MPI ranks to LogFiles::EventType::Timers.
     * The file is human readable.
     * Method Timers::collect_timer_data must be called before
     */
    static void print_human_readable();

    /**
     * @brief Prints all timers with min, max, and sum across all MPI ranks to LogFiles::EventType::TimersExtraP.
     * The file is a raw text file following the Extra-P format
     * Method Timers::collect_timer_data must be called before
     * @param total_steps The number of steps of the simulation
     * @param number_neurons_per_rank The number of neurons per rank
     */
    static void
    print_extrap(RelearnTypes::step_type total_steps, RelearnTypes::number_neurons_type number_neurons_per_rank);

    /**
     * @brief Prints all timers with min, max, and sum across all MPI ranks to LogFiles::EventType::TimersJson.
     * The file is in a json format preserving the hierarchy of the timers.
     * Method Timers::collect_timer_data must be called before
     */
    static void print_json();

    /**
     * @brief Collects the timer data from all ranks to prepare writing the global timers to file
     * Performs MPI communication.
     */
    static void collect_timer_data();

    static void print_human_readable(std::stringstream& console_output, const std::vector<std::uint64_t>& timers_min, const std::vector<std::uint64_t>& timers_max, const std::vector<std::uint64_t>& timers_sum, const std::vector<std::uint64_t>& timers_children);

    static void print_local_human_readable();

    /**
     * @brief Returns the current time as a string
     * @return The current time as a string
     */
    [[nodiscard]] static std::string wall_clock_time();

private:
    /**
     * @brief Casts the value of timer to an index for the vectors
     * @param timer The timer as an enum value
     * @result The timer as an index
     */
    [[nodiscard]] static index_type get_timer_index(const TimerRegion timer) noexcept {
        const auto timer_id = static_cast<index_type>(timer);
        return timer_id;
    }

    static inline std::vector<std::uint64_t> timers_min{};
    static inline std::vector<std::uint64_t> timers_max{};
    static inline std::vector<std::uint64_t> timers_sum{};
    static inline std::vector<std::uint64_t> timers_children{};

    static inline std::vector<std::uint64_t> timers_local{};
    static inline std::vector<std::uint64_t> timers_children_local{};

    static inline std::optional<TimerHierarchy> root_tree{};

    // NOLINTNEXTLINE

    static inline timer_stack current_timer_stack{};
    static inline std::unordered_map<timer_stack, profile_accumulator, TimerStackHash> accumulators_{};
};