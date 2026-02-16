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

#include "Types.h"

#include "util/RelearnException.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class Essentials;

/**
 * This type allows type-safe specification of a specific timer
 */
enum class TimerRegion : std::uint8_t {
    INITIALIZATION = 0,
    LOAD_SYNAPSES,
    INITIALIZE_NETWORK_GRAPH,
    INITIALIZE_SYNAPTIC_ELEMENTS,

    SIMULATION_LOOP,

    UPDATE_ELECTRICAL_ACTIVITY,
    EXCHANGE_FIRED_STATUS,
    PREPARE_SENDING_SPIKES,
    EXCHANGE_NEURON_IDS,

    CALC_ACTIVITY_INPUT,
    CALC_ACTIVITY_INPUT_COMBINE,
    CALC_ACTIVITY_INPUT_CONSTANT,
    CALC_ACTIVITY_INPUT_FAST_NORMAL,
    CALC_ACTIVITY_INPUT_FLEXIBLE,
    CALC_ACTIVITY_INPUT_NORMAL,
    CALC_ACTIVITY_INPUT_SCALE,
    CALC_ACTIVITY_INPUT_STIMULATION,
    CALC_ACTIVITY_INPUT_SYNAPSES,
    CALC_ACTIVITY_INPUT_SYNAPSES_LOCAL,
    CALC_ACTIVITY_INPUT_SYNAPSES_DISTANT,

    CALC_ACTIVITY,
    CALC_CALCIUM_EXTREME_VALUES,

    UPDATE_CALCIUM,
    UPDATE_TARGET_CALCIUM,

    UPDATE_SYNAPTIC_ELEMENTS_DELTA,

    UPDATE_CONNECTIVITY,
    UPDATE_FIRE_HISTORY,

    UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES,
    COMMIT_NUM_SYNAPTIC_ELEMENTS,
    FIND_SYNAPSES_TO_DELETE,
    DELETE_SYNAPSES_ALL_TO_ALL,
    DELETE_SYNAPSES_EXCHANGE_WHITELIST,
    PROCESS_DELETE_REQUESTS,

    UPDATE_LEAF_NODES,
    UPDATE_LOCAL_TREES,
    EXCHANGE_BRANCH_NODES,
    INSERT_BRANCH_NODES_INTO_GLOBAL_TREE,
    UPDATE_GLOBAL_TREE,

    CREATE_SYNAPSES,
    FIND_TARGET_NEURONS,
    CALC_TAYLOR_COEFFICIENTS,
    CALC_HERMITE_COEFFICIENTS,
    EXCHANGE_CREATION_REQUESTS,
    PROCESS_CREATION_REQUESTS,
    CREATE_CREATION_RESPONSES,
    PROCESS_CREATION_RESPONSES,

    ADD_SYNAPSES_TO_NETWORK_GRAPH,

    EMPTY_REMOTE_NODES_CACHE,

    CAPTURE_NEURON_MONITORS,
    CAPTURE_FIRE_STEPS,
    CAPTURE_GROUP_MONITORS,

    GROUP_MONITORS_PREPARE,
    GROUP_MONITORS_REQUEST,
    GROUP_MONITORS_EXCHANGE,
    GROUP_MONITORS_RECORD_DATA,
    GROUP_MONITORS_LOCAL_EDGES,
    GROUP_MONITORS_DISTANT_EDGES,
    GROUP_MONITORS_DELETIONS,
    GROUP_MONITORS_STATISTICS,
    GROUP_MONITORS_FINISH,
    PRINT_IO,

    ALL,
};

struct TimerHierarchy {
    /**
     * Struct that represent the hierarchy of the timers.
     * A root timer contains all other timers. Each timer can have children that further split its run time. The parent timer is at least the sum of all its child timers but can also be larger.
     */
    TimerRegion timer_region{};
    std::string timer_name;
    std::vector<TimerHierarchy> children;
};

// clang-format off
static const TimerHierarchy root_timer{
 .timer_region=TimerRegion::ALL, .timer_name="All", .children ={
{.timer_region=TimerRegion::INITIALIZATION, .timer_name="Initialization", .children ={
    {.timer_region=TimerRegion::LOAD_SYNAPSES, .timer_name="Load Synapses", .children={}},
    {.timer_region=TimerRegion::INITIALIZE_NETWORK_GRAPH, .timer_name="Initialize Network Graph", .children={}},
    {.timer_region=TimerRegion::INITIALIZE_SYNAPTIC_ELEMENTS, .timer_name="Initialize synaptic elements", .children={}},
}},
{.timer_region=TimerRegion::SIMULATION_LOOP, .timer_name="Simulation loop", .children ={
    {.timer_region=TimerRegion::UPDATE_ELECTRICAL_ACTIVITY, .timer_name="Update electrical activity", .children={
        {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT, .timer_name="Calculate activity input", .children={
            {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_COMBINE, .timer_name="Calculate activity input combination", .children={}},
            {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_FAST_NORMAL, .timer_name="Calculate activity input fast normal", .children={}},
            {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_NORMAL, .timer_name="Calculate activity input normal", .children={}},
            {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_CONSTANT, .timer_name="Calculate activity input constant", .children={}},
            {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_FLEXIBLE, .timer_name="Calculate activity input flexible", .children={}},
            {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_SCALE, .timer_name="Calculate activity input scale", .children={}},
            {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_STIMULATION, .timer_name="Calculate activity input stimulation", .children={}},
            {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_SYNAPSES, .timer_name="Calculate activity input synapses", .children={
             {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_SYNAPSES_DISTANT, .timer_name="Calculate activity input synapses (d)", .children={}},
             {.timer_region=TimerRegion::CALC_ACTIVITY_INPUT_SYNAPSES_LOCAL, .timer_name="Calculate activity input synapses (l)", .children={}},
             }},
        }},
        {.timer_region=TimerRegion::CALC_ACTIVITY, .timer_name="Calculate activity", .children={}},
        {.timer_region=TimerRegion::EXCHANGE_FIRED_STATUS, .timer_name="Exchange Fired Status", .children={
            {.timer_region=TimerRegion::PREPARE_SENDING_SPIKES, .timer_name="Prepare sending spikes", .children={}},
            {.timer_region=TimerRegion::EXCHANGE_NEURON_IDS, .timer_name="Exchange neuron ids", .children={}},
        }},
        {.timer_region=TimerRegion::UPDATE_CALCIUM, .timer_name="Calculate calcium", .children={}},
        {.timer_region=TimerRegion::UPDATE_TARGET_CALCIUM, .timer_name="Calculate target calcium", .children={}},
        {.timer_region=TimerRegion::CALC_CALCIUM_EXTREME_VALUES, .timer_name="Update Calcium extreme values", .children={}},
    }},
    {.timer_region=TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA, .timer_name="Update #synaptic elements delta", .children={}},
    {.timer_region=TimerRegion::UPDATE_CONNECTIVITY, .timer_name="Connectivity update", .children={
        {.timer_region=TimerRegion::UPDATE_FIRE_HISTORY, .timer_name="Update fire history", .children={}},
        {.timer_region=TimerRegion::UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES, .timer_name="Delete synapses", .children={
            {.timer_region=TimerRegion::COMMIT_NUM_SYNAPTIC_ELEMENTS, .timer_name="Commit #synaptic elements", .children={}},
            {.timer_region=TimerRegion::FIND_SYNAPSES_TO_DELETE, .timer_name="Find synapses to delete", .children={}},
            {.timer_region=TimerRegion::DELETE_SYNAPSES_EXCHANGE_WHITELIST, .timer_name="Exchange deletions (w/ all to all)", .children={}},
            {.timer_region=TimerRegion::DELETE_SYNAPSES_ALL_TO_ALL, .timer_name="Exchange deletions (w/ whitelist)", .children={}},
            {.timer_region=TimerRegion::PROCESS_DELETE_REQUESTS, .timer_name="Process deletion requests", .children={}},
        }},
    {.timer_region=TimerRegion::UPDATE_LEAF_NODES, .timer_name="Update leaf nodes", .children={}},
    {.timer_region=TimerRegion::UPDATE_LOCAL_TREES, .timer_name="Update local trees", .children={}},
    {.timer_region=TimerRegion::EXCHANGE_BRANCH_NODES, .timer_name="Exchange branch nodes (w/ All-gather)", .children={}},
    {.timer_region=TimerRegion::INSERT_BRANCH_NODES_INTO_GLOBAL_TREE, .timer_name="Insert branch nodes into global tree", .children={}},
    {.timer_region=TimerRegion::UPDATE_GLOBAL_TREE, .timer_name="Update global tree", .children={}},
    {.timer_region=TimerRegion::CREATE_SYNAPSES, .timer_name="Create synapses", .children={
        {.timer_region=TimerRegion::FIND_TARGET_NEURONS, .timer_name="Find target neurons (w/ RMA)", .children={
            {.timer_region=TimerRegion::CALC_TAYLOR_COEFFICIENTS, .timer_name="FMM: Calculate Taylor Coefficients", .children={}},
            {.timer_region=TimerRegion::CALC_HERMITE_COEFFICIENTS, .timer_name="FMM: Calculate Hermite Coefficients", .children={}}
        }},
        {.timer_region=TimerRegion::EXCHANGE_CREATION_REQUESTS, .timer_name="Create synapses Exchange Requests", .children={}},
        {.timer_region=TimerRegion::PROCESS_CREATION_REQUESTS, .timer_name="Create synapses Process Requests", .children={}},
        {.timer_region=TimerRegion::CREATE_CREATION_RESPONSES, .timer_name="Create synapses Exchange Responses", .children={}},
        {.timer_region=TimerRegion::PROCESS_CREATION_RESPONSES, .timer_name="Create synapses Process Responses", .children={}}
    }},
    {.timer_region=TimerRegion::ADD_SYNAPSES_TO_NETWORK_GRAPH, .timer_name="Add synapses in local network graphs", .children={}},
    {.timer_region=TimerRegion::EMPTY_REMOTE_NODES_CACHE, .timer_name="Empty remote nodes cache", .children={}}
    }},
    {.timer_region=TimerRegion::CAPTURE_NEURON_MONITORS, .timer_name="Capture neuron monitors", .children={}},
    {.timer_region=TimerRegion::CAPTURE_FIRE_STEPS, .timer_name="Capture fire steps", .children={}},
    {.timer_region=TimerRegion::CAPTURE_GROUP_MONITORS, .timer_name="Capture group monitors", .children={
        {.timer_region=TimerRegion::GROUP_MONITORS_PREPARE, .timer_name="Prepare", .children={}},
        {.timer_region=TimerRegion::GROUP_MONITORS_REQUEST, .timer_name="Request", .children={}},
        {.timer_region=TimerRegion::GROUP_MONITORS_EXCHANGE, .timer_name="Exchange", .children={}},
        {.timer_region=TimerRegion::GROUP_MONITORS_RECORD_DATA, .timer_name="Record", .children={
                {.timer_region=TimerRegion::GROUP_MONITORS_LOCAL_EDGES, .timer_name="Local edges", .children={}},
                {.timer_region=TimerRegion::GROUP_MONITORS_DISTANT_EDGES, .timer_name="Distant edges", .children={}},
                {.timer_region=TimerRegion::GROUP_MONITORS_DELETIONS, .timer_name="Deletions", .children={}},
                {.timer_region=TimerRegion::GROUP_MONITORS_STATISTICS, .timer_name="Statistics", .children={}}
        }},
        {.timer_region=TimerRegion::GROUP_MONITORS_FINISH, .timer_name="Finish", .children={}}
    }},
    {.timer_region=TimerRegion::PRINT_IO, .timer_name="Print IO", .children={}}
}}}
};
// clang-format on

/**
 * This number is used as a shortcut to count the number of values valid for TimerRegion
 */
constexpr std::size_t NUMBER_TIMERS = 62;

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
        time_start[timer_id] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stops the respective timer
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void stop(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::stop: timer_id was: {}", timer_id);
        time_stop[timer_id] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Stops the respective timer and adds the elapsed time
     * @param timer The timer to stops
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void stop_and_add(const TimerRegion timer) {
        stop(timer);
        add_start_stop_diff_to_elapsed(timer);
    }

    /**
     * @brief Adds the difference between the current start and stop time points to the elapsed time
     * @param timer The timer for which to add the difference
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void add_start_stop_diff_to_elapsed(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::add_start_stop_diff_to_elapsed: timer_id was: {}",
                                timer_id);
        time_elapsed[timer_id] += (time_stop[timer_id] - time_start[timer_id]);
    }

    /**
     * @brief Resets the elapsed time for the timer
     * @param timer The timer for which to reset the elapsed time
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     */
    static void reset_elapsed(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::reset_elapsed: timer_id was: {}", timer_id);
        time_elapsed[timer_id] = std::chrono::nanoseconds(0);
    }

    /**
     * @brief Returns the elapsed time for the respective timer
     * @param timer The timer for which to return the elapsed time
     * @exception Throws a RelearnException if the timer casts to an index that is >= NUMBER_TIMERS
     * @return The elapsed time
     */
    [[nodiscard]] static std::chrono::nanoseconds get_elapsed(const TimerRegion timer) {
        const auto timer_id = get_timer_index(timer);
        RelearnException::check(timer_id < NUMBER_TIMERS, "Timers::get_elapsed: timer_id was: {}", timer_id);
        return time_elapsed[timer_id];
    }

    /**
     * @brief Prints all timers with min, max, and sum across all MPI ranks to LogFiles::EventType::Timers.
     * The file is human readable.
     * Method Timers::collect_timer_data must be called before
     * @param essentials The essentials
     */
    static void print_human_readable(const std::unique_ptr<Essentials>& essentials);

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

    // NOLINTNEXTLINE
    static inline std::vector<time_point> time_start{ NUMBER_TIMERS };
    // NOLINTNEXTLINE
    static inline std::vector<time_point> time_stop{ NUMBER_TIMERS };

    static inline std::array<double, NUMBER_TIMERS> timers_min{};
    static inline std::array<double, NUMBER_TIMERS> timers_max{};
    static inline std::array<double, NUMBER_TIMERS> timers_sum{};

    // NOLINTNEXTLINE
    static inline std::vector<std::chrono::nanoseconds> time_elapsed{ NUMBER_TIMERS };
};