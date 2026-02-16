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

#include "util/RelearnException.h"

#include "mpi-wrapper/MPIRank.h"

#include <fmt/core.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>

class Event;

/**
 * This class provides a static interface that allows for writing log messages to predefined files.
 * The path can be set and the filename's prefix can be chosen freely.
 * Some files are only created for the MPI rank 0, some for all.
 */
class LogFiles {
public:
    /**
     * This enum classifies the different type of files that can be written to.
     * It also includes Cout, however, using this value does not automatically print to std::cout
     */
    enum class EventType : char {
        PlasticityUpdate,
        PlasticityUpdateCSV,
        PlasticityUpdateLocal,
        NeuronsOverview,
        NeuronsOverviewCSV,
        Sums,
        InNetwork,
        OutNetwork,
        Positions,
        Cout,
        Timers,
        TimersExtraP,
        TimersJson,
        TimersLocal,
        NetworkInInhibitoryHistogramLocal,
        NetworkInExcitatoryHistogramLocal,
        NetworkOutHistogramLocal,
        Essentials,
        CalciumValues,
        FireSteps,
        FireRates,
        ExtremeCalciumValues,
        SynapticInput,
        GroupMapping,
        Groups,
        GroupToFileMapping,
        Events,
    };

    /**
     * @brief Sets the folder path in which the log files will be generated. Automatically appends '/' if necessary.
     *      Default is: "../output/"
     * @param path_to_containing_folder The path to the folder in which the files should be generated
     * @exception Throws a RelearnException if called after init()
     */
    static void set_output_path(const std::filesystem::path& path_to_containing_folder) {
        RelearnException::check(!initialized, "LogFiles::set_output_path: LogFiles are already initialized");
        output_path = path_to_containing_folder;
    }

    /**
     * @brief Returns the currently used output path
     * @return The output path
     */
    static std::filesystem::path get_output_path() noexcept {
        return output_path;
    }

    /**
     * @brief Sets the general prefix for every log file.
     *      Default is: "rank_"
     * @param prefix The prefix for every file
     * @exception Throws a RelearnException if called after init()
     */
    static void set_general_prefix(const std::string& prefix) {
        RelearnException::check(!initialized, "LogFiles::set_general_prefix: LogFiles are already initialized");
        general_prefix = prefix;
    }

    /**
     * @brief Initializes all log files.
     *      Call this method after setting the output path and the general prefix when they should be user defined
     * @exception Throws a RelearnException if creating the files fails
     */
    static void init();

    /**
     * @brief Saves the specified file, closes it, and opens a new file as sink with the specified file name.
     *      Does nothing if the file is not present
     * @param type The event type whose file should be replaced
     * @param new_file_name The new file name
     * @param directory_prefix The prefix for the directory
     */
    static void save_and_open_new(EventType type, const std::string& new_file_name, const std::string& directory_prefix = "");

    /**
     * @brief Clears all log files to create new files in future runs. Also clears the initialized status.
     */
    static void clear_log_files() {
        log_files.clear();
        log_disable.clear();
        initialized = false;
    }

    /**
     * @brief Sets the status of the event type, i.e., if the log for that type is disabled
     * @param type The event type
     * @param disabled True iff the log shall be disabled
     * @exception Throws a RelearnException if called after init()
     */
    static void set_log_status(const EventType type, const bool disabled) {
        RelearnException::check(!initialized, "LogFiles::set_log_status: LogFiles are already initialized");

        log_disable[type] = disabled;
    }

    /**
     * @brief Returns the current log status for the event type
     * @param type The event type
     * @return True iff the log is disabled
     */
    static bool get_log_status(const EventType type) noexcept {
        return log_disable[type];
    }

    /**
     * @brief Flushes the associated file if it exists
     * @param type The event type which should be flushed
     */
    static void flush_file(const EventType type) {
        if (auto iterator = log_files.find(type); iterator != log_files.end()) {
            iterator->second->flush();
        }
    }

    /**
     * @brief Write the message into the file which is associated with the type.
     *      Optionally prints the message also to std::cout. The message can have place-holders of the form "{}", which are filled with additional arguments in the order of occurrence.
     *      If the log is disabled, nothing is written to the file (but to std::cout if specified so).
     * @param type The event type to which the message belongs
     * @param also_to_cout A flag that indicates if the formatted string should also be print to std::cout
     * @param format Some type of string, optionally with place-holders of the form {}
     * @param args Variably many additional arguments that are inserted for the place-holders
     */
    template <typename... Args>
    static void write_to_file(const EventType type, const bool also_to_cout, const std::string_view format, Args&&... args) {
        auto message = fmt::format(fmt::runtime(format), std::forward<Args>(args)...);

        if (also_to_cout) {
            spdlog::info(message);
        }

        const auto disabled = log_disable[type];
        if (disabled) {
            return;
        }

        // Not all ranks have all log files
        if (auto iterator = log_files.find(type); iterator != log_files.end()) {
            iterator->second->info(message);
        }
    }

    /**
     * @brief Write the message into the file which is associated with the type.
     *      Optionally prints the message also to std::cout. The message will be forwarded as it is without formatting.
     *      If the log is disabled, nothing is written to the file (but to std::cout if specified so).
     * @param type The event type to which the message belongs
     * @param also_to_cout A flag that indicates if the formatted string should also be print to std::cout
     * @param message Some type of string, optionally with place-holders of the form {}
     */
    static void write_raw_string_to_file(const EventType type, const bool also_to_cout, const std::string_view message) {
        if (also_to_cout) {
            spdlog::info(message);
        }

        const auto disabled = log_disable[type];
        if (disabled) {
            return;
        }

        // Not all ranks have all log files
        if (auto iterator = log_files.find(type); iterator != log_files.end()) {
            iterator->second->info(message);
        }
    }

    /**
     * @brief Prints a message to std::cout (and the associated file), if rank matches the current MPI rank.
     *      The message can have place-holders of the form "{}", which are filled with additional arguments in the order of occurrence.
     * @param rank The MPI rank that should print the message. -1 for all MPI ranks
     * @param format Some type of string, optionally with place-holders of the form {}
     * @param args Variably many additional arguments that are inserted for the place-holders
     */
    template <typename... Args>
    static void print_message_rank(const mpiPP::MPIRank rank, const std::string format, Args&&... args) { // NOLINT(readability-avoid-const-params-in-decls)
        if (do_i_print(LogFiles::EventType::Cout, rank)) {
            write_to_file(LogFiles::EventType::Cout, true, "[INFO:Rank {}] {}", get_my_rank_str(), fmt::format(fmt::runtime(format), std::forward<Args>(args)...));
        }
    }

    /**
     * @brief Adds an event to the trace
     * @param event The event to print
     */
    static void add_event_to_trace(const Event& event);

private:
    using Logger = std::shared_ptr<spdlog::logger>;

    static inline std::map<EventType, Logger> log_files{};
    static inline std::map<EventType, bool> log_disable{};
    static inline bool initialized{};

    // NOLINTNEXTLINE
    static inline std::filesystem::path output_path{ "../output/" };
    // NOLINTNEXTLINE
    static inline std::string general_prefix{ "rank_" };

    static std::string get_specific_file_prefix();

    /**
     * Creates a log file for the given event type. Call it only once for each event type
     * @param type The event type of the log file.
     * @param file_name File name
     * @param rank The current mpi rank
     * @param file_ending File ending
     * @param directory_prefix Directory of the log file. If "" the log file will be stored in the output directory root
     * @param max_file_size The maximum file size of the log file. 0 is interpreted as limitless. Otherwise, older information will be discarded when we read the maximum file limit
     */
    static void add_logfile(EventType type, const std::string& file_name, mpiPP::MPIRank rank, const std::string& file_ending = ".txt", const std::string& directory_prefix = "", std::uint32_t max_file_size = 0);

    [[nodiscard]] static bool do_i_print(EventType type, mpiPP::MPIRank rank);

    [[nodiscard]] static std::string get_my_rank_str();

public:
    static inline bool disable{ false }; // This is public for test purposes
};
