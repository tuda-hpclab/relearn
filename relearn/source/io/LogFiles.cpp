/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "LogFiles.h"

#include "Config.h"

#include "io/Event.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPISynchronization.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

bool LogFiles::do_i_print(const EventType type, const mpiPP::MPIRank rank) {
    if (disable) {
        return false;
    }

    if (log_disable[type]) {
        return false;
    }

    if (rank == mpiPP::MPIRank::uninitialized_rank()) {
        return true;
    }

    return rank == mpiPP::MPIInfo::get_my_rank();
}

std::string LogFiles::get_my_rank_str() {
    return std::string(mpiPP::MPIInfo::get_my_rank_str());
}

void LogFiles::init() {
    if (disable) {
        return;
    }

    if (mpiPP::MPIRank::root_rank() == mpiPP::MPIInfo::get_my_rank()) {
        if (!std::filesystem::exists(output_path)) {
            std::filesystem::create_directory(output_path);
        }

        if (!std::filesystem::exists(output_path / "positions")) {
            std::filesystem::create_directory(output_path / "positions");
        }

        if (!std::filesystem::exists(output_path / "network")) {
            std::filesystem::create_directory(output_path / "network");
        }

        if (!std::filesystem::exists(output_path / "events")) {
            std::filesystem::create_directory(output_path / "events");
        }

        if (!std::filesystem::exists(output_path / "timers")) {
            std::filesystem::create_directory(output_path / "timers");
        }

        if (!std::filesystem::exists(output_path / "changes")) {
            std::filesystem::create_directory(output_path / "changes");
        }

        if (!std::filesystem::exists(output_path / "cout")) {
            std::filesystem::create_directory(output_path / "cout");
        }
    }

    // Wait until directory is created before any rank proceeds
    mpiPP::MPISynchronization::barrier();

    // Create log file for neurons overview on rank 0
    LogFiles::add_logfile(EventType::NeuronsOverview, "neurons_overview", mpiPP::MPIRank::root_rank());
    // LogFiles::add_logfile(EventType::NeuronsOverviewCSV, "neurons_overview_csv", MPIRank::root_rank(), ".csv");

    // Create log file for sums on rank 0
    LogFiles::add_logfile(EventType::Sums, "sums", mpiPP::MPIRank::root_rank());

    // Create log file for network on all ranks
    LogFiles::add_logfile(EventType::InNetwork, "in_network", mpiPP::MPIRank::uninitialized_rank(), ".txt", "network/");
    LogFiles::add_logfile(EventType::OutNetwork, "out_network", mpiPP::MPIRank::uninitialized_rank(), ".txt", "network/");

    // Create log file for positions on all ranks
    LogFiles::add_logfile(EventType::Positions, "positions", mpiPP::MPIRank::uninitialized_rank(), ".txt", "positions/");

    // Create log file for events on all ranks
    LogFiles::add_logfile(EventType::Events, "events", mpiPP::MPIRank::uninitialized_rank(), ".txt", "events/");

    // Create log file for group mappings on all ranks
    LogFiles::add_logfile(EventType::GroupMapping, "group_mapping", mpiPP::MPIRank::uninitialized_rank(), ".txt", "group_mapping/");

    // Create log file for neuron to group mappings on all ranks
    LogFiles::add_logfile(EventType::Groups, "groups", mpiPP::MPIRank::uninitialized_rank(), ".txt", "groups/");

    // Create log file for group name to file name mappings on all ranks
    LogFiles::add_logfile(EventType::GroupToFileMapping, "group_to_file_mapping", mpiPP::MPIRank::uninitialized_rank(), ".txt", "group_to_file_mapping/");

    // Create log file for std::cout
    LogFiles::add_logfile(EventType::Cout, "stdcout", mpiPP::MPIRank::uninitialized_rank(), ".txt", "cout/");

    // Create log file for the timers
    LogFiles::add_logfile(EventType::Timers, "timers", mpiPP::MPIRank::root_rank());
    LogFiles::add_logfile(EventType::TimersExtraP, "timers_extrap", mpiPP::MPIRank::root_rank());
    LogFiles::add_logfile(EventType::TimersJson, "timers_json", mpiPP::MPIRank::root_rank());

    // Create log file for the local timers
    LogFiles::add_logfile(EventType::TimersLocal, "timers_local", mpiPP::MPIRank::uninitialized_rank(), ".txt", "timers/");

    // Create log file for the synapse creation and deletion
    LogFiles::add_logfile(EventType::PlasticityUpdate, "plasticity_changes", mpiPP::MPIRank::root_rank());
    // LogFiles::add_logfile(EventType::PlasticityUpdateCSV, "plasticity_changes_csv", mpiPP::MPIRank::root_rank(), ".csv");

    // Create log file for the local synapse creation and deletion
    LogFiles::add_logfile(EventType::PlasticityUpdateLocal, "plasticity_changes_local", mpiPP::MPIRank::uninitialized_rank(), ".txt", "changes/");

    // LogFiles::add_logfile(EventType::NetworkInInhibitoryHistogramLocal, "network_in_inhibitory_histogram_local", mpiPP::MPIRank::uninitialized_rank());
    // LogFiles::add_logfile(EventType::NetworkInExcitatoryHistogramLocal, "network_in_excitatory_histogram_local", mpiPP::MPIRank::uninitialized_rank());
    // LogFiles::add_logfile(EventType::NetworkOutHistogramLocal, "network_out_histogram_local", mpiPP::MPIRank::uninitialized_rank());

    // Create log file for the essentials of the simulation
    LogFiles::add_logfile(EventType::Essentials, "essentials", mpiPP::MPIRank::root_rank());

    // Create log file for all calcium values
    LogFiles::add_logfile(EventType::CalciumValues, "calcium_values", mpiPP::MPIRank::uninitialized_rank(), ".txt", "", Config::calcium_max_file_size);
    LogFiles::add_logfile(EventType::ExtremeCalciumValues, "extreme_calcium_values", mpiPP::MPIRank::uninitialized_rank(), ".txt", "", Config::extreme_calcium_max_file_size);
    LogFiles::add_logfile(EventType::FireRates, "fire_rates", mpiPP::MPIRank::uninitialized_rank(), ".txt", "", Config::fire_rates_max_file_size);
    LogFiles::add_logfile(EventType::FireSteps, "fire_steps", mpiPP::MPIRank::uninitialized_rank(), ".txt", "", Config::fire_steps_max_file_size);

    // Create log file for all synaptic inputs
    LogFiles::add_logfile(EventType::SynapticInput, "synaptic_inputs", mpiPP::MPIRank::uninitialized_rank());

    initialized = true;
}

void LogFiles::add_event_to_trace(const Event& event) {
    const auto iterator = log_files.find(EventType::Events);
    if (iterator == log_files.end()) {
        return;
    }

    iterator->second->info(event);
}

std::string LogFiles::get_specific_file_prefix() {
    return std::string(mpiPP::MPIInfo::get_my_rank_str());
}

void LogFiles::save_and_open_new(EventType type, const std::string& new_file_name, const std::string& directory_prefix) {
    const auto iterator = log_files.find(type);
    if (iterator == log_files.end()) {
        return;
    }

    auto complete_path = (directory_prefix.empty() ? output_path : (output_path / directory_prefix)) / (general_prefix + get_specific_file_prefix() + "_" + new_file_name + ".txt");

    iterator->second->flush();

    spdlog::drop(iterator->second->name());

    auto new_logger = spdlog::basic_logger_mt(new_file_name, complete_path.string());
    new_logger->set_pattern("%v");
    iterator->second = std::move(new_logger);
}

void LogFiles::add_logfile(const EventType type, const std::string& file_name, const mpiPP::MPIRank rank, const std::string& file_ending, const std::string& directory_prefix, const std::uint32_t max_file_size) {
    if (do_i_print(type, rank)) {
        auto complete_path = (directory_prefix.empty() ? output_path : (output_path / directory_prefix)) / (general_prefix + get_specific_file_prefix() + "_" + file_name + file_ending);
        auto logger = std::shared_ptr<spdlog::logger>{};
        if (max_file_size != 0) {
            logger = spdlog::rotating_logger_mt(file_name, complete_path.string(), max_file_size, 2);
        } else {
            logger = spdlog::basic_logger_mt(file_name, complete_path.string());
        }
        logger->set_pattern("%v");
        log_files.emplace(type, std::move(logger));
    }
}
