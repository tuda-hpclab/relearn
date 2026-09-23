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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#ifdef _OPENMP
constexpr bool OPENMPAVAILABLE = true;
#else
constexpr bool OPENMPAVAILABLE = false;
#endif

// This exists for easier switching of compilation modes
// NOLINTNEXTLINE
#define RELEARN_MPI_FOUND MPI_FOUND

class Constants {
public:
    constexpr static unsigned int number_oct = 8;
    constexpr static std::size_t uninitialized = (1U << 31U);

    constexpr static std::size_t number_prealloc_space = 30;

    constexpr static RelearnTypes::level_type max_lvl_subdomains = 20;

    constexpr static RelearnTypes::acceptance_criterion_type eps = RelearnTypes::as<RelearnTypes::acceptance_criterion_type>(0.00001);

    constexpr static std::size_t print_string_width = 60;
    constexpr static std::size_t print_width = 22;
    constexpr static std::size_t print_precision = 8;

    constexpr static std::size_t mpi_alloc_mem = std::size_t{ 1024U } * 1024U;

    // Constants for Fast Gauss
    constexpr static std::size_t p = 4;
    constexpr static std::size_t p3 = p * p * p;
    constexpr static std::size_t max_neurons_in_target = 70; // cutoff for target box
    constexpr static std::size_t max_neurons_in_source = 70; // cutoff for source box

    constexpr static RelearnTypes::level_type unpacking = 0; // indicates how many levels a node is unpacked to give the synaptic elements more choice to connect
    // only used when FMM is selected. When unpacking == 0 normal FMM is used.

    constexpr static RelearnTypes::acceptance_criterion_type bh_default_theta{ RelearnTypes::as<RelearnTypes::acceptance_criterion_type>(0.3) };
    constexpr static RelearnTypes::acceptance_criterion_type bh_max_theta{ RelearnTypes::as<RelearnTypes::acceptance_criterion_type>(0.5) };

    constexpr static int number_rma_download_retries = 10;

    constexpr static std::size_t default_group_id = 0;
    constexpr static std::string_view default_group_name = "default_group";
};

class Config {
public:
    inline static bool do_debug_checks = false;

    // By default: Update synaptic elements every <synaptic_elements_update_step> ms
    inline static std::uint32_t synaptic_elements_update_step = 100; // NOLINT

    // By default: Update electrical activity every <electrical_activity_update_step> ms
    inline static std::uint32_t electrical_activity_update_step = 100; // NOLINT

    // End the connectivity updates at <last_plasticity_update> ms
    inline static std::uint32_t last_plasticity_update = std::numeric_limits<std::uint32_t>::max(); // NOLINT
    // By default: Update plasticity every <plasticity_update_step> ms
    inline static std::uint32_t plasticity_update_step = 100; // NOLINT

    // By default: Print details every <logfile_update_step> ms
    inline static std::uint32_t logfile_update_step = 100; // NOLINT

    inline static bool calculate_fire_history = false;

    // By default: Print to cout every <console_update_step> ms
    inline static std::uint32_t console_update_step = 100; // NOLINT

    inline static bool do_binary_search = true;

    // Capture individual neuron information every <monitor_step> ms
    inline static std::uint32_t monitor_step = 100; // NOLINT
    // By default: Capture individual neuron informations every <neuron_monitor_log_step> ms
    inline static std::uint32_t neuron_monitor_log_step = 100; // NOLINT
    // By default: The fire history captures the last <fire_history_reset_step> ms
    inline static std::uint32_t fire_history_reset_step = 10000; // NOLINT

    // Capture ensemble information every <monitor_group_step> ms
    inline static std::uint32_t monitor_group_step = 100; // NOLINT
    // By default: Capture ensemble informations every <group_monitor_log_step> ms
    inline static std::uint32_t group_monitor_log_step = 100; // NOLINT

    // By default: Capture the global statistics every <statistics_log_step> ms
    inline static std::uint32_t statistics_log_step = 100; // NOLINT

    // By default: Capture the neuron histogram every <histogram_log_step> ms
    inline static std::uint32_t histogram_log_step = 100; // NOLINT

    // By default: Capture the calcium values every <calcium_log_step> ms
    inline static std::uint32_t calcium_log_step = 1000000; // NOLINT

    // By default: Capture the fire rate every <fire_rate_log_step> ms
    inline static std::uint32_t fire_rate_log_step = fire_history_reset_step; // NOLINT

    // By default: Capture the network every <network_log_step> ms
    inline static std::uint32_t network_log_step = 10000; // NOLINT

    // By default: Capture the syanptic input every <synaptic_input_log_step> ms
    inline static std::uint32_t synaptic_input_log_step = 10000; // NOLINT

    // By default: Flush the group monitors every <flush_group_monitor_step> ms
    inline static std::uint32_t flush_group_monitor_step = 100000; // NOLINT

    // By default: Flush the neuron monitors every <flush_monitor_step> ms
    inline static std::uint32_t flush_monitor_step = 30000; // NOLINT

    // The maximum file size for the logging of the fire rates. Default: 0 is unlimited
    inline static std::uint32_t fire_rates_max_file_size = 0;

    // The maximum file size for the logging of the fire steps. Default: 0 is unlimited
    inline static std::uint32_t fire_steps_max_file_size = 0;

    // The maximum file size for the logging of the calcium values. Default: 0 is unlimited
    inline static std::uint32_t calcium_max_file_size = 0;

    // The maximum file size for the logging of the extreme calcium values. Default: 0 is unlimited
    inline static std::uint32_t extreme_calcium_max_file_size = 0;

    inline static std::uint64_t random_seed{};

    inline static bool cuda_aware_mpi_available{};
};
