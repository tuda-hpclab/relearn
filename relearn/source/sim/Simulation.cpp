/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2021-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Simulation.h"

#include "Config.h"

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/BarnesHutInternal/BarnesHut.h"
#include "algorithm/BarnesHutInternal/BarnesHutInverted.h"
#include "algorithm/BarnesHutInternal/BarnesHutLocationAware.h"
#include "algorithm/BarnesHutInternal/BarnesHutLocationAwareModified.h"
#include "algorithm/BarnesHutInternal/BarnesHutRestricted.h"
#include "algorithm/CombinedAlgorithmsInternal/CombinedAlgorithms.h"
#include "algorithm/FMMInternal/FastMultipoleMethod.h"
#include "algorithm/NaiveInternal/Naive.h"
#include "cuda/CudaConfig.h"
#include "cuda/algorithm/BarnesHutInternalCUDA/BarnesHutCUDA.h"
#include "cuda/algorithm/NaiveInternalCUDA/NaiveCUDA.h"
#include "cuda/util/Util.h"
#include "io/LogFiles.h"
#include "neurons/NetworkGraph.h"
#include "neurons/Neurons.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/GlobalGroupMapper.h"
#include "neurons/helper/GroupMonitor.h"
#include "neurons/helper/NeuronMonitor.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/models/NeuronModel.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/SynapseLoader.h"
#include "structure/Partition.h"
#include "types/AlgorithmTypes.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/Random.h"
#include "util/RandomHolderKey.h"
#include "util/RelearnException.h"
#include "util/Timers.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#pragma GCC diagnostic pop

#include <cpp-utility/MemoryFootprint.hpp>
#include <cpp-utility/ranges/Functional.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/patterns/MPIAdvancedCommunicationPatterns.h>
#include <mpi-wrapper/reductions/MPIComponentwiseReductions.h>
#include <mpi-wrapper/reductions/MPIReductions.h>

#include <range/v3/action/transform.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/map.hpp>
#include <range/v3/view/repeat_n.hpp>

#include <sys/resource.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

Simulation::Simulation(std::unique_ptr<Essentials> _essentials, std::shared_ptr<Partition> _partition)
    : essentials(std::move(_essentials))
    , footprint(std::make_unique<utility::MemoryFootprint>(100))
    , usage_footprint(std::make_unique<utility::MemoryFootprint>(100))
    , partition(std::move(_partition)) {

    neuron_monitor = std::make_unique<NeuronMonitor>();
    neuron_monitor->set_output_path(LogFiles::get_output_path(), true);
    group_monitors = std::make_shared<std::unordered_map<RelearnTypes::group_id, GroupMonitor>>();
}

void Simulation::register_neuron_monitor(const NeuronID neuron_id) {
    neuron_monitor->register_neuron(neuron_id.get_neuron_id());
}

void Simulation::set_acceptance_criterion_for_barnes_hut(const acceptance_criterion_type value) {
    // Needed to avoid creating autapses
    RelearnException::check(value <= Constants::bh_max_theta,
                            "Simulation::set_acceptance_criterion_for_barnes_hut: Acceptance criterion must be smaller or equal to {} but was {}",
                            Constants::bh_max_theta, value);
    RelearnException::check(value > acceptance_criterion_type{ 0 },
                            "Simulation::set_acceptance_criterion_for_barnes_hut: Acceptance criterion must larger than 0.0, but it was {}",
                            value);

    accept_criterion = value;
}

void Simulation::set_algorithm_vector_for_combined_algorithms(RelearnTypes::AlgorithmConfigs&& algorithm_configs_to_use) { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - moved into algorithms below
    RelearnException::check(!algorithm_configs_to_use.empty(), "Simulation::set_algorithm_vector_for_combined_algorithms: Vector of algorithm configs should not be empty, but was!");
    algorithms = std::move(algorithm_configs_to_use);
}

void Simulation::set_indices_and_neurons_for_combined_algorithms(const RelearnTypes::AlgorithmIndexWithNeuronsType& inds_and_neurons) {
    RelearnException::check(!inds_and_neurons.empty(), "Simulation::set_indices_and_neurons_for_combined_algorithms: Indices and neurons vector should not be empty, but was!");
    indices_and_neurons = inds_and_neurons;
}

void Simulation::set_neuron_model(std::unique_ptr<NeuronModel>&& _neuron_model) noexcept { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - moved into neuron_models below
    neuron_models = std::move(_neuron_model);
}

void Simulation::set_calcium_calculator(std::unique_ptr<CalciumCalculator>&& calculator) noexcept { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - moved into calcium_calculator below
    calcium_calculator = std::move(calculator);
}

void Simulation::set_synaptic_elements(std::shared_ptr<SynapticElements>&& _synaptic_elements) noexcept { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - moved into synaptic_elements below
    synaptic_elements = std::move(_synaptic_elements);
}

void Simulation::set_synapse_deletion_finder(std::unique_ptr<SynapseDeletionFinder>&& sdf) noexcept { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - moved into synapse_deletion_finder below
    synapse_deletion_finder = std::move(sdf);
}

namespace {
constexpr auto sort_ids = [](auto entry) {
    ranges::sort(entry.second);
    return entry;
};
} // namespace

void Simulation::set_enable_interrupts(std::vector<std::pair<step_type, std::vector<NeuronID>>> interrupts) {
    enable_interrupts = std::move(interrupts) | ranges::actions::transform(sort_ids);
}

void Simulation::set_disable_interrupts(std::vector<std::pair<step_type, std::vector<NeuronID>>> interrupts) {
    disable_interrupts = std::move(interrupts) | ranges::actions::transform(sort_ids);
}

void Simulation::set_creation_interrupts(std::vector<std::pair<step_type, number_neurons_type>> interrupts) noexcept {
    creation_interrupts = std::move(interrupts);
}

void Simulation::set_algorithm(const AlgorithmEnum new_algorithm_enum) noexcept {
    algorithm_enum = new_algorithm_enum;
}

void Simulation::set_probability_kernel(std::unique_ptr<KernelBase>&& kernel) noexcept { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - moved into probability_kernel below
    probability_kernel = std::move(kernel);
}

void Simulation::set_percentage_initial_fired_neurons(const RelearnTypes::percentage_type percentage) {
    RelearnException::check(percentage >= RelearnTypes::percentage_type{ 0 },
                            "Simulation::set_percentage_initial_fired_neurons: percentage is too low: {}", percentage);
    RelearnException::check(percentage <= RelearnTypes::percentage_type{ 1 },
                            "Simulation::set_percentage_initial_fired_neurons: percentage is too high: {}", percentage);
    percentage_initially_fired = percentage;
}

void Simulation::set_subdomain_assignment(std::unique_ptr<NeuronToSubdomainAssignment>&& subdomain_assignment) noexcept { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved) - moved into neuron_to_subdomain_assignment below
    neuron_to_subdomain_assignment = std::move(subdomain_assignment);
}

void Simulation::initialize() {
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();
    start_time = std::chrono::system_clock::now();

    RelearnException::check(neuron_models != nullptr, "Simulation::initialize: neuron_models is nullptr");
    RelearnException::check(calcium_calculator != nullptr, "Simulation::initialize: calcium_calculator is nullptr");
    RelearnException::check(synaptic_elements != nullptr, "Simulation::initialize: synaptic_elements is nullptr");
    RelearnException::check(neuron_to_subdomain_assignment != nullptr, "Simulation::initialize: neuron_to_subdomain_assignment is nullptr");

    neuron_to_subdomain_assignment->initialize();

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Neurons loaded");

    const auto number_total_neurons = neuron_to_subdomain_assignment->get_total_number_placed_neurons();

    // GPU edge storage packs neuron IDs into 3 bytes (SmallNeuronIdType) by default to save GPU
    // memory; that only addresses up to max_small_neuron_id (2^24 - 1) distinct neuron IDs. Decide
    // once, here, whether this run needs the 4-byte fallback -- before any GPUEdgesBase is
    // constructed -- based on the *global* neuron count (other-neuron IDs can reference neurons on
    // any MPI rank, not just the local one).
    CudaConfig::use_wide_neuron_ids = partition->get_number_local_neurons() > max_small_neuron_id;
    if (CudaConfig::use_wide_neuron_ids) {
        LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(),
                                     "Neuron count {} exceeds the 3-byte GPU neuron-ID limit ({}); using 4-byte neuron IDs for GPU edge storage",
                                     partition->get_number_local_neurons(), max_small_neuron_id);
    }

    auto number_local_neurons_ntsa = neuron_to_subdomain_assignment->get_number_neurons_in_subdomains();
    auto neuron_positions = neuron_to_subdomain_assignment->get_neuron_positions_in_subdomains();
    auto local_group_translator = neuron_to_subdomain_assignment->get_local_group_translator();
    auto signal_types = neuron_to_subdomain_assignment->get_neuron_types_in_subdomains();

    RelearnException::check(number_local_neurons_ntsa > 0, "I have 0 neurons at rank {}", my_rank.get_rank());
    RelearnException::check(neuron_positions.size() == number_local_neurons_ntsa,
                            "Simulation::initialize: neuron_positions had the wrong size");
    RelearnException::check(local_group_translator->get_number_neurons_in_total() == number_local_neurons_ntsa,
                            "Simulation::initialize: neuron_id_vs_group_id had the wrong size {} != {}",
                            local_group_translator->get_number_neurons_in_total(), number_local_neurons_ntsa);
    RelearnException::check(signal_types.size() == number_local_neurons_ntsa,
                            "Simulation::initialize: signal_types had the wrong size");

    partition->set_total_number_neurons(number_total_neurons);

    const auto number_local_neurons = partition->get_number_local_neurons();
    const auto& simulation_box = partition->get_simulation_box_size();

    partition->print_my_subdomains_info_rank();

    RelearnException::check(number_local_neurons_ntsa == number_local_neurons,
                            "Simulation::initialize: The partition and the NTSA had a disagreement about the number of local neurons");

    Timers::start(TimerRegion::LOAD_SYNAPSES);
    auto synapse_loader = neuron_to_subdomain_assignment->get_synapse_loader();
    const auto& [synapses_static, synapses_plastic] = synapse_loader->load_synapses(essentials);
    const auto& [local_synapses_static, distant_in_synapses_static, distant_out_synapses_static] = synapses_static;
    const auto& [local_synapses_plastic, distant_in_synapses_plastic, distant_out_synapses_plastic] = synapses_plastic;
    Timers::stop_and_add(TimerRegion::LOAD_SYNAPSES);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Synapses loaded");

    auto network_graph = std::make_shared<NetworkGraph>(my_rank);
    network_graph->init(number_local_neurons, network_gpu_type);

    Timers::start(TimerRegion::INITIALIZE_NETWORK_GRAPH);
    network_graph->add_edges(local_synapses_plastic, distant_in_synapses_plastic, distant_out_synapses_plastic);
    network_graph->add_edges(local_synapses_static, distant_in_synapses_static, distant_out_synapses_static);
#ifdef RELEARN_CUDA_ENABLED
    network_graph->sync_with_gpu();
#endif

    Timers::stop_and_add(TimerRegion::INITIALIZE_NETWORK_GRAPH);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Network graph created");

    synapse_deletion_finder->init(number_local_neurons);
    synapse_deletion_finder->set_synaptic_elements(synaptic_elements);

    neuron_models->set_network_graph(network_graph);

    neurons = std::make_shared<Neurons>(partition, std::move(neuron_models), std::move(calcium_calculator), std::move(network_graph),
                                        synaptic_elements, std::move(synapse_deletion_finder));

    const auto& space_filling_curve = partition->get_space_filling_curve();

    switch (algorithm_enum) {
    case AlgorithmEnum::BarnesHut: // NOLINT(bugprone-branch-clone) - each case constructs a genuinely different concrete Algorithm subclass; checker false positive
        neurons->set_algorithm(std::make_shared<BarnesHut>(simulation_box, space_filling_curve, accept_criterion));
        break;
    case AlgorithmEnum::BarnesHutInverted:
        neurons->set_algorithm(std::make_shared<BarnesHutInverted>(simulation_box, space_filling_curve, accept_criterion));
        break;
    case AlgorithmEnum::BarnesHutLocationAware:
        neurons->set_algorithm(std::make_shared<BarnesHutLocationAware>(simulation_box, space_filling_curve, accept_criterion));
        break;
    case AlgorithmEnum::BarnesHutLocationAwareModified:
        neurons->set_algorithm(std::make_shared<BarnesHutLocationAwareModified>(simulation_box, space_filling_curve, accept_criterion));
        break;
    case AlgorithmEnum::BarnesHutRestricted:
        neurons->set_algorithm(std::make_shared<BarnesHutRestricted>(simulation_box, space_filling_curve, accept_criterion));
        break;
    case AlgorithmEnum::FastMultipoleMethod:
        neurons->set_algorithm(std::make_shared<FastMultipoleMethod>(simulation_box, space_filling_curve));
        break;
    case AlgorithmEnum::Naive:
        neurons->set_algorithm(std::make_shared<Naive>(simulation_box, space_filling_curve));
        break;
    case AlgorithmEnum::CombinedAlgorithms:
        neurons->set_algorithm(std::make_shared<CombinedAlgorithms>(simulation_box, space_filling_curve, std::move(algorithms), indices_and_neurons, accept_criterion));
        break;
    case AlgorithmEnum::NaiveCuda:
        neurons->set_algorithm(std::make_shared<NaiveCUDA>(simulation_box, space_filling_curve));
        break;
    case AlgorithmEnum::BarnesHutCuda:
        neurons->set_algorithm(std::make_shared<BarnesHutCUDA>(simulation_box, space_filling_curve, accept_criterion));
        break;
    default:
        RelearnException::fail("Simulation::initialize: AlgorithmEnum {} not yet implemented!", algorithm_enum);
    }

    neurons->set_local_group_translator(local_group_translator);
    neurons->set_probability_kernel(std::move(probability_kernel));
    neurons->init(number_local_neurons, std::move(neuron_positions));
    neurons->register_neuron_monitor(*neuron_monitor);
    neurons->set_signal_types(std::move(signal_types));
    neurons->set_static_neurons(static_neurons);

    Timers::start(TimerRegion::INITIALIZE_SYNAPTIC_ELEMENTS);
    neurons->init_synaptic_elements(local_synapses_plastic, distant_in_synapses_plastic, distant_out_synapses_plastic);
    Timers::stop_and_add(TimerRegion::INITIALIZE_SYNAPTIC_ELEMENTS);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Synaptic elements initialized");

    global_group_mapper = std::make_shared<GlobalGroupMapper>(local_group_translator, mpiPP::MPIInfo::get_number_ranks(),
                                                              my_rank);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Neurons created");

    if (group_monitor_enabled) {
        for (auto group_id = std::size_t{ 0 }; group_id < local_group_translator->get_number_of_groups(); group_id++) {
            const auto& group_name = local_group_translator->get_group_name_for_group_id(group_id);
            const std::filesystem::path dir = LogFiles::get_output_path() / "group_monitors";
            if (!std::filesystem::exists(dir)) {
                std::filesystem::create_directories(dir);
            }
            auto path = dir / (std::string(mpiPP::MPIInfo::get_my_rank_str()) + "_group_" + std::to_string(group_id) + ".csv");
            group_monitors->insert(
                std::make_pair(group_id, GroupMonitor(neurons, global_group_mapper, group_id, group_name, my_rank.get_rank(), path, group_monitor_connectivity)));
        }

        // Update group monitor
        Timers::start(TimerRegion::CAPTURE_GROUP_MONITORS);

        Timers::start(TimerRegion::GROUP_MONITORS_PREPARE);
        for (auto& [_, group_monitor] : *group_monitors) {
            group_monitor.prepare_recording();
        }

        Timers::stop_and_add(TimerRegion::GROUP_MONITORS_PREPARE);
        Timers::start(TimerRegion::GROUP_MONITORS_REQUEST);

        for (auto& [_, group_monitor] : *group_monitors) {
            group_monitor.request_data();
        }

        Timers::stop_and_add(TimerRegion::GROUP_MONITORS_REQUEST);
        Timers::start(TimerRegion::GROUP_MONITORS_EXCHANGE);

        global_group_mapper->exchange_requests();

        Timers::stop_and_add(TimerRegion::GROUP_MONITORS_EXCHANGE);
        Timers::start(TimerRegion::GROUP_MONITORS_RECORD_DATA);

        for (auto& [_, group_monitor] : *group_monitors) {
            group_monitor.monitor_connectivity();
        }

        Timers::stop_and_add(TimerRegion::GROUP_MONITORS_RECORD_DATA);
        Timers::start(TimerRegion::GROUP_MONITORS_FINISH);

        for (auto& [_, group_monitor] : *group_monitors) {
            group_monitor.finish_recording();
        }
        Timers::stop_and_add(TimerRegion::GROUP_MONITORS_FINISH);
        Timers::stop_and_add(TimerRegion::CAPTURE_GROUP_MONITORS);
    }

    if (percentage_initially_fired > percentage_type{ 0 }) {
        const auto fired_neurons = static_cast<std::ptrdiff_t>(static_cast<percentage_type>(number_local_neurons) * percentage_initially_fired);
        const auto inactive_neurons = static_cast<std::ptrdiff_t>(number_local_neurons) - fired_neurons;

        auto initial_fired = ranges::views::concat(
                                 ranges::views::repeat_n(FiredStatus::Fired, fired_neurons),
                                 ranges::views::repeat_n(FiredStatus::Inactive, inactive_neurons))
                             | ranges::to_vector
                             | RandomHolder::shuffleAction(RandomHolderKey::BackgroundActivity);

        neurons->set_fired(std::move(initial_fired));
    }

    // RelearnException::fail("Fix this");
    // MPIWrapper::create_rma_window<boost::dynamic_bitset<>>(MPIWindow::FireHistory, number_local_neurons, MPIWrapper::get_number_ranks());

    neurons->debug_check_counts();
    neurons->print_neurons_overview_to_log_file_on_rank_0(0);
    neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(0, 0, 0, 0);
    neurons->print_group_mapping_to_log_file();
    neurons->print_groups_to_log_file();
    neurons->print_groups_to_file_name_to_log_file();
}

void Simulation::simulate(const step_type number_steps) {
    RelearnException::check(number_steps > 0, "Simulation::simulate: number_steps must be greater than 0");
    const auto my_rank = mpiPP::MPIInfo::get_my_rank();

    const auto previous_synapse_creations = total_synapse_creations;
    const auto previous_synapse_deletions = total_synapse_deletions;

    /**
     * Simulation loop
     */
    Timers::start(TimerRegion::SIMULATION_LOOP);
    const auto final_step_count = step + number_steps;
    for (; step <= final_step_count; ++step) { // NOLINT(altera-id-dependent-backward-branch)
        for (const auto& [disable_step, disable_ids] : disable_interrupts | ranges::views::filter(utility::equal_to(step), &std::pair<step_type, std::vector<NeuronID>>::first)) {
            LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Disabling {} neurons in step {}",
                                    disable_ids.size(), disable_step);

            const auto& [num_deleted_synapses, synapse_deletion_requests_outgoing] = neurons->disable_neurons(step,
                                                                                                              disable_ids,
                                                                                                              mpiPP::MPIInfo::get_number_ranks());
            total_synapse_deletions += static_cast<std::int64_t>(num_deleted_synapses);

            const auto& synapse_deletion_requests_ingoing = mpiPP::MPIAdvancedCommunicationPatterns::exchange_requests(
                synapse_deletion_requests_outgoing);
            total_synapse_deletions += static_cast<std::int64_t>(neurons->delete_disabled_distant_synapses(
                synapse_deletion_requests_ingoing, my_rank));
        }

        for (const auto& [enable_step, enable_ids] : enable_interrupts | ranges::views::filter(utility::equal_to(step), &std::pair<step_type, std::vector<NeuronID>>::first)) {
            LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Enabling {} neurons in step {}",
                                    enable_ids.size(), enable_step);
            neurons->enable_neurons(enable_ids);
        }

        for (const auto& [creation_step, creation_count] : creation_interrupts | ranges::views::filter(utility::equal_to(step), &std::pair<step_type, number_neurons_type>::first)) {
            LogFiles::write_to_file(LogFiles::EventType::Cout, true, "Creating {} neurons in step {}", creation_count,
                                    creation_step);
            neurons->create_neurons(creation_count);
        }

        if (interval_neuron_monitor.hits_step(step)) {
            Timers::start(TimerRegion::CAPTURE_NEURON_MONITORS);
            neuron_monitor->record_data(step);
            Timers::stop_and_add(TimerRegion::CAPTURE_NEURON_MONITORS);
        }

        if (interval_fire_rate_log.hits_step(step)) {
            Timers::start(TimerRegion::CAPTURE_FIRE_STEPS);
            neurons->print_fire_rate_to_file(step);
            // neurons->print_fire_steps_to_file(step, interval_fire_rate_log.frequency);
            const auto& fired_recorder = neurons->get_neuron_model()->get_fired_status_recorder();
            fired_recorder->reset(FiredStatusRecorder::FireRecorderPeriod::NeuronMonitor);
            Timers::stop_and_add(TimerRegion::CAPTURE_FIRE_STEPS);
        }

        if (interval_update_electrical_activity.hits_step(step)) {
            Timers::start(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
            neurons->update_electrical_activity(step);
            Timers::stop_and_add(TimerRegion::UPDATE_ELECTRICAL_ACTIVITY);
        }

        if (interval_update_synaptic_elements.hits_step(step)) {
            Timers::start(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
            neurons->update_number_synaptic_elements_delta(step);
            Timers::stop_and_add(TimerRegion::UPDATE_SYNAPTIC_ELEMENTS_DELTA);
        }

        if (interval_update_plasticity.hits_step(step)) {
            Timers::start(TimerRegion::UPDATE_CONNECTIVITY);

            const auto& [num_axons_deleted, num_dendrites_deleted, num_synapses_created] = neurons->update_connectivity(
                step);

            // Get total number of synapses deleted and created
            const auto local_counts = std::array<std::int64_t, 3>{ static_cast<std::int64_t>(num_axons_deleted),
                                                                   static_cast<std::int64_t>(num_dendrites_deleted),
                                                                   static_cast<std::int64_t>(num_synapses_created) };
            const std::array<std::int64_t, 3> global_counts = mpiPP::MPIReductions::reduce_componentwise_sum(
                local_counts);

            const auto local_deletions = local_counts[0] + local_counts[1];
            const auto local_creations = local_counts[2];

            const auto global_deletions = global_counts[0] + global_counts[1];
            const auto global_creations = global_counts[2];

            if (mpiPP::MPIRank::root_rank() == my_rank) {
                total_synapse_deletions += global_deletions;
                total_synapse_creations += global_creations;
            }

            Timers::stop_and_add(TimerRegion::UPDATE_CONNECTIVITY);

            Timers::start(TimerRegion::UPDATE_FIRE_HISTORY);
            const auto& fired_recorder = neurons->get_neuron_model()->get_fired_status_recorder();
            fired_recorder->reset(FiredStatusRecorder::FireRecorderPeriod::Plasticity);
            Timers::stop_and_add(TimerRegion::UPDATE_FIRE_HISTORY);

            Timers::start(TimerRegion::PRINT_IO);

            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdate, false, "{}: {} {} {}", step,
                                    global_creations, global_deletions, global_creations - global_deletions);
            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateCSV, false, "{};{};{};{}", step,
                                    global_creations, global_deletions, global_creations - global_deletions);
            LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateLocal, false, "{}: {} {} {}", step,
                                    local_creations, local_deletions, local_creations - local_deletions);

            neurons->print_sums_of_synapses_and_elements_to_log_file_on_rank_0(step, num_axons_deleted,
                                                                               num_dendrites_deleted,
                                                                               num_synapses_created);

            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (group_monitor_enabled && interval_neuron_monitor.hits_step(step)) {
            // Update group monitor
            Timers::start(TimerRegion::CAPTURE_GROUP_MONITORS);

            Timers::start(TimerRegion::GROUP_MONITORS_PREPARE);
            ranges::for_each(*group_monitors | ranges::views::values, &GroupMonitor::prepare_recording);

            Timers::stop_and_add(TimerRegion::GROUP_MONITORS_PREPARE);
            Timers::start(TimerRegion::GROUP_MONITORS_REQUEST);

            global_group_mapper->clear_cache();

            ranges::for_each(*group_monitors | ranges::views::values, &GroupMonitor::request_data);

            Timers::stop_and_add(TimerRegion::GROUP_MONITORS_REQUEST);
            Timers::start(TimerRegion::GROUP_MONITORS_EXCHANGE);

            global_group_mapper->exchange_requests();

            Timers::stop_and_add(TimerRegion::GROUP_MONITORS_EXCHANGE);
            Timers::start(TimerRegion::GROUP_MONITORS_RECORD_DATA);

            ranges::for_each(*group_monitors | ranges::views::values, &GroupMonitor::monitor_connectivity);

            const auto disable_flags = neurons->get_disable_flags();
            const auto& local_translator = neurons->get_local_group_translator();

            for (const auto neuron_id : NeuronIDRange::range(neurons->get_number_neurons())) {
                const auto id = neuron_id.get_neuron_id();
                if (disable_flags[id] == UpdateStatus::Disabled) {
                    continue;
                }
                const auto& group_ids = local_translator->get_group_ids_for_neuron_id(id);

                for (const auto& group_id : group_ids) {
                    auto& group_monitor = group_monitors->at(group_id);
                    group_monitor.record_data(neuron_id);
                }
            }

            Timers::stop_and_add(TimerRegion::GROUP_MONITORS_RECORD_DATA);
            Timers::start(TimerRegion::GROUP_MONITORS_FINISH);

            ranges::for_each(*group_monitors | ranges::views::values, &GroupMonitor::finish_recording);

            Timers::stop_and_add(TimerRegion::GROUP_MONITORS_FINISH);

            const auto& fired_recorder = neurons->get_neuron_model()->get_fired_status_recorder();
            fired_recorder->reset(FiredStatusRecorder::FireRecorderPeriod::GroupMonitor);
            neurons->get_extra_info()->reset_deletion_log();

            Timers::stop_and_add(TimerRegion::CAPTURE_GROUP_MONITORS);
        }

        if (interval_calcium_log.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            neurons->print_calcium_values_to_file(step);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (interval_synaptic_input_log.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            // neurons->print_synaptic_inputs_to_file(step);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (interval_network_log.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            neurons->print_network_graph_to_log_file(step, true);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (interval_statistics_log.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            neurons->print_neurons_overview_to_log_file_on_rank_0(step);

            for (auto& [attribute, vector] : statistics) {
                vector.emplace_back(neurons->get_statistics(attribute));
            }
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (interval_flush_all_logs_step.hits_step(step)) {
            Timers::start(TimerRegion::PRINT_IO);
            neuron_monitor->flush_current_contents();
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (step % Config::flush_group_monitor_step == 0) {
            Timers::start(TimerRegion::PRINT_IO);
            ranges::for_each(*group_monitors | ranges::views::values, &GroupMonitor::write_data_to_file);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }

        if (step % Config::console_update_step == 0) {
            if (my_rank != mpiPP::MPIRank::root_rank()) {
                continue;
            }

            const auto net_creations = total_synapse_creations - total_synapse_deletions;

            Timers::start(TimerRegion::PRINT_IO);
            LogFiles::write_to_file(LogFiles::EventType::Cout, true,
                                    "[Step: {}\t] Total up to now     (creations, deletions, net):\t{}\t\t{}\t\t{}",
                                    step, total_synapse_creations, total_synapse_deletions, net_creations);
            Timers::stop_and_add(TimerRegion::PRINT_IO);
        }
    }

    Timers::stop_and_add(TimerRegion::SIMULATION_LOOP);

    neurons->record_memory_footprint(footprint);
    neurons->record_usage_footprint(usage_footprint);
    print_memory_footprint();

    delta_synapse_creations = total_synapse_creations - previous_synapse_creations;
    delta_synapse_deletions = total_synapse_deletions - previous_synapse_deletions;

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Final flush of neuron monitors");
    neuron_monitor->flush_current_contents();

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Final flush of group monitors");
    ranges::for_each(*group_monitors | ranges::views::values, &GroupMonitor::write_data_to_file);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Print positions");
    neurons->print_positions_to_log_file();

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Print group mapping");
    neurons->print_group_mapping_to_log_file();

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Print neuron to group mapping");
    neurons->print_groups_to_log_file();

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Print group name to file name mapping");
    neurons->print_groups_to_file_name_to_log_file();

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Final flush of calcium values");
    neurons->print_calcium_values_to_file(step);
}

double get_max_rss_mb() {
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<double>(usage.ru_maxrss) / 1024.0; // MB
}

void Simulation::finalize() const {

    const auto net_creations = total_synapse_creations - total_synapse_deletions;
    const auto previous_net_creations = delta_synapse_creations - delta_synapse_deletions;

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(),
                                 "Total up to now     (creations, deletions, net): {}\t{}\t{}\nDiff. from previous (creations, deletions, net): {}\t{}\t{}\nEND: {}",
                                 total_synapse_creations, total_synapse_deletions, net_creations,
                                 delta_synapse_creations, delta_synapse_deletions, previous_net_creations,
                                 Timers::wall_clock_time());

    const auto end_time = std::chrono::system_clock::now();
    const auto duration_s = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(),
                                 "Simulation took {} seconds",
                                 duration_s);
    essentials->insert("Simulation-Time-Seconds", duration_s);
    essentials->insert("Max-node-degree-in-sim", neurons->get_network_graph()->highest_number_synapses_per_neuron_until_now());

    neurons->print_calcium_statistics_to_essentials(essentials);
#ifndef RELEARN_CUDA_ENABLED
    neurons->print_synaptic_changes_to_essentials(essentials);
#endif

    essentials->insert("Created-Synapses", total_synapse_creations);
    essentials->insert("Deleted-Synapses", total_synapse_deletions);
    essentials->insert("net-Synapses", net_creations);

#if RELEARN_CUDA_ENABLED
    const auto gpu_mem_used = get_gpu_max_memory_used();
    const auto overall_gpu_mem_used = mpiPP::MPIReductions::reduce_sum(gpu_mem_used);
    const auto overall_gpu_mem_used_mb = static_cast<double>(overall_gpu_mem_used) / 1024.0 / 1024.0;
    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Used gpu memory {} MB", overall_gpu_mem_used_mb);
    essentials->insert("gpu-mem-used", overall_gpu_mem_used_mb);

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Send {} and received {} bytes", get_send_bytes(), get_recv_bytes());
#endif

    auto ss = std::stringstream{};
    essentials->print(ss);

    LogFiles::write_to_file(LogFiles::EventType::Essentials, false, ss.str());

    // Print final network graph
    neurons->print_network_graph_to_log_file(step, false);

    neurons->get_neuron_model()->finalize();

    const auto mem_usage = get_max_rss_mb();
    double sum_mem_usage{};
    MPI_Reduce(&mem_usage, &sum_mem_usage, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Simulation used {} CPU memory", sum_mem_usage);
}

void Simulation::snapshot_monitors() {
    // record data at step 0
    Timers::start(TimerRegion::CAPTURE_NEURON_MONITORS);
    neuron_monitor->record_data(0);
    Timers::stop_and_add(TimerRegion::CAPTURE_NEURON_MONITORS);

    Timers::start(TimerRegion::CAPTURE_FIRE_STEPS);
    const auto& fired_recorder = neurons->get_neuron_model()->get_fired_status_recorder();
    fired_recorder->reset(FiredStatusRecorder::FireRecorderPeriod::NeuronMonitor);
    Timers::stop_and_add(TimerRegion::CAPTURE_FIRE_STEPS);
}

void Simulation::set_static_neurons(std::vector<NeuronID> _static_neurons) {
    static_neurons = std::move(_static_neurons);
}

void Simulation::final_timer_print() const {

    Timers::collect_timer_data();
    Timers::print_local_human_readable();

    Timers::print_human_readable();
    Timers::print_extrap(step, neurons->get_number_neurons());
    Timers::print_json();
}

void Simulation::print_memory_footprint() const {
    auto ss = std::stringstream{};
    for (const auto& [descr, mem_used] : footprint->get_descriptions()) {
        ss << descr << ":" << mem_used << '\n';
    }
    LogFiles::write_raw_string_to_file(LogFiles::EventType::MemoryFootprint, false, ss.str());

    auto cpu_mem = 0U;
    auto gpu_mem = 0U;
    for (const auto& [descr, mem_used] : footprint->get_descriptions()) {
        if (descr.find("GPU") == std::string::npos) {
            cpu_mem += static_cast<unsigned int>(mem_used);
        } else {
            gpu_mem += static_cast<unsigned int>(mem_used);
        }
    }
    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Memory footprint total cpu {} bytes", cpu_mem);
    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Memory footprint total gpu {} bytes", gpu_mem);

    auto ss_usage = std::stringstream{};
    for (const auto& [descr, mem_used] : usage_footprint->get_descriptions()) {
        ss_usage << descr << ":" << mem_used << "%" << '\n';
    }
    LogFiles::write_raw_string_to_file(LogFiles::EventType::UsageFootprint, false, ss_usage.str());
}
