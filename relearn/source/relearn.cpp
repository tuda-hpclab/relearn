/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Config.h"

#include "cuda/CudaConfig.h"

#ifdef RELEARN_CUDA_ENABLED
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#endif

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "algorithm/Internal/octree/BaseCell.h"
#include "algorithm/Internal/octree/Cell.h"
#include "algorithm/Kernel/Gamma.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Kernel.h"
#include "algorithm/Kernel/KernelBase.h"
#include "algorithm/Kernel/KernelType.h"
#include "algorithm/Kernel/Linear.h"
#include "algorithm/Kernel/Weibull.h"
#include "algorithm/NaiveInternal/NaiveCell.h"
#include "algorithm/VirtualPlasticityElement.h"
#include "cuda/firing/FireStatusCommunicatorGPUUncompressed.h"
#include "cuda/network_graph/NetworkGPUType.h"
#include "io/AxonPositionIO.h"
#include "io/BackgroundActivityIO.h"
#include "io/CalciumIO.h"
#include "io/InteractiveNeuronIO.h"
#include "io/LogFiles.h"
#include "io/NeuronToAlgorithmIO.h"
#include "io/SynapticElementsIO.h"
#include "io/parser/ActivityInputParser.h"
#include "io/parser/MonitorParser.h"
#include "neurons/Neurons.h"
#include "neurons/calcium/AbsoluteDecayCalciumCalculator.h"
#include "neurons/calcium/CalciumCalculator.h"
#include "neurons/calcium/RelativeDecayCalciumCalculator.h"
#include "neurons/enums/AxonsType.h"
#include "neurons/enums/CalciumCalculatorType.h"
#include "neurons/enums/FiredStatusCommunicatorType.h"
#include "neurons/enums/GrowthrateCalculatorType.h"
#include "neurons/enums/NeuronModelType.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/SynapticInputCalculatorType.h"
#include "neurons/firing/FiredStatusApproximator.h"
#include "neurons/firing/FiredStatusCommunicationMap.h"
#include "neurons/firing/FiredStatusCommunicator.h"
#include "neurons/growthrate/ConstantGrowthrateCalculator.h"
#include "neurons/growthrate/GrowthrateCalculator.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionFinder.h"
#include "neurons/input/ConstantActivityInput.h"
#include "neurons/input/NormalActivityInput.h"
#include "neurons/models/NeuronModel.h"
#include "neurons/models/aeif/AEIFModel.h"
#include "neurons/models/fitzhughnagumo/FitzHughNagumoModel.h"
#include "neurons/models/izhikevich/IzhikevichModel.h"
#include "neurons/models/poisson/PoissonModel.h"
#include "neurons/synaptic_elements/Axons.h"
#include "neurons/synaptic_elements/Dendrites.h"
#include "neurons/synaptic_elements/MultiPositionAxons.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "sim/Simulation.h"
#include "sim/file/MultipleSubdomainsFromFile.h"
#include "sim/random/SubdomainFromNeuronDensity.h"
#include "sim/random/SubdomainFromNeuronPerRank.h"
#include "structure/Partition.h"
#include "structure/SpaceFillingCurveType.h"
#include "types/BasicTypes.h"
#include "types/SynapseTypes.h"
#include "util/NeuronID.h"
#include "util/NeuronIDRange.h"
#include "util/OMPHelper.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Timers.h"
#include "util/Vec3.h"

#include <boost/container_hash/hash.hpp>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/ExtraValidators.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Validators.hpp>

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Interval.hpp>

#include <mpi-wrapper/MPIWrapper.h>
#include <mpi-wrapper/core/MPICounters.h>
#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPISynchronization.h>

#include <spdlog/spdlog.h>

#include <mpi.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

struct empty_t {
    using position_type = VirtualPlasticityElement::position_type;
    using counter_type = VirtualPlasticityElement::counter_type;

    constexpr static bool has_excitatory_dendrite = false;
    constexpr static bool has_inhibitory_dendrite = false;
    constexpr static bool has_excitatory_axon = false;
    constexpr static bool has_inhibitory_axon = false;
};

namespace {
void print_sizes() {
    constexpr auto number_bits_in_byte = CHAR_BIT;

    constexpr auto sizeof_position_type = sizeof(RelearnTypes::position_type);
    constexpr auto sizeof_vec3_size_t = sizeof(Vec3s);

    constexpr auto sizeof_virtual_plasticity_element = sizeof(VirtualPlasticityElement);

    constexpr auto sizeof_empty_t = sizeof(empty_t);
    constexpr auto sizeof_fmm_cell_attributes = sizeof(FastMultipoleMethodCell);
    constexpr auto sizeof_bh_cell_attributes = sizeof(BarnesHutCell);
    constexpr auto sizeof_bh_naive_attributes = sizeof(NaiveCell);

    constexpr auto sizeof_empty_cell = sizeof(Cell<empty_t>);
    constexpr auto sizeof_fmm_cell = sizeof(Cell<FastMultipoleMethodCell>);
    constexpr auto sizeof_bh_cell = sizeof(Cell<BarnesHutCell>);
    constexpr auto sizeof_naive_cell = sizeof(Cell<NaiveCell>);

    constexpr auto sizeof_octree_node = sizeof(OctreeNode<empty_t>);
    constexpr auto sizeof_fmm_octree_node = sizeof(OctreeNode<FastMultipoleMethodCell>);
    constexpr auto sizeof_bh_octree_node = sizeof(OctreeNode<BarnesHutCell>);
    constexpr auto sizeof_naive_octree_node = sizeof(OctreeNode<NaiveCell>);

    constexpr auto sizeof_neuron_id = sizeof(NeuronID);
    constexpr auto sizeof_rank_neuron_id = sizeof(RankNeuronId);

    constexpr auto sizeof_mpi_rank = sizeof(mpiPP::MPIRank);
    constexpr auto sizeof_int = sizeof(int);

    constexpr auto sizeof_plastic_local_synapse = sizeof(PlasticLocalSynapse);
    constexpr auto sizeof_plastic_distant_in_synapse = sizeof(PlasticDistantInSynapse);
    constexpr auto sizeof_plastic_distant_out_synapse = sizeof(PlasticDistantOutSynapse);

    constexpr auto sizeof_static_local_synapse = sizeof(StaticLocalSynapse);
    constexpr auto sizeof_static_distant_in_synapse = sizeof(StaticDistantInSynapse);
    constexpr auto sizeof_static_distant_out_synapse = sizeof(StaticDistantOutSynapse);

    constexpr auto sizeof_empty_base_cell = sizeof(BaseCell<false, false, false, false>);
    constexpr auto sizeof_full_base_cell = sizeof(BaseCell<true, true, true, true>);
    constexpr auto sizeof_dendrites_base_cell = sizeof(BaseCell<true, true, false, false>);
    constexpr auto sizeof_axons_base_cell = sizeof(BaseCell<false, false, true, true>);

    auto ss = std::stringstream{};

    ss << '\n';

    ss << "Number of bits in a byte: " << number_bits_in_byte << '\n';

    ss << "Size of position_type: " << sizeof_position_type << '\n';
    ss << "Size of Vec3s: " << sizeof_vec3_size_t << '\n';

    ss << "Size of VirtualPlasticityElement: " << sizeof_virtual_plasticity_element << '\n';
    ss << "Size of FastMultipoleMethodCell: " << sizeof_fmm_cell_attributes << '\n';

    ss << "Size of empty_t: " << sizeof_empty_t << '\n';
    ss << "Size of BarnesHutCell: " << sizeof_bh_cell_attributes << '\n';
    ss << "Size of NaiveCell: " << sizeof_bh_naive_attributes << '\n';

    ss << "Size of Cell<empty_t>: " << sizeof_empty_cell << '\n';
    ss << "Size of Cell<FastMultipoleMethodCell>: " << sizeof_fmm_cell << '\n';
    ss << "Size of Cell<BarnesHutCell>: " << sizeof_bh_cell << '\n';
    ss << "Size of Cell<NaiveCell>: " << sizeof_naive_cell << '\n';

    ss << "Size of OctreeNode<empty_t>: " << sizeof_octree_node << '\n';
    ss << "Size of OctreeNode<FastMultipoleMethodCell>: " << sizeof_fmm_octree_node << '\n';
    ss << "Size of OctreeNode<BarnesHutCell>: " << sizeof_bh_octree_node << '\n';
    ss << "Size of OctreeNode<NaiveCell>: " << sizeof_naive_octree_node << '\n';

    ss << "Size of NeuronID: " << sizeof_neuron_id << '\n';
    ss << "Size of RankNeuronID: " << sizeof_rank_neuron_id << '\n';

    ss << "Size of MPIRank: " << sizeof_mpi_rank << '\n';
    ss << "Size of int: " << sizeof_int << '\n';

    ss << "Size of PlasticLocalSynapse: " << sizeof_plastic_local_synapse << '\n';
    ss << "Size of PlasticDistantInSynapse: " << sizeof_plastic_distant_in_synapse << '\n';
    ss << "Size of PlasticDistantOutSynapse: " << sizeof_plastic_distant_out_synapse << '\n';

    ss << "Size of StaticLocalSynapse: " << sizeof_static_local_synapse << '\n';
    ss << "Size of StaticDistantInSynapse: " << sizeof_static_distant_in_synapse << '\n';
    ss << "Size of StaticDistantOutSynapse: " << sizeof_static_distant_out_synapse << '\n';

    ss << "Size of BaseCell<false, false, false, false>: " << sizeof_empty_base_cell << '\n';
    ss << "Size of BaseCell<true, true, true, true>: " << sizeof_full_base_cell << '\n';
    ss << "Size of BaseCell<true, true, false, false>: " << sizeof_dendrites_base_cell << '\n';
    ss << "Size of BaseCell<false, false, true, true>: " << sizeof_axons_base_cell << '\n';

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), ss.str());
}

void print_arguments(int argc, char** argv) {
    auto ss = std::stringstream{};

    for (auto i = 0; i < argc; i++) {
        ss << argv[i] << ' ';
    }

    LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), ss.str());
}

void simulate(Simulation simulation, const RelearnTypes::step_type simulation_steps, const bool interactive) {

    mpiPP::MPISynchronization::barrier();

    auto simulate = [&simulation, &simulation_steps]() {
        Timers::start(TimerRegion::ALL);
        simulation.simulate(simulation_steps);

        mpiPP::MPISynchronization::barrier();

        simulation.finalize();
        Timers::stop_and_add(TimerRegion::ALL);
    };

    Timers::stop_and_add(TimerRegion::ALL);
    simulate();

    if (interactive) {
        while (true) {
            spdlog::info("Interactive run. Run another {} simulation steps? [y/n]\n", simulation_steps);

            auto yn = 'n';
            std::cin >> std::ws >> yn;

            if (yn == 'n' || yn == 'N') {
                break;
            }

            if (yn == 'y' || yn == 'Y') {
                simulate();
            } else {
                RelearnException::fail("Input for question to run another {} simulation steps was not valid.", simulation_steps);
            }
        }
    }

    simulation.final_timer_print();

    LogFiles::write_to_file(LogFiles::EventType::Cout, false, "Number of bytes send: {}, Number  of bytes received: {}, Number  of bytes accessed remotely: {}",
                            mpiPP::MPICounters::get_number_bytes_sent(), mpiPP::MPICounters::get_number_bytes_received(), mpiPP::MPICounters::get_number_bytes_remotely_accessed());
}
} // namespace

int main(int argc, char** argv) {
    try {
        /**
         * Init MPI and store some MPI infos
         */
        mpiPP::MPIWrapper::init(argc, argv);

        print_arguments(argc, argv);
        print_sizes();

        Timers::start(TimerRegion::ALL);

        if (Config::do_debug_checks) {
            LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "I'm performing Debug Checks");
        } else {
            LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "I'm skipping Debug Checks");
        }

        const auto my_rank = mpiPP::MPIInfo::get_my_rank();
        const auto num_ranks = mpiPP::MPIInfo::get_number_ranks();

        // Command line arguments
        auto app = CLI::App{ "" };

        auto chosen_axons = AxonsType::Normal;
        const auto cli_parse_axons = std::map<std::string, AxonsType>{
            { "normal", AxonsType::Normal },
            { "multi-position", AxonsType::MultiPosition }
        };

        // sleep(10);

        auto chosen_algorithm = AlgorithmEnum::BarnesHut;
        const auto cli_parse_algorithm = std::map<std::string, AlgorithmEnum>{
            { "naive", AlgorithmEnum::Naive },
            { "barnes-hut", AlgorithmEnum::BarnesHut },
            { "barnes-hut-inverted", AlgorithmEnum::BarnesHutInverted },
            { "barnes-hut-location-aware", AlgorithmEnum::BarnesHutLocationAware },
            { "fast-multipole-methods", AlgorithmEnum::FastMultipoleMethod },
            { "combined-algorithms", AlgorithmEnum::CombinedAlgorithms },
            { "naive-cuda", AlgorithmEnum::NaiveCuda },
            { "barnes-hut-cuda", AlgorithmEnum::BarnesHutCuda },
            { "barnes-hut-location-aware-modified", AlgorithmEnum::BarnesHutLocationAwareModified },
            { "barnes-hut-restricted", AlgorithmEnum::BarnesHutRestricted },
            { "fast-multipole-methods", AlgorithmEnum::FastMultipoleMethod }
        };

        auto chosen_neuron_model = NeuronModelType::Poisson;
        const auto cli_parse_neuron_model = std::map<std::string, NeuronModelType>{
            { "poisson", NeuronModelType::Poisson },
            { "izhikevich", NeuronModelType::Izhikevich },
            { "aeif", NeuronModelType::AEIF },
            { "fitzhughnagumo", NeuronModelType::FitzHughNagumo }
        };

        auto chosen_synapse_deleter = SynapseDeletionFinderType::Random;
        const auto cli_parse_synapse_deleter = std::map<std::string, SynapseDeletionFinderType>{
            { "random", SynapseDeletionFinderType::Random },
            { "inverse", SynapseDeletionFinderType::InverseLength },
            //     { "coactivation", SynapseDeletionFinderType::CoActivation },
        };

        auto chosen_growth_rate_calculator = GrowthrateCalculatorType::Constant;
        const auto cli_parse_growth_rate_calculator = std::map<std::string, GrowthrateCalculatorType>{
            { "constant", GrowthrateCalculatorType::Constant },
        };

        auto chosen_kernel_type = KernelType::Gaussian;
        const auto cli_parse_kernel_type = std::map<std::string, KernelType>{
            { "gamma", KernelType::Gamma },
            { "gaussian", KernelType::Gaussian },
            { "linear", KernelType::Linear },
            { "weibull", KernelType::Weibull }
        };

        auto calcium_decay_type = CalciumCalculatorType::Normal;
        const auto cli_parse_decay_type = std::map<std::string, CalciumCalculatorType>{
            { "none", CalciumCalculatorType::Normal },
            { "relative", CalciumCalculatorType::RelativeDecay },
            { "absolute", CalciumCalculatorType::AbsoluteDecay }
        };

        auto chosen_synapse_input_calculator_type = SynapticInputCalculatorType::Linear;
        const auto cli_parse_synapse_input_calculator_type = std::map<std::string, SynapticInputCalculatorType>{
            { "linear", SynapticInputCalculatorType::Linear },
            { "logarithmic", SynapticInputCalculatorType::Logarithmic },
            { "hyptan", SynapticInputCalculatorType::HyperbolicTangent },
        };

        auto chosen_network_type = NetworkGPUType::MEMORY_POOL;
        const auto cli_network_type = std::map<std::string, NetworkGPUType>{
            { "memory-pool", NetworkGPUType::MEMORY_POOL },
            { "memory-pool-weighted", NetworkGPUType::MEMORY_POOL_WEIGHTED },
        };

        auto chosen_fired_status_communicator_type =
#ifdef RELEARN_CUDA_ENABLED
            FiredStatusCommunicatorType::GpuUncompressed;
#else
            FiredStatusCommunicatorType::Map;
#endif
        const auto cli_parse_fired_status_communicator_type = std::map<std::string, FiredStatusCommunicatorType>{
            { "map", FiredStatusCommunicatorType::Map },
            { "approximator", FiredStatusCommunicatorType::Approximator },
            { "gpu-uncompressed", FiredStatusCommunicatorType::GpuUncompressed },
        };

        auto chosen_space_filling_curve_type = SpaceFillingCurveType::MortonCurve;
        const auto cli_parse_space_filling_curve_type = std::map<std::string, SpaceFillingCurveType>{
            { "morton", SpaceFillingCurveType::MortonCurve },
            { "hilbert", SpaceFillingCurveType::HilbertCurve },
        };

        auto simulation_steps = RelearnTypes::step_type{};
        app.add_option("-s,--steps", simulation_steps, "Simulation steps in ms.")->required();

        auto first_plasticity_step = RelearnTypes::step_type{ 0 };
        app.add_option("--first-plasticity-step", first_plasticity_step, "The first step in which the plasticity is updated.");

        auto last_plasticity_step = RelearnTypes::step_type{ std::numeric_limits<RelearnTypes::step_type>::max() };
        app.add_option("--last-plasticity-step", last_plasticity_step, "The last step in which the plasticity is updated.");

        app.add_option("--plasticity-update-step", Config::plasticity_update_step, "The interval of steps between a plasticity update.");
        app.add_option("--calcium-log-step", Config::calcium_log_step, "Sets the interval for logging all calcium values.");
        app.add_option("--fire-rate-log-step", Config::fire_rate_log_step, "Sets the interval for logging all fire rates.");
        app.add_option("--synaptic-input-log-step", Config::synaptic_input_log_step, "Sets the interval for logging all synaptic inputs.");
        app.add_option("--network-log-step", Config::network_log_step, "Steps between saving the network graph");
        app.add_option("--monitor-steps", Config::neuron_monitor_log_step, "Every time the neuron state is captured");
        app.add_option("--monitor-ensemble-steps", Config::group_monitor_log_step, "Every time the ensemble information are captured");

        auto* flag_group_monitor = app.add_flag("--enable-group-monitor", "Enables the group monitor");
        auto* flag_group_monitor_connectivity = app.add_flag("--enable-group-monitor-connectivity", "Enables the monitoring of the connectivity by the group monitor");
        flag_group_monitor_connectivity->needs(flag_group_monitor);

        const auto* flag_interactive = app.add_flag("-i,--interactive", "Run interactively.");

        auto random_seed = std::uint64_t{ 0 };
        app.add_option("-r,--random-seed", random_seed, "Random seed. Default: 0.");

        auto openmp_threads = 1;
        app.add_option("--openmp", openmp_threads, "Number of OpenMP Threads.");

        // auto* flag_gpu = app.add_flag("--use-gpu", Config::use_gpu, "Enables computing on the GPU using CUDA (if present)");

        auto log_path = std::filesystem::path{};
        auto* const opt_log_path = app.add_option("-l,--log-path", log_path, "Path for log files.");

        auto log_prefix = std::string{};
        const auto* opt_log_prefix = app.add_option("-p,--log-prefix", log_prefix, "Prefix for log files.");

        const auto* flag_enable_printing_events = app.add_flag("--print-events", "Enables printing the events to a file.");
        const auto* flag_disable_printing_positions = app.add_flag("--no-print-positions", "Disables printing the positions to a file.");
        const auto* flag_disable_printing_network = app.add_flag("--no-print-network", "Disables printing the network to a file.");
        const auto* flag_disable_printing_plasticity = app.add_flag("--no-print-plasticity", "Disables printing the plasticity changes to a file.");
        const auto* flag_disable_printing_calcium = app.add_flag("--no-print-calcium", "Disables printing the calcium changes to a file.");
        const auto* flag_disable_printing_fire_rate = app.add_flag("--no-print-fire-rate", "Disables printing the fire rate changes to a file.");
        const auto* flag_disable_printing_fire_steps = app.add_flag("--no-print-fire-steps", "Disables printing of the steps when each neuron fired.");
        const auto* flag_disable_printing_overview = app.add_flag("--no-print-overview", "Disables printing the overviews to a file.");
        const auto* flag_disable_printing_group_mapping = app.add_flag("--no-print-mapping", "Disables printing the group mapping to a file.");
        const auto* flag_disable_printing_neuron_to_groups = app.add_flag("--no-print-neuron-to-groups", "Disables printing the neuron to group mapping to a file.");
        const auto* flag_disable_printing_group_name_to_file_name = app.add_flag("--no-print-group-name-to-file-name", "Disables printing the group name to file name mapping to a file");
        const auto* flag_disable_printing_sums = app.add_flag("--no-print-sums", "Disables printing the sums to a file.");

        [[maybe_unused]] const auto* flag_cuda_aware_mpi = app.add_flag("--cuda-aware-mpi", "Use cuda aware mpi");

        const auto* flag_pre_drawn_cpu_random_values = app.add_flag("--use-pre-drawn-cpu-random-values", "");

        app.add_option("--max-file-size-fire-rates", Config::fire_rates_max_file_size,
                       "The maximum file size for the fire rates files. 0 will be interpreted as limitless");

        app.add_option("--max-file-size-calcium", Config::calcium_max_file_size,
                       "The maximum file size for the calcium files. 0 will be interpreted as limitless");

        app.add_option("--max-file-size-extreme-calcium", Config::extreme_calcium_max_file_size,
                       "The maximum file size for the extreme calcium files. 0 will be interpreted as limitless");

        app.add_option("--max-file-size-fire-steps", Config::fire_steps_max_file_size,
                       "The maximum file size for the fire steps files. 0 will be interpreted as limitless");

        auto number_neurons = RelearnTypes::number_neurons_type{};
        auto* const opt_num_neurons = app.add_option("-n,--num-neurons", number_neurons, "Number of neurons. This option only works with one MPI rank!");

        auto number_neurons_per_rank = RelearnTypes::number_neurons_type{};
        auto* const opt_num_neurons_per_rank = app.add_option("--num-neurons-per-rank", number_neurons_per_rank, "Number neurons per MPI rank.");

        auto fraction_excitatory_neurons = RelearnTypes::percentage_type{ 1.0 };
        app.add_option("--fraction-excitatory-neurons", fraction_excitatory_neurons, "The fraction of excitatory neurons, must be from [0.0, 1.0]. Requires --num-neurons or --num-neurons-per-rank to take effect.");

        auto um_per_neuron = RelearnTypes::space_type{ 1.0 };
        app.add_option("--um-per-neuron", um_per_neuron, "The micrometer per neuron in one dimension, must be from (0.0, \\inf). Requires --num-neurons or --num-neurons-per-rank to take effect.");

        auto file_positions = std::filesystem::path{};
        auto* const opt_file_positions = app.add_option("-f,--file", file_positions, "File or directory with neuron positions.");

        auto file_groups = std::filesystem::path{};
        auto* const opt_file_groups = app.add_option("--file-groups", file_groups, "File or directory with neuron groups");

        auto file_network = std::filesystem::path{};
        auto* const opt_file_network = app.add_option("-g,--graph", file_network, "Folder that contains the files with the networks. The network files must be names rank_0_in_network.txt and rank_0_out_network.txt. This option only works with one MPI rank!");

        auto file_enable_interrupts = std::filesystem::path{};
        auto* const opt_file_enable_interrupts = app.add_option("--enable-interrupts", file_enable_interrupts, "File with the enable interrupts.");

        auto file_disable_interrupts = std::filesystem::path{};
        auto* const opt_file_disable_interrupts = app.add_option("--disable-interrupts", file_disable_interrupts, "File with the disable interrupts.");

        auto file_creation_interrupts = std::filesystem::path{};
        auto* const opt_file_creation_interrupts = app.add_option("--creation-interrupts", file_creation_interrupts, "File with the creation interrupts.");

        auto* const opt_algorithm = app.add_option("-a,--algorithm", chosen_algorithm, "The algorithm that is used for finding the targets");
        opt_algorithm->required()->transform(CLI::CheckedTransformer(cli_parse_algorithm, CLI::ignore_case));

        auto accept_criterion = RelearnTypes::acceptance_criterion_type{ Constants::bh_default_theta };
        const auto* const opt_accept_criterion = app.add_option("-t,--theta", accept_criterion, "Theta, the acceptance criterion for Barnes-Hut. Default: 0.3. Requires Barnes-Hut or inverted Barnes-Hut.");

        auto* const opt_kernel_type = app.add_option("--kernel-type", chosen_kernel_type, "The probability kernel type, cannot be set for the fast multipole methods.");
        opt_kernel_type->transform(CLI::CheckedTransformer(cli_parse_kernel_type, CLI::ignore_case));

        auto gamma_k = RelearnTypes::attraction_type{ GammaDistributionKernel::default_k };
        app.add_option("--gamma-k", gamma_k, "Shape parameter for the gamma probability kernel.");

        auto gamma_theta = RelearnTypes::attraction_type{ GammaDistributionKernel::default_theta };
        app.add_option("--gamma-theta", gamma_theta, "Scale parameter for the gamma probability kernel.");

        auto gaussian_sigma = RelearnTypes::attraction_type{ GaussianDistributionKernel::default_sigma };
        app.add_option("--gaussian-sigma", gaussian_sigma, "Scaling parameter for the gaussian probability kernel. Default: 750");

        auto gaussian_mu = RelearnTypes::attraction_type{ GaussianDistributionKernel::default_mu };
        app.add_option("--gaussian-mu", gaussian_mu, "Translation parameter for the gaussian probability kernel. Default: 0");

        auto linear_cutoff = RelearnTypes::attraction_type{ LinearDistributionKernel::default_cutoff };
        app.add_option("--linear-cutoff", linear_cutoff, "Cut-off parameter for the linear probability kernel. Default: +inf");

        auto weibull_k = RelearnTypes::attraction_type{ WeibullDistributionKernel::default_k };
        app.add_option("--weibull-k", weibull_k, "Shape parameter for the weibull probability kernel.");

        auto weibull_b = RelearnTypes::attraction_type{ WeibullDistributionKernel::default_b };
        app.add_option("--weibull-b", weibull_b, "Scale parameter for the weibull probability kernel.");

        auto individual_algorithms_for_neurons_file_path = std::filesystem::path{};
        auto* const opt_individual_algorithms_for_neurons_file_path = app.add_option("--combined-algorithms-file-path", individual_algorithms_for_neurons_file_path, "File with algorithms and the neurons that should use the respective algorithms.");

        auto* const opt_neuron_model = app.add_option("--neuron-model", chosen_neuron_model, "The neuron model.");
        opt_neuron_model->transform(CLI::CheckedTransformer(cli_parse_neuron_model, CLI::ignore_case));

        auto* const opt_synapse_deleter = app.add_option("--synapse-deleter", chosen_synapse_deleter, "The algorithm for deleting synapses.");
        opt_synapse_deleter->transform(CLI::CheckedTransformer(cli_parse_synapse_deleter, CLI::ignore_case));

        auto static_neurons_str = std::string{};
        auto* opt_static_neurons = app.add_option("--static-neurons", static_neurons_str, "String with neuron ids for static neurons. Format is <mpi_rank>:<neuron_id>;<mpi_rank>:<neuron_id>;... where <mpi_rank> can be -1 to indicate \"on every rank\". Alternatively use group names instead of neuron ids");

        auto file_external_stimulation = std::filesystem::path{};
        auto* opt_file_external_stimulation = app.add_option("--external-stimulation", file_external_stimulation, "File with the external stimulation.");

        auto base_background_activity = RelearnTypes::activity_type{ ConstantActivityInput::default_constant_activity };
        app.add_option("--base-background-activity", base_background_activity,
                       "The base background activity by which all neurons are excited");
        auto flexible_background_file_path = std::filesystem::path{};
        auto* const opt_flexible_background_file_path = app.add_option("--background-activity-file-path", flexible_background_file_path,
                                                                       "The file path for the flexible background activity");

        auto background_activity_mean = RelearnTypes::activity_type{ NormalActivityInput::default_mean_activity };
        app.add_option("--background-activity-mean", background_activity_mean,
                       "The mean background activity by which all neurons are excited. The background activity is calculated N(mean, stddev)");

        auto background_activity_stddev = RelearnTypes::activity_type{ NormalActivityInput::default_stddev_activity };
        app.add_option("--background-activity-stddev", background_activity_stddev,
                       "The standard deviation of the background activity by which all neurons are excited. The background activity is calculated as N(mean, stddev)");

        auto activity_input_str = std::string{ "synaptic_equally_weighted" };
        app.add_option("--activity-input", activity_input_str,
                       R"help(This parameter specifies the activity input to be used inside the neuron modeles via a DSL.
The DSL is a simple functional language, where each activity type can be created
using its name (e.g.: `NormalActivityInput` -> `normal`) and the required arguments to construct it.
The values of variables or hidden parameters like the FiredStatusCommunicator
come from the CLI arguments passed to relearn and are the defaults otherwise.
The DSL has the following structure (quotes " are  for illustrative purposes only):

activity -> const(double constant)
activity -> normal(double mean, double stddev)
activity -> fastnormal(double mean, double stddev, size_t multiplier)
activity -> combined(activity, activities ...)
activity -> synaptic_equally_weighted
activity -> synaptic_scaling(double constant)
activity -> scale(activity, scaling_function fun)
activity -> stimulated

scaling_function -> "linear" | "logarithmic" | "hyperbolic_tangent"
double -> "background_base" | "background_mean" | "background_stddev" | literal
size_t -> literal)help");

        auto synapse_conductance = utility::as<RelearnTypes::activity_type>(0.03);
        app.add_option("--synapse-conductance", synapse_conductance, "The activity that is transferred to its neighbors when a neuron spikes. Default is 0.03");

        auto input_scale = RelearnTypes::activity_type{ 1.0 };
        app.add_option("--input-scale", input_scale, "The scale factor for the input via synapses. Default is 1.0");

        auto* const opt_synapse_input_calculator_type = app.add_option("--synapse-input-calculator-type", chosen_synapse_input_calculator_type, "The type calculator that transforms the synapse input.");
        opt_synapse_input_calculator_type->transform(CLI::CheckedTransformer(cli_parse_synapse_input_calculator_type, CLI::ignore_case));

        auto* const opt_fired_status_communicator_type = app.add_option("--fired-status-communicator-type", chosen_fired_status_communicator_type, "The type of communicator between MPI ranks that exchange the fired status.");
        opt_fired_status_communicator_type->transform(CLI::CheckedTransformer(cli_parse_fired_status_communicator_type, CLI::ignore_case));

        auto* const opt_space_filling_curve_type = app.add_option("--space-filling-curve-type", chosen_space_filling_curve_type, "The type of space filling curve.");
        opt_space_filling_curve_type->transform(CLI::CheckedTransformer(cli_parse_space_filling_curve_type, CLI::ignore_case));

        auto calcium_decay = RelearnTypes::calcium_type{ CalciumCalculator::default_tau_C };
        app.add_option("--calcium-decay", calcium_decay, "The decay constant for the intercellular calcium. Must be greater than 0.0");

        auto first_decay_step = RelearnTypes::step_type{ 0 };
        app.add_option("--target-calcium-first-decay-step", first_decay_step, "The first decay step of the calcium.");

        auto last_decay_step = RelearnTypes::step_type{ std::numeric_limits<RelearnTypes::step_type>::max() };
        app.add_option("--target-calcium-last-decay-step", last_decay_step, "The last decay step of the calcium.");

        auto target_calcium_decay_step = RelearnTypes::step_type{ 0 };
        app.add_option("--target-calcium-decay-step", target_calcium_decay_step, "The decay step for the target calcium values.");

        auto target_calcium_decay_amount = RelearnTypes::calcium_type{ 0.0 };
        app.add_option("--target-calcium-amount", target_calcium_decay_amount, "The decay amount for the target calcium values.");

        auto* const opt_decay_type = app.add_option("--decay-type", calcium_decay_type, "The decay type for the target calcium values.");
        opt_decay_type->transform(CLI::CheckedTransformer(cli_parse_decay_type, CLI::ignore_case));

        auto target_calcium = RelearnTypes::calcium_type{ CalciumCalculator::default_C_target };
        auto* const opt_target_calcium = app.add_option("--target-ca", target_calcium, "The target Ca2+ ions in each neuron. Default is 0.7.");

        auto initial_calcium = RelearnTypes::calcium_type{ 0.0 };
        auto* const opt_initial_calcium = app.add_option("--initial-ca", initial_calcium, "The initial Ca2+ ions in each neuron. Default is 0.0.");

        auto file_calcium = std::string{};
        auto* const opt_file_calcium = app.add_option("--file-calcium", file_calcium, "File with calcium values.");

        auto beta = RelearnTypes::calcium_type{ CalciumCalculator::default_beta };
        app.add_option("--beta", beta, "The amount of calcium ions gathered when a neuron fires. Default is 0.001.");

        auto h = std::uint32_t{ NeuronModel::default_h };
        app.add_option("--integration-step-size", h, "The step size for the numerical integration of the electrical activity. Default is 10.");

        auto retract_ratio = RelearnTypes::grown_type{ SynapticElements::default_vacant_retract_ratio };
        auto* const opt_retract_ratio = app.add_option("--retract-ratio", retract_ratio, "The ratio by which vacant synapses retract.");

        auto synaptic_elements_init_lb = RelearnTypes::grown_type{ 0.0 };
        app.add_option("--synaptic-elements-lower-bound", synaptic_elements_init_lb, "The minimum number of vacant synaptic elements per neuron. Must be smaller of equal to synaptic-elements-upper-bound.");

        auto synaptic_elements_init_ub = RelearnTypes::grown_type{ 0.0 };
        app.add_option("--synaptic-elements-upper-bound", synaptic_elements_init_ub, "The maximum number of vacant synaptic elements per neuron. Must be larger or equal to synaptic-elements-lower-bound.");

        auto nu_axon = RelearnTypes::grown_type{ SynapticElements::default_nu };
        auto* const opt_nu_axon = app.add_option("--growth-rate-axon", nu_axon, "The growth rate for the axons. Default is 1e-5");

        auto nu_dend_inh = RelearnTypes::grown_type{ SynapticElements::default_nu };
        auto* const opt_nu_dend_inh = app.add_option("--growth-rate-dendrite-inh", nu_dend_inh, "The growth rate for the inhibitory dendrites. Default is 1e-5");

        auto nu_dend_ex = RelearnTypes::grown_type{ SynapticElements::default_nu };
        auto* const opt_nu_dend_ex = app.add_option("--growth-rate-dendrite-exc", nu_dend_ex, "The growth rate for the excitatory dendrites. Default is 1e-5");

        auto* const opt_axons_type = app.add_option("--axons-type", chosen_axons, "The type of axons.");
        opt_axons_type->transform(CLI::CheckedTransformer(cli_parse_axons, CLI::ignore_case));

        auto file_axon_positions = std::string{};
        auto* const opt_file_axon = app.add_option("--file-axon-positions", file_axon_positions, "File with axon positions.");

        auto* const opt_growth_rate_calculator = app.add_option("--growth-rate-calculator", chosen_growth_rate_calculator, "The growth rate calculator.");
        opt_growth_rate_calculator->transform(CLI::CheckedTransformer(cli_parse_growth_rate_calculator, CLI::ignore_case));

        auto* const opt_network_type_calculator = app.add_option("--network-type", chosen_network_type, ".");
        opt_network_type_calculator->transform(CLI::CheckedTransformer(cli_network_type, CLI::ignore_case));

        auto growth_rate_decay = RelearnTypes::grown_type{ 10000.0 };
        auto* const opt_growth_rate_decay = app.add_option("--growth-rate-decay", growth_rate_decay, "The decay of the growth rate. Default is 1e4");

        auto min_calcium_axons = RelearnTypes::calcium_type{ SynapticElements::default_eta_Axons };
        auto* const opt_min_calcium_axons = app.add_option("--min-calcium-axons", min_calcium_axons, "The minimum intercellular calcium for axons to grow. Default is 0.4");

        auto min_calcium_excitatory_dendrites = RelearnTypes::calcium_type{ SynapticElements::default_eta_Dendrites_exc };
        auto* const opt_min_calcium_excitatory_dendrites = app.add_option("--min-calcium-excitatory-dendrites", min_calcium_excitatory_dendrites, "The minimum intercellular calcium for excitatory dendrites to grow. Default is 0.1");

        auto min_calcium_inhibitory_dendrites = RelearnTypes::calcium_type{ SynapticElements::default_eta_Dendrites_inh };
        auto* const opt_min_calcium_inhibitory_dendrites = app.add_option("--min-calcium-inhibitory-dendrites", min_calcium_inhibitory_dendrites, "The minimum intercellular calcium for inhibitory dendrites to grow. Default is 0.0");

        auto synaptic_elements_file = std::filesystem::path{};
        auto* const opt_synaptic_elements_file = app.add_option("--synaptic-elements-file", synaptic_elements_file, "File or directory with synaptic elements parameter.");

        auto neuron_monitors_description = std::string{};
        auto* const monitor_option = app.add_option("--neuron-monitors", neuron_monitors_description,
                                                    "The description which neurons to monitor. Format is <mpi_rank>:<neuron_id>;<mpi_rank>:<neuron_id>;...<group_name>;... where <mpi_rank> can be -1 to indicate \"on every rank\"");

        auto* const flag_monitor_all = app.add_flag("--neuron-monitors-all", "Monitors all neurons.");
        // auto* flag_group_monitor_all = app.add_flag("--group-monitors-all", "Monitors all groups.");

        app.add_option("--flush-step", Config::flush_monitor_step, "The steps when to flush the neuron monitors. Must be > 0");
        app.add_option("--group-monitor-flush-step", Config::flush_group_monitor_step, "The steps when to flush the group monitors. Must be > 0");

        auto percentage_initial_fired_neurons = RelearnTypes::percentage_type{ 0.0 };
        app.add_option("--percentage-initial-fired-neurons", percentage_initial_fired_neurons, "The percentage of neurons that fired in the (imaginary) 0th step. Must be from [0.0, 1.0]. Default ist 0.0");

        auto expected_synapses_per_neuron = CudaConfig::synaptic_count_type{ 100 };
        app.add_option("--expected-synapses-per-neuron", expected_synapses_per_neuron,
                       "Expected number of synapses per neuron, used to size GPU edge storage.");

        auto overflow_chunk_size_factor = CudaConfig::overflow_chunk_size_factor;
        app.add_option("--overflow-chunk-size-factor", overflow_chunk_size_factor, "Size of a single overflow chunk, as a fraction of expected-synapses-per-neuron (default: 0.2)");

        auto number_overflow_chunks_factor = CudaConfig::number_overflow_chunks_factor;
        app.add_option("--number-overflow-chunks-factor", number_overflow_chunks_factor, "Multiplier for the number of overflow chunks pre-allocated in the shared pool (default: 2.0)");

        auto non_overflow_size_factor = CudaConfig::non_overflow_size_factor;
        app.add_option("--non-overflow-size-factor", non_overflow_size_factor, "Fraction of expected synapses per neuron used as the non-overflow (main) chunk size (default: 0.8)");

        auto local_edges_ratio = CudaConfig::local_edges_ratio;
        app.add_option("--local-edges-ratio", local_edges_ratio, "Fraction of max synapses used as main-chunk size for local edges; distant gets the remainder (default: 0.7)")->check(CLI::Range(0.0, 1.0));

        monitor_option->excludes(flag_monitor_all);
        flag_monitor_all->excludes(monitor_option);

        opt_num_neurons->excludes(opt_file_positions);
        opt_num_neurons->excludes(opt_file_network);
        opt_num_neurons->excludes(opt_num_neurons_per_rank);

        opt_num_neurons_per_rank->excludes(opt_num_neurons);
        opt_num_neurons_per_rank->excludes(opt_file_positions);
        opt_num_neurons_per_rank->excludes(opt_file_network);

        opt_file_positions->excludes(opt_num_neurons);
        opt_file_network->excludes(opt_num_neurons);
        opt_file_positions->excludes(opt_num_neurons_per_rank);
        opt_file_network->excludes(opt_num_neurons_per_rank);

        opt_file_network->needs(opt_file_positions);

        opt_file_positions->check(CLI::ExistingPath);
        opt_file_network->check(CLI::ExistingDirectory);

        opt_file_calcium->excludes(opt_initial_calcium);
        opt_file_calcium->excludes(opt_target_calcium);
        opt_initial_calcium->excludes(opt_file_calcium);
        opt_target_calcium->excludes(opt_file_calcium);

        opt_file_axon->check(CLI::ExistingPath);
        if (chosen_axons == AxonsType::MultiPosition) {
            opt_axons_type->needs(opt_file_axon);
        }

        opt_file_calcium->check(CLI::ExistingFile);

        opt_file_enable_interrupts->check(CLI::ExistingFile);
        opt_file_disable_interrupts->check(CLI::ExistingFile);
        opt_file_creation_interrupts->check(CLI::ExistingFile);

        opt_file_external_stimulation->check(CLI::ExistingFile);

        opt_log_path->check(CLI::ExistingDirectory);

        opt_individual_algorithms_for_neurons_file_path->check(CLI::ExistingFile);

        opt_synaptic_elements_file->check(CLI::ExistingFile);
        opt_synaptic_elements_file->excludes(opt_min_calcium_excitatory_dendrites);
        opt_synaptic_elements_file->excludes(opt_min_calcium_axons);
        opt_synaptic_elements_file->excludes(opt_min_calcium_inhibitory_dendrites);
        opt_synaptic_elements_file->excludes(opt_nu_axon);
        opt_synaptic_elements_file->excludes(opt_nu_dend_ex);
        opt_synaptic_elements_file->excludes(opt_nu_dend_inh);
        opt_synaptic_elements_file->excludes(opt_retract_ratio);
        opt_synaptic_elements_file->excludes(opt_growth_rate_decay);

        CLI11_PARSE(app, argc, argv);

#ifdef RELEARN_CUDA_ENABLED
        Config::cuda_aware_mpi_available = static_cast<bool>(*flag_cuda_aware_mpi);

        // Prepare cuda
        cudaDeviceSynchronize_bridge();
        const auto gpu_count = cudaGetDeviceCount_bridge();
        const auto gpu_id = my_rank.get_rank() % gpu_count;
        std::cout << "MPI rank " << my_rank << " uses GPU " << gpu_id << std::endl;
        cudaSetDevice_bridge(gpu_id);
        CudaConfig::local_gpu_id = gpu_id;
        cudaDeviceSynchronize_bridge();
        if (Config::cuda_aware_mpi_available) {
            check_cuda_awareness(my_rank.get_rank(), num_ranks);
        } else {
            std::cerr << "WARN: No cuda-aware mpi available" << std::endl;
        }
        cudaDeviceSynchronize_bridge();
        MPI_Barrier(MPI_COMM_WORLD);
        cudaProfilerStart_bridge();
#endif

        if (static_cast<bool>(*opt_accept_criterion)) {
            RelearnException::check(is_barnes_hut(chosen_algorithm) || chosen_algorithm == AlgorithmEnum::CombinedAlgorithms, "Acceptance criterion can only be set if Barnes-Hut is used");
            RelearnException::check(accept_criterion <= Constants::bh_max_theta, "Acceptance criterion must be smaller or equal to {}", Constants::bh_max_theta);
            RelearnException::check(accept_criterion > RelearnTypes::acceptance_criterion_type{ 0 }, "Acceptance criterion must be larger than 0.0");
        }

        if (static_cast<bool>(*opt_num_neurons)) {
            RelearnException::check(num_ranks == 1, "The option --num-neurons can only be used for one MPI rank. There are {} ranks.", num_ranks);
        }

        RelearnException::check(fraction_excitatory_neurons >= RelearnTypes::percentage_type{ 0 } && fraction_excitatory_neurons <= RelearnTypes::percentage_type{ 1 }, "The fraction of excitatory neurons must be from [0.0, 1.0]");
        RelearnException::check(um_per_neuron > RelearnTypes::space_type{ 0 }, "The micrometer per neuron must be greater than 0.0.");

        RelearnException::check(synaptic_elements_init_lb >= RelearnTypes::grown_type{ 0 }, "The minimum number of vacant synaptic elements must not be negative");
        RelearnException::check(synaptic_elements_init_ub >= synaptic_elements_init_lb, "The minimum number of vacant synaptic elements must not be larger than the maximum number");
        RelearnException::check(static_cast<bool>(*opt_num_neurons) || static_cast<bool>(*opt_file_positions) || static_cast<bool>(*opt_num_neurons_per_rank),
                                "Missing command line option, need a total number of neurons (-n,--num-neurons), a number of neurons per rank (--num-neurons-per-rank), or file_positions (-f,--file).");
        RelearnException::check(openmp_threads > 0, "Number of OpenMP Threads must be greater than 0 (or not set).");
        RelearnException::check(calcium_decay > RelearnTypes::calcium_type{ 0 }, "The calcium decay constant must be greater than 0.");

        RelearnException::check(percentage_initial_fired_neurons >= RelearnTypes::percentage_type{ 0 } && percentage_initial_fired_neurons <= RelearnTypes::percentage_type{ 1 }, "The percentage of neurons that fired in the 0th step must be from [0.0, 1.0]: {}", percentage_initial_fired_neurons);

        if (static_cast<bool>(*opt_target_calcium)) {
            RelearnException::check(target_calcium >= SynapticElements::min_C_target, "Target calcium is smaller than {}", SynapticElements::min_C_target);
            RelearnException::check(target_calcium <= SynapticElements::max_C_target, "Target calcium is larger than {}", SynapticElements::max_C_target);
        }

        if (calcium_decay_type == CalciumCalculatorType::RelativeDecay) {
            RelearnException::check(target_calcium_decay_step > 0, "The target calcium decay step is 0 but must be larger than 0.");
            RelearnException::check(target_calcium_decay_amount < RelearnTypes::calcium_type{ 1 }, "The target calcium decay amount must be smaller than 1.0 for relative decay.");
            RelearnException::check(target_calcium_decay_amount >= RelearnTypes::calcium_type{ 0 }, "The target calcium decay amount must be larger than or equal to 0.0 for relative decay.");
        }

        if (calcium_decay_type == CalciumCalculatorType::AbsoluteDecay) {
            RelearnException::check(target_calcium_decay_step > 0, "The target calcium decay step is 0 but must be larger than 0.");
            RelearnException::check(target_calcium_decay_amount > RelearnTypes::calcium_type{ 0 }, "The target calcium decay amount must be larger than 0.0 for absolute decay.");
        }

        RelearnException::check(nu_axon >= SynapticElements::min_nu, "Growth rate is smaller than {}", SynapticElements::min_nu);
        RelearnException::check(nu_axon <= SynapticElements::max_nu, "Growth rate is larger than {}", SynapticElements::max_nu);
        RelearnException::check(nu_dend_inh >= SynapticElements::min_nu, "Growth rate is smaller than {}", SynapticElements::min_nu);
        RelearnException::check(nu_dend_inh <= SynapticElements::max_nu, "Growth rate is larger than {}", SynapticElements::max_nu);
        RelearnException::check(nu_dend_ex >= SynapticElements::min_nu, "Growth rate is smaller than {}", SynapticElements::min_nu);
        RelearnException::check(nu_dend_ex <= SynapticElements::max_nu, "Growth rate is larger than {}", SynapticElements::max_nu);

        RelearnException::check(Config::flush_monitor_step > 0, "The step for flushing the neuron monitors must be > 0.");
        RelearnException::check(Config::flush_group_monitor_step > 0, "The step for flushing the group monitors must be > 0.");

        omp_set_num_threads(openmp_threads);

        std::size_t current_seed = 0;
        boost::hash_combine(current_seed, my_rank.get_rank());
        boost::hash_combine(current_seed, random_seed);

        Config::random_seed = random_seed;
        CudaConfig::use_pre_drawn_cpu = static_cast<bool>(*flag_pre_drawn_cpu_random_values);
        CudaConfig::expected_synapses_per_neuron = expected_synapses_per_neuron;
        CudaConfig::overflow_chunk_size_factor = overflow_chunk_size_factor;
        CudaConfig::number_overflow_chunks_factor = number_overflow_chunks_factor;
        CudaConfig::local_edges_ratio = local_edges_ratio;
        CudaConfig::non_overflow_size_factor = non_overflow_size_factor;

        RandomHolder::seed_all(current_seed);

        auto init_log_files = [&]() -> void {
            if (static_cast<bool>(*opt_log_path)) {
                LogFiles::set_output_path(log_path);
            }
            if (static_cast<bool>(*opt_log_prefix)) {
                LogFiles::set_general_prefix(log_prefix);
            }

            LogFiles::set_log_status(LogFiles::EventType::Events, !static_cast<bool>(*flag_enable_printing_events));

            LogFiles::set_log_status(LogFiles::EventType::Positions, static_cast<bool>(*flag_disable_printing_positions));

            LogFiles::set_log_status(LogFiles::EventType::Groups, static_cast<bool>(*flag_disable_printing_neuron_to_groups));

            // also disable group to file mapping if group monitoring is disabled
            auto disable_group_to_file_mapping = !static_cast<bool>(*flag_group_monitor) || static_cast<bool>(*flag_disable_printing_group_name_to_file_name);
            LogFiles::set_log_status(LogFiles::EventType::GroupToFileMapping, disable_group_to_file_mapping);

            if (static_cast<bool>(*flag_disable_printing_network)) {
                LogFiles::set_log_status(LogFiles::EventType::InNetwork, true);
                LogFiles::set_log_status(LogFiles::EventType::OutNetwork, true);
                LogFiles::set_log_status(LogFiles::EventType::NetworkInExcitatoryHistogramLocal, true);
                LogFiles::set_log_status(LogFiles::EventType::NetworkInInhibitoryHistogramLocal, true);
                LogFiles::set_log_status(LogFiles::EventType::NetworkOutHistogramLocal, true);
            }

            if (static_cast<bool>(*flag_disable_printing_plasticity)) {
                LogFiles::set_log_status(LogFiles::EventType::PlasticityUpdate, true);
                LogFiles::set_log_status(LogFiles::EventType::PlasticityUpdateCSV, true);
                LogFiles::set_log_status(LogFiles::EventType::PlasticityUpdateLocal, true);
            }

            LogFiles::set_log_status(LogFiles::EventType::CalciumValues, static_cast<bool>(*flag_disable_printing_calcium));
            LogFiles::set_log_status(LogFiles::EventType::ExtremeCalciumValues, static_cast<bool>(*flag_disable_printing_calcium));

            if (static_cast<bool>(*flag_disable_printing_fire_rate)) {
                LogFiles::set_log_status(LogFiles::EventType::FireRates, true);
            }

            if (static_cast<bool>(*flag_disable_printing_fire_steps)) {
                LogFiles::set_log_status(LogFiles::EventType::FireSteps, true);
            }

            if (static_cast<bool>(*flag_disable_printing_overview)) {
                LogFiles::set_log_status(LogFiles::EventType::SynapticInput, true);
                LogFiles::set_log_status(LogFiles::EventType::NeuronsOverview, true);
                LogFiles::set_log_status(LogFiles::EventType::NeuronsOverviewCSV, true);
                Config::statistics_log_step = 99999999;
            }

            if (static_cast<bool>(*flag_disable_printing_group_mapping)) {
                LogFiles::set_log_status(LogFiles::EventType::GroupMapping, true);
            }

            if (static_cast<bool>(*flag_disable_printing_sums)) {
                LogFiles::set_log_status(LogFiles::EventType::Sums, true);
            }

            LogFiles::init();
        };
        init_log_files();

        auto essentials = std::make_unique<Essentials>();

        mpiPP::MPISynchronization::barrier();

        // Rank 0 prints start time of simulation
        if (mpiPP::MPIRank::root_rank() == my_rank) {
            essentials->insert("Start", Timers::wall_clock_time());
            essentials->insert("Number-of-Ranks", num_ranks);
            essentials->insert("Number-of-Steps", simulation_steps);
            essentials->insert("Initial-Elements-Lower-Bound", synaptic_elements_init_lb);
            essentials->insert("Initial-Elements-Upper-Bound", synaptic_elements_init_ub);
            essentials->insert("Calcium-Target", target_calcium);
            essentials->insert("Beta", beta);
            essentials->insert("Calcium-Decay", calcium_decay);
            essentials->insert("Nu-Axons", nu_axon);
            essentials->insert("Nu-Dendrites inh", nu_dend_inh);
            essentials->insert("Nu-Dendrites ex", nu_dend_ex);
            essentials->insert("Retract-Ratio", retract_ratio);
            essentials->insert("Synapse-Conductance", synapse_conductance);
            essentials->insert("Background-Base", base_background_activity);
            essentials->insert("Background-Mean", background_activity_mean);
            essentials->insert("Background-Stddev", background_activity_stddev);

            essentials->insert("Log-path", log_path.string());
            essentials->insert("Algorithm", stringify(chosen_algorithm));
            essentials->insert("Neuron-model", stringify(chosen_neuron_model));
            essentials->insert("First-plasticity-step", first_plasticity_step);
            essentials->insert("Last-plasticity-step", last_plasticity_step);

            essentials->insert("Calcium-Minimum-Axons", min_calcium_axons);
            essentials->insert("Calcium-Minimum-Excitatory-Dendrites", min_calcium_excitatory_dendrites);
            essentials->insert("Calcium-Minimum-Inhibitory-Dendrites", min_calcium_inhibitory_dendrites);

            if (chosen_synapse_input_calculator_type == SynapticInputCalculatorType::Logarithmic) {
                essentials->insert("Synapse-Input", "Logarithmic");
                essentials->insert("Synapse-Input-Scaling", input_scale);
            } else if (chosen_synapse_input_calculator_type == SynapticInputCalculatorType::Linear) {
                essentials->insert("Synapse-Input", "Linear");
            } else if (chosen_synapse_input_calculator_type == SynapticInputCalculatorType::HyperbolicTangent) {
                essentials->insert("Synapse-Input", "Hyperbolic-Tangent");
                essentials->insert("Synapse-Input-Scaling", input_scale);
            }

            if (chosen_kernel_type == KernelType::Gamma) {
                essentials->insert("Kernel-Type", "Gamma");
                essentials->insert("Kernel-Shape-Parameter", gamma_k);
                essentials->insert("Kernel-Scale-Parameter", gamma_theta);
            } else if (chosen_kernel_type == KernelType::Gaussian) {
                essentials->insert("Kernel-Type", "Gaussian");
                essentials->insert("Kernel-Translation-Parameter", gaussian_mu);
                essentials->insert("Kernel-Scale-Parameter", gaussian_sigma);
            } else if (chosen_kernel_type == KernelType::Linear) {
                essentials->insert("Kernel-Type", "Linear");
                essentials->insert("Kernel-Cut-off-Parameter", linear_cutoff);
            } else if (chosen_kernel_type == KernelType::Weibull) {
                essentials->insert("Kernel-Type", "Weibull");
                essentials->insert("Kernel-Shape-Parameter", weibull_k);
                essentials->insert("Kernel-Scale-Parameter", weibull_b);
            }

            if (static_cast<bool>(*opt_num_neurons)) {
                essentials->insert("number neurons", number_neurons);
                essentials->insert("Fraction-excitatory-neurons", fraction_excitatory_neurons);
                essentials->insert("um-per-neuron", um_per_neuron);
            } else if (static_cast<bool>(*opt_num_neurons_per_rank)) {
                essentials->insert("number neurons per rank", number_neurons_per_rank);
                essentials->insert("Fraction-excitatory-neurons", fraction_excitatory_neurons);
                essentials->insert("um-per-neuron", um_per_neuron);
            } else {
                essentials->insert("positions directory", file_positions.string());

                essentials->insert("network directory",
                                   file_network.string());
            }
            essentials->insert("external stimulation file", file_external_stimulation.string());
            essentials->insert("static neurons",
                               static_neurons_str);
        }

        LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdate, false, "#step: creations deletions net");
        LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateCSV, false, "#step;creations;deletions;net");
        LogFiles::write_to_file(LogFiles::EventType::PlasticityUpdateLocal, false, "#step: creations deletions net");

        Timers::start(TimerRegion::INITIALIZATION);

        const auto prepare_algorithm = [&]() -> std::unique_ptr<KernelBase> {
            if (is_fast_multipole_method(chosen_algorithm)) {
                RelearnException::check(chosen_kernel_type == KernelType::Gaussian, "Setting the probability kernel type is not supported for the fast multipole methods!");
            }

            switch (chosen_kernel_type) {
            case KernelType::Gamma:
                return std::make_unique<GammaDistributionKernel>(gamma_k, gamma_theta);
            case KernelType::Gaussian:
                return std::make_unique<GaussianDistributionKernel>(gaussian_mu, gaussian_sigma);
            case KernelType::Linear:
                return std::make_unique<LinearDistributionKernel>(linear_cutoff);
            case KernelType::Weibull:
                return std::make_unique<WeibullDistributionKernel>(weibull_k, weibull_b);
            default:
                RelearnException::fail("The kernel type {} is not supported", chosen_kernel_type);
            }
        };
        auto kernel = prepare_algorithm();

        auto partition = std::make_shared<Partition>(num_ranks, my_rank, chosen_space_filling_curve_type);

        auto construct_subdomain = [&]() -> std::unique_ptr<NeuronToSubdomainAssignment> {
            if (static_cast<bool>(*opt_num_neurons)) {
                return std::make_unique<SubdomainFromNeuronDensity>(number_neurons, fraction_excitatory_neurons, um_per_neuron, partition);
            }

            if (static_cast<bool>(*opt_num_neurons_per_rank)) {
                return std::make_unique<SubdomainFromNeuronPerRank>(number_neurons_per_rank, fraction_excitatory_neurons, um_per_neuron, partition);
            }

            auto path_to_network = std::optional<std::filesystem::path>{};
            if (static_cast<bool>(*opt_file_network)) {
                path_to_network = file_network;
            }

            return std::make_unique<MultipleSubdomainsFromFile>(file_positions, std::move(path_to_network), partition);
        };
        auto subdomain = construct_subdomain();

        subdomain->initialize_groups_and_local_group_translator(static_cast<bool>(*opt_file_groups) ? std::make_optional(file_groups) : std::nullopt);

        auto construct_fired_status_communicator = [&]() -> std::shared_ptr<FiredStatusCommunicator> {
            if (chosen_fired_status_communicator_type == FiredStatusCommunicatorType::Map) {
                return std::make_shared<FiredStatusCommunicationMap>(my_rank, mpiPP::MPIInfo::get_number_ranks());
            }

            if (chosen_fired_status_communicator_type == FiredStatusCommunicatorType::GpuUncompressed) {
                return std::make_shared<FireStatusCommunicatorGPUUncompressed>(my_rank, mpiPP::MPIInfo::get_number_ranks());
            }

            RelearnException::check(chosen_fired_status_communicator_type == FiredStatusCommunicatorType::Approximator, "Type {} of fired status communicator is not supported.", chosen_fired_status_communicator_type);
            return std::make_shared<FiredStatusApproximator>(my_rank, mpiPP::MPIInfo::get_number_ranks());
        };
        auto fired_status_comm = construct_fired_status_communicator();

        auto parse_context = ActivityInputParseContext{
            .background_base = utility::cast<double>(base_background_activity),
            .background_mean = utility::cast<double>(background_activity_mean),
            .background_stddev = utility::cast<double>(background_activity_stddev),

            .communicator = fired_status_comm,
            .synapse_conductance = utility::cast<double>(synapse_conductance),
#ifndef RELEARN_CUDA_ENABLED
            .linear = [synapse_conductance](const RelearnTypes::activity_type val) { return synapse_conductance * val; },
            .logarithmic = [input_scale, synapse_conductance](const RelearnTypes::activity_type val) { return input_scale * std::log10(synapse_conductance * val + RelearnTypes::activity_type{ 1 }); },
            .hyperbolic_tangent = [input_scale, synapse_conductance](const RelearnTypes::activity_type val) { return input_scale * std::tanh(synapse_conductance * val); },
#else
            .linear = CudaConfig::LINEAR,
            .logarithmic = CudaConfig::LOGARITHMIC,
            .hyperbolic_tangent = CudaConfig::HYPERBOLIC_TANGENT,
            .input_scale = input_scale,
#endif
            .load_stimulus =
                [&file_external_stimulation, my_rank, &subdomain, opt_file_external_stimulation]() {
                    if (!*opt_file_external_stimulation) {
                        RelearnException::fail("Tried to use stimulus interrupts, but no file that specified the stimulation was passed.");
                    }
                    return InteractiveNeuronIO::load_stimulus_interrupts(file_external_stimulation, my_rank, subdomain->get_local_group_translator());
                },

            .flexible_background = [opt_flexible_background_file_path, flexible_background_file_path, my_rank, &subdomain]() {
            if(*opt_flexible_background_file_path) {
                return BackgroundActivityIO::load_background_activity(flexible_background_file_path, my_rank, subdomain->get_local_group_translator());

            }
            RelearnException::fail("FlexibleBackgroundActivity was requested but no file was provided"); }
        };

        auto combined_activity_input = parse_activity(activity_input_str, parse_context);

        auto construct_neuron_model = [&]() -> std::unique_ptr<NeuronModel> {
            using namespace models;

            if (chosen_neuron_model == NeuronModelType::Poisson) {
                return std::make_unique<PoissonModel>(h, std::move(combined_activity_input), std::move(fired_status_comm),
                                                      poisson::Parameters<RelearnTypes::activity_type, unsigned int>{});
            }

            if (chosen_neuron_model == NeuronModelType::Izhikevich) {
                return std::make_unique<IzhikevichModel>(h, std::move(combined_activity_input), std::move(fired_status_comm),
                                                         izhikevich::Parameters<RelearnTypes::activity_type>{});
            }

            if (chosen_neuron_model == NeuronModelType::FitzHughNagumo) {
                return std::make_unique<FitzHughNagumoModel>(h, std::move(combined_activity_input), std::move(fired_status_comm),
                                                             fitzhughnagumo::Parameters<RelearnTypes::activity_type>{});
            }

            RelearnException::check(chosen_neuron_model == NeuronModelType::AEIF, "Chose a neuron model that is not implemented");
            return std::make_unique<AEIFModel>(h, std::move(combined_activity_input), std::move(fired_status_comm),
                                               aeif::Parameters<RelearnTypes::activity_type>{});
        };
        auto neuron_model = construct_neuron_model();

        auto construct_calcium_calculator = [&]() -> std::unique_ptr<CalciumCalculator> {
            auto calcium_calculator = std::unique_ptr<CalciumCalculator>{};

            if (calcium_decay_type == CalciumCalculatorType::Normal) {
                calcium_calculator = std::make_unique<CalciumCalculator>();
            } else if (calcium_decay_type == CalciumCalculatorType::AbsoluteDecay) {
                const auto decay_interval = utility::Interval<RelearnTypes::step_type>{ .begin = first_decay_step, .end = last_decay_step, .frequency = target_calcium_decay_step };
                calcium_calculator = std::make_unique<AbsoluteDecayCalciumCalculator>(target_calcium_decay_amount, decay_interval);
            } else if (calcium_decay_type == CalciumCalculatorType::RelativeDecay) {
                const auto decay_interval = utility::Interval<RelearnTypes::step_type>{ .begin = first_decay_step, .end = last_decay_step, .frequency = target_calcium_decay_step };
                calcium_calculator = std::make_unique<RelativeDecayCalciumCalculator>(target_calcium_decay_amount, decay_interval);
            } else {
                RelearnException::fail("CalciumCalculatorType {} not implemented", calcium_decay_type);
            }

            calcium_calculator->set_beta(beta);
            calcium_calculator->set_tau_C(calcium_decay);
            calcium_calculator->set_h(h);

            if (*opt_file_calcium) {
                auto [initial_calcium_calculator, target_calcium_calculator] = CalciumIO::load_initial_and_target_function(file_calcium, subdomain->get_local_group_translator(), my_rank);

                calcium_calculator->set_initial_calcium_calculator(std::move(initial_calcium_calculator));
                calcium_calculator->set_target_calcium_calculator(std::move(target_calcium_calculator));
            } else {
                auto initial_calcium_calculator = [initial = initial_calcium](mpiPP::MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return initial; };
                calcium_calculator->set_initial_calcium_calculator(std::move(initial_calcium_calculator));

                auto target_calcium_calculator = [target = target_calcium](mpiPP::MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return target; };
                calcium_calculator->set_target_calcium_calculator(std::move(target_calcium_calculator));
            }

            return calcium_calculator;
        };
        auto calcium_calculator = construct_calcium_calculator();

        auto construct_growth_rate_calculator = [&](const RelearnTypes::grown_type growth_rate) -> std::shared_ptr<GrowthrateCalculator> {
            if (chosen_growth_rate_calculator == GrowthrateCalculatorType::Constant) {
                return std::make_shared<ConstantGrowthrateCalculator>(growth_rate);
            }
            RelearnException::fail("Other growthrate calculators are not implemented: {}", chosen_growth_rate_calculator);
        };

        auto construct_synaptic_elements = [&] {
            auto construct_axons = [&]() -> std::shared_ptr<Axons> {
                if (chosen_axons == AxonsType::Normal) {
                    return std::make_shared<Axons>();
                }

                if (chosen_axons == AxonsType::MultiPosition) {
                    auto positions = AxonPositionIO::read_axon_positions(file_axon_positions);
                    auto axons = std::make_shared<MultiPositionAxons>();
                    axons->set_bouton_positions(positions);

                    return axons;
                }

                return nullptr;
            };

            auto axons = construct_axons();
            auto dendrites = std::make_shared<Dendrites>();

            auto synaptic_elements = std::make_shared<SynapticElements>(std::move(axons), std::move(dendrites));

            if (*opt_synaptic_elements_file) {
                const auto& [axons_parameters, den_exc_parameters, den_inh_parameters]
                    = SynapticElementsIO::load_function_from_file(synaptic_elements_file, my_rank, subdomain->get_local_group_translator());

                synaptic_elements->set_minimum_calcium_calculator(axons_parameters.min_calcium, SynapticElementType::Axon);
                synaptic_elements->set_minimum_calcium_calculator(den_exc_parameters.min_calcium, SynapticElementType::DendriteExcitatory);
                synaptic_elements->set_minimum_calcium_calculator(den_inh_parameters.min_calcium, SynapticElementType::DendriteInhibitory);

                synaptic_elements->set_vacant_retract_ratio_calculator(axons_parameters.vacant_retract_ratio, SynapticElementType::Axon);
                synaptic_elements->set_vacant_retract_ratio_calculator(den_exc_parameters.vacant_retract_ratio, SynapticElementType::DendriteExcitatory);
                synaptic_elements->set_vacant_retract_ratio_calculator(den_inh_parameters.vacant_retract_ratio, SynapticElementType::DendriteInhibitory);

                return synaptic_elements;
            }

            auto construct_const_calculator = [](const auto constant) {
                return [constant](const RelearnTypes::number_neurons_type /*neuron_id*/) {
                    return constant;
                };
            };

            auto construct_grown_element_calculator = [synaptic_elements_init_lb, synaptic_elements_init_ub](const RelearnTypes::number_neurons_type /*neuron_id*/) {
                if (synaptic_elements_init_lb == synaptic_elements_init_ub) {
                    return synaptic_elements_init_lb;
                }

                return RandomHolder::get_random_uniform_double<RelearnTypes::grown_type>(RandomHolderKey::SynapticElements, synaptic_elements_init_lb, synaptic_elements_init_ub);
            };

            synaptic_elements->set_growthrate_calculator(construct_growth_rate_calculator(nu_axon), SynapticElementType::Axon);
            synaptic_elements->set_growthrate_calculator(construct_growth_rate_calculator(nu_dend_ex), SynapticElementType::DendriteExcitatory);
            synaptic_elements->set_growthrate_calculator(construct_growth_rate_calculator(nu_dend_inh), SynapticElementType::DendriteInhibitory);

            synaptic_elements->set_grown_elements_calculator(construct_grown_element_calculator, SynapticElementType::Axon);
            synaptic_elements->set_grown_elements_calculator(construct_grown_element_calculator, SynapticElementType::DendriteExcitatory);
            synaptic_elements->set_grown_elements_calculator(construct_grown_element_calculator, SynapticElementType::DendriteInhibitory);

            synaptic_elements->set_vacant_retract_ratio_calculator(construct_const_calculator(retract_ratio), SynapticElementType::Axon);
            synaptic_elements->set_vacant_retract_ratio_calculator(construct_const_calculator(retract_ratio), SynapticElementType::DendriteExcitatory);
            synaptic_elements->set_vacant_retract_ratio_calculator(construct_const_calculator(retract_ratio), SynapticElementType::DendriteInhibitory);

            synaptic_elements->set_minimum_calcium_calculator(construct_const_calculator(min_calcium_axons), SynapticElementType::Axon);
            synaptic_elements->set_minimum_calcium_calculator(construct_const_calculator(min_calcium_excitatory_dendrites), SynapticElementType::DendriteExcitatory);
            synaptic_elements->set_minimum_calcium_calculator(construct_const_calculator(min_calcium_inhibitory_dendrites), SynapticElementType::DendriteInhibitory);

            return synaptic_elements;
        };

        auto construct_synapse_deletion_finder = [&]() -> std::unique_ptr<SynapseDeletionFinder> {
            if (chosen_synapse_deleter == SynapseDeletionFinderType::Random) {
                return std::make_unique<RandomSynapseDeletionFinder>();
            }

            RelearnException::check(chosen_synapse_deleter == SynapseDeletionFinderType::InverseLength, "Type {} of synapse deleter is not supported.", chosen_synapse_deleter);
            return std::make_unique<InverseLengthSynapseDeletionFinder>();
        };
        auto synapse_deletion_finder = construct_synapse_deletion_finder();

        auto synaptic_elements = construct_synaptic_elements();

        auto sim = Simulation(std::move(essentials), partition);
        sim.set_network_type(chosen_network_type);
        sim.set_neuron_model(std::move(neuron_model));
        sim.set_calcium_calculator(std::move(calcium_calculator));
        sim.set_synaptic_elements(std::move(synaptic_elements));
        sim.set_synapse_deletion_finder(std::move(synapse_deletion_finder));
        sim.set_probability_kernel(std::move(kernel));

        sim.set_percentage_initial_fired_neurons(percentage_initial_fired_neurons);

        if (*opt_static_neurons) {
            auto static_neurons = MonitorParser::parse_my_ids(static_neurons_str, my_rank, subdomain->get_local_group_translator());
            sim.set_static_neurons(static_neurons);
        }

        if (is_barnes_hut(chosen_algorithm)) {
            sim.set_acceptance_criterion_for_barnes_hut(accept_criterion);
        }

        if (chosen_algorithm == AlgorithmEnum::CombinedAlgorithms) {
            sim.set_acceptance_criterion_for_barnes_hut(accept_criterion);
            auto pair = NeuronToAlgorithmIO::read_descriptions(individual_algorithms_for_neurons_file_path, my_rank, subdomain->get_local_group_translator());
            auto& indices_and_neurons = pair.first;
            auto algorithm_configs = std::move(pair.second);
            sim.set_indices_and_neurons_for_combined_algorithms(indices_and_neurons);
            sim.set_algorithm_vector_for_combined_algorithms(std::move(algorithm_configs));
        }

#ifdef RELEARN_CUDA_ENABLED
        RelearnException::check(chosen_algorithm == AlgorithmEnum::BarnesHutCuda || chosen_algorithm == AlgorithmEnum::NaiveCuda,
                                "Cuda only supports Barnes-Hut and Naive");
#endif

        sim.set_algorithm(chosen_algorithm);
        sim.set_subdomain_assignment(std::move(subdomain));

        if (*opt_file_enable_interrupts) {
            auto enable_interrupts = InteractiveNeuronIO::load_enable_interrupts(file_enable_interrupts, my_rank);
            sim.set_enable_interrupts(std::move(enable_interrupts));
        }

        if (*opt_file_disable_interrupts) {
            auto disable_interrupts = InteractiveNeuronIO::load_disable_interrupts(file_disable_interrupts, my_rank);
            sim.set_disable_interrupts(std::move(disable_interrupts));
        }

        if (*opt_file_creation_interrupts) {
            auto creation_interrupts = InteractiveNeuronIO::load_creation_interrupts(file_creation_interrupts);
            sim.set_creation_interrupts(std::move(creation_interrupts));
        }

        RelearnException::check(Config::plasticity_update_step > 0, "update-plasticity-step must be greater than 0");

        sim.set_update_plasticity_interval(utility::Interval<RelearnTypes::step_type>{ .begin = first_plasticity_step, .end = last_plasticity_step, .frequency = Config::plasticity_update_step });
        sim.set_update_synaptic_elements_interval(utility::Interval<RelearnTypes::step_type>{ .begin = first_plasticity_step, .end = last_plasticity_step, .frequency = 1 });
        sim.set_log_calcium_interval(utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::calcium_log_step });
        sim.set_log_fire_rate_interval(utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::fire_rate_log_step });
        sim.set_log_synaptic_input_interval(utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::synaptic_input_log_step });
        sim.set_log_network_interval(utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::network_log_step });
        sim.set_update_neuron_monitor_interval(utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::neuron_monitor_log_step });
        sim.enable_group_monitor(static_cast<bool>(*flag_group_monitor), static_cast<bool>(*flag_group_monitor_connectivity));
        sim.set_update_group_monitor_interval(utility::Interval<RelearnTypes::step_type>{ .begin = 0, .end = std::numeric_limits<RelearnTypes::step_type>::max(), .frequency = Config::group_monitor_log_step });

        /**********************************************************************************/

        // The barrier ensures that every rank finished its local stores.
        // Otherwise, a "fast" rank might try to read from the RMA window of another
        // rank which has not finished (or even begun) its local stores

        mpiPP::MPISynchronization::barrier(); // TODO(future) Really needed?
        sim.initialize();
        Timers::stop_and_add(TimerRegion::INITIALIZATION);

        if (static_cast<bool>(*flag_monitor_all)) {
            const auto number_local_neurons = sim.get_partition()->get_number_local_neurons();
            for (const auto& neuron_id : NeuronIDRange::range(number_local_neurons)) {
                sim.register_neuron_monitor(neuron_id);
            }
        } else {
            const auto& my_neuron_ids_to_monitor = MonitorParser::parse_my_ids(neuron_monitors_description, my_rank, sim.get_neurons()->get_local_group_translator());
            for (const auto& neuron_id : my_neuron_ids_to_monitor) {
                sim.register_neuron_monitor(neuron_id);
            }
        }

        simulate(std::move(sim), simulation_steps, static_cast<bool>(*flag_interactive));

#ifdef RELEARN_CUDA_ENABLED
        // Frees device state held by function-static/namespace-static globals (RandomNumbers'
        // cuRAND states and config array, the synaptic-activity lookup set) here, while the CUDA
        // context is still definitely alive. Otherwise they are only freed by static destructors at
        // process exit, whose order relative to the CUDA driver's own atexit-registered shutdown is
        // unspecified -- if the driver tears down first, freeing device memory afterward aborts the
        // process ("driver shutting down").
        release_synaptic_activity_set();
        RandomNumbers::reset();

        cudaProfilerStop_bridge();
#endif
        mpiPP::MPIWrapper::finalize();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "relearn: fatal error: " << e.what() << '\n';
        // A single rank failing must not leave the others hanging in a collective/barrier; take the whole job down.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
