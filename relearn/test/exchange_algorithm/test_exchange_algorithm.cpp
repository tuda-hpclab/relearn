/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#ifdef RELEARN_CUDA_ENABLED

#include "RelearnTest.hpp"

#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/network_graph/NetworkGPUType.h"
#include "cuda/spikes/ExchangeAlgorithm.h"
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "neurons/NetworkGraph.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "neurons/synaptic_elements/Dendrites.h"

#include "factory/extra_info/extra_info_factory.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <unordered_map>
#include <vector>

class ExchangeAlgorithmTest : public RelearnTest { };

// Regression test for a fairness bug in process_requests_entry_aware(): when a target neuron has
// fewer vacant dendrites than incoming requests, contention used to be resolved by an explicit
// ascending-source-id tiebreak inside partition_after_target()'s sort comparator -- meaning the
// smallest-numbered contending source neuron would win every single time, deterministically,
// regardless of seed/step. The fix replaces that tiebreak with a seed/step-derived shuffle. This
// test sets up many source neurons competing for a single target's one vacant dendrite and checks,
// across many independently-seeded rounds, that the winner is not always the same source.
TEST_F(ExchangeAlgorithmTest, testDendriteContentionDoesNotAlwaysFavorSmallestSourceId) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    constexpr auto number_contenders = CudaConfig::number_neurons_type{ 50 };
    constexpr auto target_neuron_id = CudaConfig::number_neurons_type{ 0 };
    // Contenders are neurons [1, number_contenders], so none of them ever equals target_neuron_id
    // (which would otherwise be rejected as a self-loop rather than compete for the dendrite).
    constexpr auto number_neurons = number_contenders + 1;
    constexpr auto number_trials = 60;
    constexpr auto my_rank = CudaConfig::mpi_rank_type{ 0 };

    auto network_graph = std::make_shared<NetworkGraph>(mpiPP::MPIRank{ my_rank }, /*number_ranks=*/1,
                                                        /*expected_synapses_per_neuron=*/number_trials + 10);
    network_graph->init(number_neurons, NetworkGPUType::MEMORY_POOL);

    auto h_source_ids = std::vector<CudaConfig::number_neurons_type>(number_contenders);
    auto h_target_ids = std::vector<CudaConfig::number_neurons_type>(number_contenders, target_neuron_id);
    auto h_source_ranks = std::vector<CudaConfig::mpi_rank_type>(number_contenders, my_rank);
    for (auto i = CudaConfig::number_neurons_type{ 0 }; i < number_contenders; ++i) {
        h_source_ids[i] = i + 1;
    }

    auto d_source_ids = DeviceArray<CudaConfig::number_neurons_type>(std::span<const CudaConfig::number_neurons_type>(h_source_ids));
    auto d_target_ids = DeviceArray<CudaConfig::number_neurons_type>(std::span<const CudaConfig::number_neurons_type>(h_target_ids));
    auto d_source_ranks = DeviceArray<CudaConfig::mpi_rank_type>(std::span<const CudaConfig::mpi_rank_type>(h_source_ranks));

    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    const auto info_handle = extra_infos->get_gpu_handle();

    // Inhibitory dendrites are never dereferenced for an Excitatory request (see
    // process_requests_kernel_aware), but the handle object itself must still be valid.
    auto inhibitory_dendrites = Dendrites{};
    inhibitory_dendrites.init(number_neurons);

    auto win_counts = std::unordered_map<CudaConfig::number_neurons_type, int>{};

    for (auto trial = 0; trial < number_trials; ++trial) {
        // Exactly one vacant excitatory dendrite on the target neuron, none anywhere else --
        // every trial has exactly number_contenders requests competing for that single slot.
        auto excitatory_dendrites = Dendrites{};
        excitatory_dendrites.set_grown_elements_calculator(
            [](const RelearnTypes::number_neurons_type i) -> double { return i == target_neuron_id ? 1.0 : 0.0; },
            SignalType::Excitatory);
        excitatory_dendrites.init(number_neurons);

        auto h_responses = std::vector<SynapseCreationResponse>(number_contenders, SynapseCreationResponse::Failed);
        auto d_responses = DeviceArray<SynapseCreationResponse>(std::span<const SynapseCreationResponse>(h_responses));

        const auto created = process_requests_entry_aware(
            SynapseCreationRequestHandle{ d_source_ids.device_ptr(), d_source_ranks.device_ptr(), d_target_ids.device_ptr(), SignalType::Excitatory, number_contenders },
            d_responses.device_ptr(),
            excitatory_dendrites.get_cuda_handle(SignalType::Excitatory), inhibitory_dendrites.get_cuda_handle(SignalType::Inhibitory),
            info_handle, network_graph->get_gpu_handle(), /*seed=*/12345, /*step=*/static_cast<std::uint64_t>(trial));

        ASSERT_EQ(created, 1UL) << "exactly one request must succeed: there is only one vacant dendrite";

        const auto responses = d_responses.get_device_data();
        auto number_succeeded = 0;
        for (auto i = CudaConfig::number_neurons_type{ 0 }; i < number_contenders; ++i) {
            if (responses[i] == SynapseCreationResponse::Succeeded) {
                ++number_succeeded;
                win_counts[h_source_ids[i]]++;
            }
        }
        ASSERT_EQ(number_succeeded, 1) << "exactly one response must be Succeeded";
    }

    const auto number_distinct_winners = win_counts.size();
    ASSERT_GT(number_distinct_winners, 1U) << "the same source neuron won every single round -- contention is still resolved deterministically";
    ASSERT_GE(number_distinct_winners, 10U) << "only " << number_distinct_winners << " distinct winners out of " << number_contenders
                                            << " contenders across " << number_trials << " trials -- too concentrated for a fair, random tiebreak";

    // The smallest-numbered contender (source id 1) must not win a disproportionate share -- the
    // original bug had it win 100% of the time.
    const auto smallest_id_wins = win_counts.count(1) ? win_counts.at(1) : 0;
    ASSERT_LT(smallest_id_wins, number_trials) << "source id 1 won every round";
    ASSERT_LT(smallest_id_wins, number_trials / 2) << "source id 1 won a disproportionate share of rounds (" << smallest_id_wins << "/" << number_trials << ")";
}

#endif
