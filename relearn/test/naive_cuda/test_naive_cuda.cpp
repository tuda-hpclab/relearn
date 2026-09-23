/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_naive_cuda.h"

#ifdef RELEARN_CUDA_ENABLED

#include "algorithm/NaiveInternalCUDA/NaiveCUDA.h"
#include "cuda/CudaTypes.h"
#include "cuda/algorithm/NaiveInternalCUDA/NaiveCUDA_CU.h"
#include "cuda/memory/DeviceArray.h"
#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "structure/Morton.h"
#include "types/BasicTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"
#include "util/Random.h"

#include "factory/kernel/kernel_factory.h"
#include "factory/network_graph/network_graph_factory.h"
#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"
#include "factory/synaptic_elements/synaptic_elements_factory.h"

#include <cpp-utility/Cast.hpp>

#include <mpi-wrapper/core/MPIInfo.h>
#include <mpi-wrapper/core/MPIRank.h>

#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

namespace {

SimpleVec3d to_simple_vec3d(const RelearnTypes::position_type& position) {
    return SimpleVec3d{ utility::cast<double>(position.get_x()), utility::cast<double>(position.get_y()), utility::cast<double>(position.get_z()) };
}

constexpr auto squared_sigma_inv = 1.0;

} // namespace

// Exercises all validation paths that should throw before ever touching the GPU.
TEST_F(NaiveCUDATest, testFindTargetNeuronsException) {
    constexpr auto number_neurons = 5UL;

    auto valid_positions = std::vector<SimpleVec3d>(number_neurons, SimpleVec3d{ 0.0, 0.0, 0.0 });
    auto valid_vacant_dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 1U);
    auto valid_mapping = std::vector<RelearnTypes::counter_type>(number_neurons, 0U);
    auto valid_target_array = std::vector<std::uint64_t>(number_neurons);
    auto valid_random_nums = std::vector<double>(number_neurons);
    RandomHolder::fill(RandomHolderKey::Algorithm, std::begin(valid_random_nums), std::end(valid_random_nums), 0.0, 1.0);

    auto too_few_vacant_dendrites = std::vector<RelearnTypes::counter_type>(3, 1U);
    auto too_few_mapping = std::vector<RelearnTypes::counter_type>(3, 0U);
    auto too_few_target_array = std::vector<std::uint64_t>(3);
    auto too_few_random_nums = std::vector<double>(3, 0.5);

    auto too_many_vacant_dendrites = std::vector<RelearnTypes::counter_type>(7, 1U);
    auto too_many_mapping = std::vector<RelearnTypes::counter_type>(7, 0U);
    auto too_many_target_array = std::vector<std::uint64_t>(7);

    auto d_pos = DeviceArray<SimpleVec3d>(std::span<const SimpleVec3d>(valid_positions));

    // mapping (== target_size) too large/small
    ASSERT_THROW_NO_PRINT(NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ too_many_mapping, valid_vacant_dendrites, valid_random_nums, number_neurons }, valid_target_array, squared_sigma_inv), std::runtime_error);
    ASSERT_THROW_NO_PRINT(NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ too_few_mapping, valid_vacant_dendrites, valid_random_nums, number_neurons }, valid_target_array, squared_sigma_inv), std::runtime_error);
    // vacant dendrites array must equal neurons_count
    ASSERT_THROW_NO_PRINT(NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ valid_mapping, too_many_vacant_dendrites, valid_random_nums, number_neurons }, valid_target_array, squared_sigma_inv), std::runtime_error);
    ASSERT_THROW_NO_PRINT(NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ valid_mapping, too_few_vacant_dendrites, valid_random_nums, number_neurons }, valid_target_array, squared_sigma_inv), std::runtime_error);
    // target array must equal target_size
    ASSERT_THROW_NO_PRINT(NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ valid_mapping, valid_vacant_dendrites, valid_random_nums, number_neurons }, too_many_target_array, squared_sigma_inv), std::runtime_error);
    ASSERT_THROW_NO_PRINT(NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ valid_mapping, valid_vacant_dendrites, valid_random_nums, number_neurons }, too_few_target_array, squared_sigma_inv), std::runtime_error);
    // random numbers array must be >= target_size
    ASSERT_THROW_NO_PRINT(NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ valid_mapping, valid_vacant_dendrites, too_few_random_nums, number_neurons }, valid_target_array, squared_sigma_inv), std::runtime_error);

    // null device pointer (a moved-from DeviceArray never re-allocates, so device_ptr() stays nullptr)
    auto d_pos_to_move = DeviceArray<SimpleVec3d>(std::span<const SimpleVec3d>(valid_positions));
    auto d_pos_moved_to = std::move(d_pos_to_move);
    ASSERT_THROW_NO_PRINT(NaiveCUDA_CU::find_target_neurons(d_pos_to_move, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ valid_mapping, valid_vacant_dendrites, valid_random_nums, number_neurons }, valid_target_array, squared_sigma_inv), std::runtime_error);
}

// If nobody in the whole population has a vacant dendrite, every task must fall back to
// its own source neuron (an "autapse", to be filtered out by the caller) rather than crash
// or pick an arbitrary neuron.
TEST_F(NaiveCUDATest, testFindTargetNeuronsNoDendritesAnywhereFallsBackToAutapse) {
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 1;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto positions = std::vector<SimpleVec3d>(number_neurons);
    for (auto i = 0UL; i < number_neurons; ++i) {
        positions[i] = to_simple_vec3d(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));
    }

    // Every neuron has exactly one task, mapping task i -> source neuron i, no vacant dendrites at all.
    auto vacant_dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 0U);
    auto mapping = std::vector<RelearnTypes::counter_type>(number_neurons);
    for (auto i = 0UL; i < number_neurons; ++i) {
        mapping[i] = static_cast<RelearnTypes::counter_type>(i);
    }

    auto random_nums = std::vector<double>(number_neurons);
    RandomHolder::fill(RandomHolderKey::Algorithm, std::begin(random_nums), std::end(random_nums), 0.0, 1.0);

    auto target_array = std::vector<std::uint64_t>(number_neurons, std::numeric_limits<std::uint64_t>::max());

    auto d_pos = DeviceArray<SimpleVec3d>(std::span<const SimpleVec3d>(positions));
    NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ mapping, vacant_dendrites, random_nums, number_neurons }, target_array, squared_sigma_inv);

    for (auto i = 0UL; i < number_neurons; ++i) {
        ASSERT_EQ(target_array[i], i);
    }
}

// With exactly one neuron holding all vacant dendrites, every source task must be routed to
// that single sink, regardless of the drawn random number -- this is deterministic because
// only one candidate contributes a non-zero connection probability.
TEST_F(NaiveCUDATest, testFindTargetNeuronsSingleSinkAlwaysChosen) {
    const auto number_neurons = std::max<RelearnTypes::number_neurons_type>(NeuronIdFactory::get_random_number_neurons(mt), 2);
    const auto sink_index = RandomFactory::get_random_integer<RelearnTypes::number_neurons_type>(0, number_neurons - 1, mt);

    auto positions = std::vector<SimpleVec3d>(number_neurons);
    for (auto i = 0UL; i < number_neurons; ++i) {
        // Distinct positions are required so the autapse-prevention (same-position) check
        // never accidentally zeroes out the one legitimate target.
        positions[i] = SimpleVec3d{ static_cast<double>(i) * 10.0, 0.0, 0.0 };
    }

    auto vacant_dendrites = std::vector<RelearnTypes::counter_type>(number_neurons, 0U);
    vacant_dendrites[sink_index] = 1000U;

    // One task per source neuron except the sink itself (the sink connecting to itself is an
    // autapse by construction and is not a meaningful case here).
    auto mapping = std::vector<RelearnTypes::counter_type>{};
    for (auto i = 0UL; i < number_neurons; ++i) {
        if (i != sink_index) {
            mapping.emplace_back(static_cast<RelearnTypes::counter_type>(i));
        }
    }
    const auto target_size = mapping.size();

    for (auto it = 0U; it < 50U; ++it) {
        auto random_nums = std::vector<double>(target_size);
        RandomHolder::fill(RandomHolderKey::Algorithm, std::begin(random_nums), std::end(random_nums), 0.0, 1.0);

        auto target_array = std::vector<std::uint64_t>(target_size, std::numeric_limits<std::uint64_t>::max());

        auto d_pos = DeviceArray<SimpleVec3d>(std::span<const SimpleVec3d>(positions));
        NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ mapping, vacant_dendrites, random_nums, target_size }, target_array, squared_sigma_inv);

        for (const auto found : target_array) {
            ASSERT_EQ(found, sink_index);
        }
    }
}

// A general randomized scenario (mirrors the population used elsewhere for CPU Naive tests):
// as long as at least one neuron other than the source has a vacant dendrite, the kernel's
// autapse-prevention (a neuron's attractiveness to itself is always zero) must guarantee the
// found target is never the source itself.
TEST_F(NaiveCUDATest, testFindTargetNeuronsNeverPicksSourceAsTarget) {
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt) + 2;
    const auto& [minimum, maximum] = SimulationFactory::get_random_simulation_box_size(mt);

    auto positions = std::vector<SimpleVec3d>(number_neurons);
    auto vacant_dendrites = std::vector<RelearnTypes::counter_type>(number_neurons);
    auto sum_vacant_dendrites = 0UL;

    for (auto i = 0UL; i < number_neurons; ++i) {
        positions[i] = to_simple_vec3d(SimulationFactory::get_random_position_in_box(minimum, maximum, mt));
        const auto vd = RandomFactory::get_random_integer<RelearnTypes::counter_type>(0, 20, mt);
        vacant_dendrites[i] = vd;
        sum_vacant_dendrites += vd;
    }

    // Guarantee at least one neuron other than neuron 0 has a vacant dendrite so that neuron 0's
    // task is guaranteed to have a real (non-autapse) candidate.
    if (sum_vacant_dendrites == 0) {
        vacant_dendrites[number_neurons - 1] = 5U;
    }

    // One task per source neuron, source i maps to itself.
    auto mapping = std::vector<RelearnTypes::counter_type>(number_neurons);
    for (auto i = 0UL; i < number_neurons; ++i) {
        mapping[i] = static_cast<RelearnTypes::counter_type>(i);
    }

    auto random_nums = std::vector<double>(number_neurons);
    RandomHolder::fill(RandomHolderKey::Algorithm, std::begin(random_nums), std::end(random_nums), 0.0, 1.0);

    auto target_array = std::vector<std::uint64_t>(number_neurons, std::numeric_limits<std::uint64_t>::max());

    auto d_pos = DeviceArray<SimpleVec3d>(std::span<const SimpleVec3d>(positions));
    NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ mapping, vacant_dendrites, random_nums, number_neurons }, target_array, squared_sigma_inv);

    for (auto i = 0UL; i < number_neurons; ++i) {
        ASSERT_NE(target_array[i], i);
    }
}

// The output must actually reflect the GPU computation for every run, not just leave the
// output buffer at its default-constructed state -- this directly guards against the output
// parameter silently failing to propagate results back to the caller.
TEST_F(NaiveCUDATest, testFindTargetNeuronsActuallyWritesOutput) {
    constexpr auto number_neurons = 3UL;

    auto positions = std::vector<SimpleVec3d>{
        SimpleVec3d{ 0.0, 0.0, 0.0 },
        SimpleVec3d{ 10.0, 0.0, 0.0 },
        SimpleVec3d{ 20.0, 0.0, 0.0 },
    };
    // Only neuron 2 has vacant dendrites, so neuron 0's task must resolve to 2 -- never to the
    // default-initialized 0.
    auto vacant_dendrites = std::vector<RelearnTypes::counter_type>{ 0U, 0U, 5U };
    auto mapping = std::vector<RelearnTypes::counter_type>{ 0U };
    auto random_nums = std::vector<double>{ 0.5 };
    auto target_array = std::vector<std::uint64_t>{ 0U };

    auto d_pos = DeviceArray<SimpleVec3d>(std::span<const SimpleVec3d>(positions));
    NaiveCUDA_CU::find_target_neurons(d_pos, number_neurons, NaiveCUDA_CU::NaiveTargetSelectionTask{ mapping, vacant_dendrites, random_nums, 1UL }, target_array, squared_sigma_inv);

    ASSERT_EQ(target_array[0], 2UL);
}

// ────────────────────────────────────────────────────────────────────────────
// NaiveCUDA (the class itself, not just the free NaiveCUDA_CU functions)
// ────────────────────────────────────────────────────────────────────────────

// Regression test for a bug where NaiveCUDA::update_connectivity() (inherited unchanged from
// ForwardAlgorithm) always routed to update_connectivity_cuda_aware() in a CUDA-enabled build,
// which calls the find_target_neurons_cuda_aware()/process_requests_aware()/
// process_responses_aware() family -- none of which NaiveCUDA overrides, since it never
// implemented CUDA-aware MPI. Every call used to fall through to the CUDA_NOT_SUPPORTED default
// and throw immediately. NaiveCUDA now overrides update_connectivity() itself to force the
// host-MPI pipeline (find_target_neurons/process_requests/process_responses, which it does
// implement). This test exercises that full pipeline end-to-end and checks it actually creates
// the expected synapse instead of throwing.
TEST_F(NaiveCUDATest, testUpdateConnectivityConnectsTwoNeurons) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI rank.\n";
        }
        return;
    }

    constexpr auto number_neurons = RelearnTypes::number_neurons_type{ 2 };
    const auto box = RelearnTypes::bounding_box_type{ RelearnTypes::position_type{ 0.0, 0.0, 0.0 }, RelearnTypes::position_type{ 1.0, 1.0, 1.0 } };

    const auto neuron_positions = std::vector<RelearnTypes::position_type>{
        RelearnTypes::position_type{ 0.1F, 0.1F, 0.1F },
        RelearnTypes::position_type{ 0.9F, 0.9F, 0.9F },
    };

    auto extra_infos = std::make_shared<NeuronsExtraInfo>();
    extra_infos->init(number_neurons);
    extra_infos->set_positions(neuron_positions);

    const auto signal_types = std::vector<SignalType>(number_neurons, SignalType::Excitatory);
    // Every neuron offers exactly one vacant axon and one vacant excitatory dendrite -- with only
    // two neurons and autapses forbidden, each one's axon is forced to connect to the other's
    // dendrite, giving a fully deterministic expected outcome.
    auto synaptic_elements = SynapticElementsFactory::construct_synaptic_elements_with_fixed_number_axons_dendrites(extra_infos, signal_types, 1.0, 1.0);

    auto network_graph = NetworkGraphFactory::construct_empty_network_graph(number_neurons);

    auto naive_cuda = NaiveCUDA(box, std::make_shared<Morton>(0));
    naive_cuda.set_probability_kernel(KernelFactory::get_standard_gaussian());
    naive_cuda.set_synaptic_elements(synaptic_elements);
    naive_cuda.set_network_graph(network_graph);
    naive_cuda.set_neuron_extra_infos(extra_infos);

    naive_cuda.init(number_neurons);

    const auto vacant_axons = synaptic_elements->get_vacant_elements(SynapticElementType::Axon);
    const auto vacant_excitatory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteExcitatory);
    const auto vacant_inhibitory_dendrites = synaptic_elements->get_vacant_elements(SynapticElementType::DendriteInhibitory);

    ASSERT_NO_THROW(naive_cuda.prepare_update_connectivity(signal_types, vacant_axons, vacant_excitatory_dendrites, vacant_inhibitory_dendrites));
    ASSERT_NO_THROW(std::ignore = naive_cuda.update_connectivity(number_neurons));

    const auto& number_connected_axons = synaptic_elements->get_connected_elements(SynapticElementType::Axon);
    const auto total_connected_axons = std::accumulate(number_connected_axons.begin(), number_connected_axons.end(), RelearnTypes::counter_type{ 0 });

    ASSERT_EQ(total_connected_axons, 2) << "Each of the two neurons has exactly one vacant axon and must connect it to the other neuron's dendrite";
}

#endif
