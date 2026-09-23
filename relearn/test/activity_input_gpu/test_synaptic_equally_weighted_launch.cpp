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

#include "cuda/CudaBaseBridgeFunctions.h"
#include "cuda/CudaConfig.h"
#include "cuda/input/Handle.h"
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#include "cuda/network_graph/NetworkGPUType.h"
#include "cuda/network_graph/NetworkGraphGPU.h"
#include "cuda/network_graph/NetworkHandle.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "neurons/NetworkGraph.h"
#include "neurons/enums/FiredStatus.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <mpi-wrapper/core/MPIRank.h>

#include <string>
#include <tuple>
#include <vector>

// Regression/coverage test for launch()'s dispatch, covering every (SpikeMode, FireInformation)
// pairing that's both valid *and* semantically reachable through production usage.
//
// launch() supports 3 SpikeModes x 2 FireInformations, but two of those six pairings abort via
// RELEARN_DEVICE_CUDA_CHECK(false, ...) on the device (LocalVectorLookupSpike::contains only
// supports LocalVectorFireInformation; BinarySearchSpike::contains only supports
// NeuronIDsFireInformation) -- not practical to assert on here, so not exercised.
//
// Of the remaining four, FireInformation::LocalVector is only ever meaningful paired with
// IterationMode::Spikes and local=true (it reads *this rank's own* fired-status array, exactly
// what update_local_input() does), and FireInformation::NeuronIDs is only ever meaningful paired
// with IterationMode::Neurons and local=false (it reads a flattened list of fired neuron IDs
// grouped by *remote* rank, exactly what update_distant_input() does). Pairing either fire
// information with the "wrong" iteration mode/locality is not a real usage the production code
// ever constructs, and (confirmed while writing this test) reliably segfaults -- OnlyOutgoing's
// local-incoming cursor was never exercised outside its one real caller, IterationMode::Spikes.
//
// That leaves exactly four combinations to cover:
//   - (LocalVector spike, LocalVector fire) -- production's real local-input path
//   - (Set spike,         LocalVector fire) -- alternate spike encoding for local input; never
//                                              used by production but valid and untested
//   - (Set spike,         NeuronIDs fire)   -- production's real distant-input path when
//                                              Config::do_binary_search is false
//   - (BinarySearch spike, NeuronIDs fire)  -- production's real distant-input path when
//                                              Config::do_binary_search is true (the default)
// The latter two are production code paths that are *never actually reached* in a typical
// single-process test run: update_distant_input() early-returns whenever
// mpiPP::MPIInfo::get_number_ranks() == 1, so launch() is never called for them at all outside
// this test.
//
// WeightMode and NetworkMode are both fields of LaunchConfig but, as of this writing, are never
// read anywhere in launch()'s dispatch -- so a fixed placeholder value is used for each below.

namespace {

template <typename T>
struct DeviceArrayFixture {
    T* ptr{};
    std::size_t count{};

    explicit DeviceArrayFixture(const std::vector<T>& host_data)
        : count(host_data.size()) {
        cudaMalloc_bridge(reinterpret_cast<void**>(&ptr), sizeof(T) * count);
        cudaMemcpy_to_device_bridge(ptr, host_data.data(), sizeof(T) * count);
    }

    ~DeviceArrayFixture() {
        cudaFree_bridge(ptr);
    }

    DeviceArrayFixture(const DeviceArrayFixture&) = delete;
    DeviceArrayFixture& operator=(const DeviceArrayFixture&) = delete;

    [[nodiscard]] std::vector<T> download() const {
        auto result = std::vector<T>(count);
        cudaMemcpy_to_host_bridge(result.data(), ptr, sizeof(T) * count);
        return result;
    }
};

} // namespace

struct LaunchCombo {
    SpikeMode spike;
    FireInformation fire;
    std::string name;
};

class SynapticEquallyWeightedLaunchTest : public RelearnTest, public ::testing::WithParamInterface<LaunchCombo> { };

// Builds a 4-local-neuron, 3-rank network with two edge sets onto neuron 3, mirroring the same
// (fired: +2, not fired: +5, fired: -3) shape for both the local and the distant case, so every
// combination must compute input[3] == 2 + (-3) == -1 regardless of which one produced it:
//   local:    0->3 (w=2), 1->3 (w=5), 2->3 (w=-3); neurons 0 and 2 fire, neuron 1 does not.
//   distant:  (rank1,7)->3 (w=2), (rank1,8)->3 (w=5), (rank2,9)->3 (w=-3); remote neurons 7 and 9
//             fire, remote neuron 8 does not.
TEST_P(SynapticEquallyWeightedLaunchTest, testAccumulatesOnlyFiredSourcesInput) {
    const auto& combo = GetParam();
    const auto local = combo.fire == FireInformation::LocalVector;
    const auto iteration = local ? IterationMode::Spikes : IterationMode::Neurons;

    constexpr auto number_neurons = CudaConfig::number_neurons_type{ 4 };
    constexpr auto my_rank = CudaConfig::mpi_rank_type{ 0 };
    constexpr auto number_ranks = 3;

    auto ng = std::make_shared<NetworkGraph>(mpiPP::MPIRank{ my_rank }, number_ranks);
    ng->init(number_neurons, NetworkGPUType::MEMORY_POOL);

    ng->add_synapse(PlasticLocalSynapse{ NeuronID{ 3 }, NeuronID{ 0 }, 2 });
    ng->add_synapse(PlasticLocalSynapse{ NeuronID{ 3 }, NeuronID{ 1 }, 5 });
    ng->add_synapse(PlasticLocalSynapse{ NeuronID{ 3 }, NeuronID{ 2 }, -3 });
    ng->add_synapse(PlasticDistantInSynapse{ NeuronID{ 3 }, RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 7 } }, 2 });
    ng->add_synapse(PlasticDistantInSynapse{ NeuronID{ 3 }, RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 8 } }, 5 });
    ng->add_synapse(PlasticDistantInSynapse{ NeuronID{ 3 }, RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 9 } }, -3 });
    ng->sync_with_gpu();
    // Required before any incoming-edge lookup: update_edges() (called by sync_with_gpu()) seeds
    // block_count/block_bitmap but not the bloom filter, which OnlyOutgoing's incoming cursor also
    // consults -- only rebuild() populates it (see NetworkGraphGPUBase::rebuild()'s rebuild_bloom()
    // call). Production always calls this once after loading initial edges (e.g. Neurons.cpp).
    ng->rebuild();

    std::unique_ptr<FireStatusCommunicatorHandle> fire_handle;
    std::unique_ptr<DeviceArrayFixture<FiredStatus>> d_fired;
    std::unique_ptr<DeviceArrayFixture<int>> d_incoming_displ;
    std::unique_ptr<DeviceArrayFixture<CudaConfig::number_neurons_type>> d_incoming_ids;

    if (local) {
        const auto h_fired = std::vector<FiredStatus>{ FiredStatus::Fired, FiredStatus::Inactive, FiredStatus::Fired, FiredStatus::Inactive };
        d_fired = std::make_unique<DeviceArrayFixture<FiredStatus>>(h_fired);
        fire_handle = std::make_unique<FireStatusLocalVectorHandle>(d_fired->ptr);
    } else {
        // number_ranks + 1 entries; rank 0 (this rank) contributes none, rank 1 has one fired id
        // (neuron 7, not 8), rank 2 has one fired id (neuron 9). Sorted within each rank's own
        // segment, as required by BinarySearchSpike's per-segment binary search.
        const auto h_incoming_displ = std::vector<int>{ 0, 0, 1, 2 };
        const auto h_incoming_ids = std::vector<CudaConfig::number_neurons_type>{ 7, 9 };
        d_incoming_displ = std::make_unique<DeviceArrayFixture<int>>(h_incoming_displ);
        d_incoming_ids = std::make_unique<DeviceArrayFixture<CudaConfig::number_neurons_type>>(h_incoming_ids);
        fire_handle = std::make_unique<FireStatusCommunicatorUncompressedHandle>(d_incoming_displ->ptr, d_incoming_ids->ptr);
    }

    const auto h_input = std::vector<CudaConfig::input_type>(number_neurons, -999.0);
    auto d_input = DeviceArrayFixture<CudaConfig::input_type>(h_input);

    const auto config = LaunchConfig{ WeightMode::Weighted, combo.spike, NetworkMode::Default, combo.fire, iteration };
    const auto handles = LaunchHandles{ ng->get_gpu_handle_const(), fire_handle.get() };
    const auto stream = StreamWrapper::default_stream();

    // Neurons mode pulls into a target via its incoming edges, so only neuron 3 needs processing.
    // Spikes mode pushes from each source via its outgoing edges, so every neuron must be
    // considered as a potential source (neuron 3 has none, so it contributes nothing either way).
    const auto [first, last] = iteration == IterationMode::Neurons
                                   ? std::pair{ CudaConfig::number_neurons_type{ 3 }, CudaConfig::number_neurons_type{ 4 } }
                                   : std::pair{ CudaConfig::number_neurons_type{ 0 }, number_neurons };

    const auto event = launch(first, last, d_input.ptr, config, handles, stream, my_rank, local, /*local_distant_helper=*/false, /*synapse_conductance=*/1.0);
    ASSERT_TRUE(event.has_value());
    cudaDeviceSynchronize_bridge();

    const auto result = d_input.download();
    ASSERT_EQ(result[3], -1.0) << "expected only the fired sources (+2, -3) to contribute";
}

// The two (SpikeMode, FireInformation) pairings launch() does not support -- checked eagerly at
// the top of launch(), before any GPU work is launched, precisely so they fail this way instead
// of aborting the whole process via a device-side RELEARN_DEVICE_CUDA_CHECK(false, ...). Since the
// check is the very first thing launch() does, it never touches handles/d_input, so passing
// trivial/empty values for them here is safe.
class SynapticEquallyWeightedLaunchInvalidComboTest : public RelearnTest { };

TEST_F(SynapticEquallyWeightedLaunchInvalidComboTest, testThrowsForLocalVectorSpikeWithNeuronIDsFireInformation) {
    const auto config = LaunchConfig{ WeightMode::Weighted, SpikeMode::LocalVector, NetworkMode::Default, FireInformation::NeuronIDs, IterationMode::Neurons };
    const auto handles = LaunchHandles{ NetworkHandle{}, nullptr };
    const auto stream = StreamWrapper::default_stream();

    ASSERT_THROW_NO_PRINT(std::ignore = launch(0, 1, nullptr, config, handles, stream, 0, true, false, 1.0), RelearnException);
}

TEST_F(SynapticEquallyWeightedLaunchInvalidComboTest, testThrowsForBinarySearchSpikeWithLocalVectorFireInformation) {
    const auto config = LaunchConfig{ WeightMode::Weighted, SpikeMode::BinarySearch, NetworkMode::Default, FireInformation::LocalVector, IterationMode::Neurons };
    const auto handles = LaunchHandles{ NetworkHandle{}, nullptr };
    const auto stream = StreamWrapper::default_stream();

    ASSERT_THROW_NO_PRINT(std::ignore = launch(0, 1, nullptr, config, handles, stream, 0, true, false, 1.0), RelearnException);
}

INSTANTIATE_TEST_SUITE_P(AllValidReachableSpikeFireCombinations, SynapticEquallyWeightedLaunchTest,
                         ::testing::Values(
                             LaunchCombo{ SpikeMode::LocalVector, FireInformation::LocalVector, "LocalVector_LocalVector" },
                             LaunchCombo{ SpikeMode::Set, FireInformation::LocalVector, "Set_LocalVector" },
                             LaunchCombo{ SpikeMode::Set, FireInformation::NeuronIDs, "Set_NeuronIDs" },
                             LaunchCombo{ SpikeMode::BinarySearch, FireInformation::NeuronIDs, "BinarySearch_NeuronIDs" }),
                         [](const ::testing::TestParamInfo<LaunchCombo>& param_info) { return param_info.param.name; });

#endif
