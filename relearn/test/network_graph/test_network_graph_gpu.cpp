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
#include "test_network_graph_gpu_device.h"

#include "cuda/network_graph/GPUEdges.h"
#include "cuda/network_graph/NetworkGraphGPU.h"
#include "cuda/network_graph/NetworkHandle.h"
#include "neurons/NetworkGraph.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"

#include <gtest/gtest.h>

#include <mpi-wrapper/core/MPIRank.h>

#include <vector>

class NetworkGraphGPUTest : public RelearnTest { };

// Type aliases matching GPUEdgesBase internals
using neuron_id_t = GPUEdgesBase::neuron_id_type; // uint32_t
using weight_t = GPUEdgesBase::weight_type;       // int16_t
using rank_t = GPUEdgesBase::mpi_rank_type;       // uint16_t

// Host-side snapshots of the internal GraphStorage objects (GraphStorage no longer exposes raw
// sizes/data arrays directly; copy_to_cpu() downloads the per-neuron vectors instead).
static std::vector<std::vector<std::uint32_t>> neuron_store(GPUEdgesBase& e) {
    return e.other_neurons_storage->copy_to_cpu();
}
static std::vector<std::vector<GPUEdgesBase::internal_weight_type>> weight_store(GPUEdgesBase& e) {
    return e.weighted.weight_storage->copy_to_cpu();
}
static std::vector<std::vector<rank_t>> rank_store(GPUEdgesBase& e) {
    return e.other_ranks.other_ranks_storage->copy_to_cpu();
}
static std::vector<std::vector<bool>> excitatory_store(GPUEdgesBase& e) {
    return e.excitatory.excitatory_storage->copy_to_cpu();
}

// ────────────────────────────────────────────────────────────────────────────
// Storage initialisation / edge counts
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, EmptyGraph_AllEdgeCountsAreZero) {
    constexpr neuron_id_t N = 6;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 1;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   true,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 128);

    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    edges.update_edges(local, distant);

    const auto ns = neuron_store(edges);
    ASSERT_EQ(ns.size(), N);
    for (neuron_id_t i = 0; i < N; ++i) {
        EXPECT_EQ(ns[i].size(), 0U) << "neuron " << i << " should have no edges";
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Neuron-ID storage
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, LocalEdges_NeuronIDsStoredInOrder) {
    constexpr neuron_id_t N = 5;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{ false, true }, R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    local[0].emplace_back(NeuronID{ 2 }, 1);
    local[0].emplace_back(NeuronID{ 4 }, 1);
    local[1].emplace_back(NeuronID{ 3 }, 1);

    edges.update_edges(local, distant);

    const auto ns = neuron_store(edges);
    ASSERT_EQ(ns.size(), N);

    EXPECT_EQ(ns[0].size(), 2U);
    EXPECT_EQ(ns[0][0], 2U);
    EXPECT_EQ(ns[0][1], 4U);

    EXPECT_EQ(ns[1].size(), 1U);
    EXPECT_EQ(ns[1][0], 3U);

    EXPECT_EQ(ns[2].size(), 0U);
    EXPECT_EQ(ns[3].size(), 0U);
    EXPECT_EQ(ns[4].size(), 0U);
}

TEST_F(NetworkGraphGPUTest, DistantEdges_NeuronIDsStoredCorrectly) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    distant[0].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 10 } }, 1);
    distant[0].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 20 } }, 1);
    distant[2].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 15 } }, 1);

    edges.update_edges(local, distant);

    const auto ns = neuron_store(edges);
    ASSERT_EQ(ns.size(), N);

    EXPECT_EQ(ns[0].size(), 2U);
    EXPECT_EQ(ns[0][0], 10U);
    EXPECT_EQ(ns[0][1], 20U);

    EXPECT_EQ(ns[1].size(), 0U);
    EXPECT_EQ(ns[2].size(), 1U);
    EXPECT_EQ(ns[2][0], 15U);

    EXPECT_EQ(ns[3].size(), 0U);
}

// ────────────────────────────────────────────────────────────────────────────
// Rank storage
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, LocalEdges_RankEqualsMyRank) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 1;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    local[0].emplace_back(NeuronID{ 2 }, 1);
    local[0].emplace_back(NeuronID{ 3 }, 1);

    edges.update_edges(local, distant);

    const auto rs = rank_store(edges);
    ASSERT_EQ(rs.size(), N);

    EXPECT_EQ(rs[0][0], my_rank);
    EXPECT_EQ(rs[0][1], my_rank);
}

TEST_F(NetworkGraphGPUTest, DistantEdges_RankMatchesTargetRank) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 4;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    distant[0].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 5 } }, 1);
    distant[0].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 3 }, NeuronID{ 7 } }, 1);

    edges.update_edges(local, distant);

    const auto rs = rank_store(edges);
    ASSERT_EQ(rs.size(), N);

    EXPECT_EQ(rs[0].size(), 2U);
    EXPECT_EQ(rs[0][0], 1U); // first edge → rank 1
    EXPECT_EQ(rs[0][1], 3U); // second edge → rank 3
}

TEST_F(NetworkGraphGPUTest, MixedEdges_LocalBeforeDistantInStorage) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    local[0].emplace_back(NeuronID{ 2 }, 1);
    distant[0].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 7 } }, 1);

    edges.update_edges(local, distant);

    const auto ns = neuron_store(edges);
    const auto rs = rank_store(edges);

    EXPECT_EQ(ns[0].size(), 2U);
    // local edge stored first
    EXPECT_EQ(ns[0][0], 2U);
    EXPECT_EQ(rs[0][0], static_cast<rank_t>(my_rank));
    // distant edge stored second
    EXPECT_EQ(ns[0][1], 7U);
    EXPECT_EQ(rs[0][1], 2U);
}

// ────────────────────────────────────────────────────────────────────────────
// Weight storage (weighted mode)
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, WeightedMode_WeightsStoredExactly) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   true,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    local[0].emplace_back(NeuronID{ 2 }, 7);
    local[0].emplace_back(NeuronID{ 3 }, -4);
    distant[1].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 5 } }, 12);

    edges.update_edges(local, distant);

    const auto ws = weight_store(edges);
    ASSERT_EQ(ws.size(), N);

    EXPECT_EQ(ws[0].size(), 2U);
    EXPECT_EQ(ws[0][0], static_cast<weight_t>(7));
    EXPECT_EQ(ws[0][1], static_cast<weight_t>(-4));

    EXPECT_EQ(ws[1].size(), 1U);
    EXPECT_EQ(ws[1][0], static_cast<weight_t>(12));
}

TEST_F(NetworkGraphGPUTest, WeightedMode_ExactlyOneEntryPerEdge) {
    // In weighted mode each edge always produces exactly 1 storage entry,
    // regardless of the magnitude of the weight.
    constexpr neuron_id_t N = 3;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   true,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    local[0].emplace_back(NeuronID{ 2 }, 5); // weight 5 → still only 1 entry

    edges.update_edges(local, distant);

    const auto ns = neuron_store(edges);
    EXPECT_EQ(ns[0].size(), 1U);

    const auto ws = weight_store(edges);
    EXPECT_EQ(ws[0].size(), 1U);
    EXPECT_EQ(ws[0][0], static_cast<weight_t>(5));
}

// ────────────────────────────────────────────────────────────────────────────
// Weight expansion (unweighted mode)
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, UnweightedMode_WeightMagnitudeExpandsEdgeCount) {
    // Without weighted storage the absolute value of the weight determines
    // how many copies of the edge are inserted.
    constexpr neuron_id_t N = 3;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    local[0].emplace_back(NeuronID{ 2 }, 3); // expanded to 3 copies

    edges.update_edges(local, distant);

    const auto ns = neuron_store(edges);
    EXPECT_EQ(ns[0].size(), 3U);
    for (auto j = 0U; j < 3U; ++j) {
        EXPECT_EQ(ns[0][j], 2U) << "all copies should point to neuron 2";
    }
}

TEST_F(NetworkGraphGPUTest, UnweightedMode_NegativeWeightExpandsEdgeCount) {
    constexpr neuron_id_t N = 3;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    local[0].emplace_back(NeuronID{ 1 }, -2); // expanded to 2 copies, all inhibitory

    edges.update_edges(local, distant);

    const auto ns = neuron_store(edges);
    EXPECT_EQ(ns[0].size(), 2U);

    const auto es = excitatory_store(edges);
    EXPECT_EQ(es[0][0], false);
    EXPECT_EQ(es[0][1], false);
}

// ────────────────────────────────────────────────────────────────────────────
// Excitatory / inhibitory flag storage
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, ExcitatoryFlags_PositiveWeightIsExcitatory) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    // has_weights=false: excitatory storage is only maintained when weighted storage is
    // disabled (the two are mutually exclusive -- see GPUEdgesBase's constructor). Weights of
    // magnitude 1 keep exactly one stored entry per pushed edge.
    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    local[0].emplace_back(NeuronID{ 2 }, 1);  // positive → excitatory
    local[0].emplace_back(NeuronID{ 3 }, -1); // negative → inhibitory

    edges.update_edges(local, distant);

    const auto es = excitatory_store(edges);
    EXPECT_EQ(es[0][0], true);
    EXPECT_EQ(es[0][1], false);
}

TEST_F(NetworkGraphGPUTest, ExcitatoryFlags_DistantEdgesAlsoTagged) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);

    distant[1].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 8 } }, 1);  // excitatory
    distant[1].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 9 } }, -1); // inhibitory

    edges.update_edges(local, distant);

    const auto es = excitatory_store(edges);
    EXPECT_EQ(es[1][0], true);
    EXPECT_EQ(es[1][1], false);
}

// ────────────────────────────────────────────────────────────────────────────
// EdgeHandle – handle validity
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, GetHandle_ReturnsNonNullView) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   true,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 1);
    edges.update_edges(local, distant);

    const EdgeHandle handle = edges.get_handle();
    EXPECT_NE(handle.view_impl, nullptr);
}

TEST_F(NetworkGraphGPUTest, GetHandle_NoWeightsGivesNonNullView) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   false,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    edges.update_edges(local, distant);

    const EdgeHandle handle = edges.get_handle();
    EXPECT_NE(handle.view_impl, nullptr);
}

// ────────────────────────────────────────────────────────────────────────────
// Integration through the high-level NetworkGraph API
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, NetworkGraph_SyncWithGPU_LocalEdgesReflectedInStorage) {
    constexpr int number_neurons = 5;
    constexpr int number_ranks = 3;
    constexpr int my_rank = 1;

    auto ng = std::make_shared<NetworkGraph>(mpiPP::MPIRank{ my_rank }, number_ranks);
    ng->init(number_neurons, NetworkGPUType::MEMORY_POOL);

    ng->add_synapse(PlasticLocalSynapse{ NeuronID{ 3 }, NeuronID{ 0 }, 1 });
    ng->add_synapse(PlasticLocalSynapse{ NeuronID{ 4 }, NeuronID{ 0 }, -1 });
    ng->add_synapse(PlasticLocalSynapse{ NeuronID{ 2 }, NeuronID{ 1 }, 1 });

    ng->sync_with_gpu();

    const NetworkHandle view = ng->get_gpu_handle();
    EXPECT_NE(view.incoming_local_handle.view_impl, nullptr);
    EXPECT_NE(view.outgoing_local_handle.view_impl, nullptr);
}

TEST_F(NetworkGraphGPUTest, NetworkGraph_SyncWithGPU_DistantEdgesReflectedInStorage) {
    constexpr int number_neurons = 5;
    constexpr int number_ranks = 4;
    constexpr int my_rank = 0;

    auto ng = std::make_shared<NetworkGraph>(mpiPP::MPIRank{ my_rank }, number_ranks);
    ng->init(number_neurons, NetworkGPUType::MEMORY_POOL);

    ng->add_synapse(PlasticDistantInSynapse{ NeuronID{ 1 },
                                             RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 10 } }, 1 });
    ng->add_synapse(PlasticDistantInSynapse{ NeuronID{ 2 },
                                             RankNeuronId{ mpiPP::MPIRank{ 3 }, NeuronID{ 20 } }, -1 });
    ng->add_synapse(PlasticDistantOutSynapse{
        RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 5 } }, NeuronID{ 0 }, 1 });

    ng->sync_with_gpu();

    const NetworkHandle view = ng->get_gpu_handle();
    EXPECT_NE(view.incoming_distant_handle.view_impl, nullptr);
    EXPECT_NE(view.outgoing_distant_handle.view_impl, nullptr);
}

TEST_F(NetworkGraphGPUTest, NetworkGraph_RebuildMultipleTimes_Stable) {
    constexpr int number_neurons = 4;
    constexpr int number_ranks = 2;
    constexpr int my_rank = 0;

    auto ng = std::make_shared<NetworkGraph>(mpiPP::MPIRank{ my_rank }, number_ranks);
    ng->init(number_neurons, NetworkGPUType::MEMORY_POOL);

    ng->add_synapse(PlasticLocalSynapse{ NeuronID{ 2 }, NeuronID{ 0 }, 1 });
    ng->sync_with_gpu();

    // rebuild should be safe to call multiple times
    ng->rebuild();
    ng->rebuild();

    const NetworkHandle view = ng->get_gpu_handle();
    EXPECT_NE(view.incoming_local_handle.view_impl, nullptr);
}

// ────────────────────────────────────────────────────────────────────────────
// DefaultView device-function tests
// Each test calls the actual __device__ methods via a small verification kernel
// (implemented in test_network_graph_gpu_device.cu) and compares the results
// read back to host against the known input data.
// ────────────────────────────────────────────────────────────────────────────

// Helper: build a GPUEdgesBase with a fixed set of local + distant edges,
// call update_edges(), and return the populated object.
static std::unique_ptr<GPUEdgesBase> make_edges_weighted(
    neuron_id_t N, rank_t R, rank_t my_rank,
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local,
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant) {
    auto edges = std::make_unique<GPUEdgesBase>(LayoutType::MemoryPool, Features{
                                                                            true,
                                                                            true,
                                                                        },
                                                R, my_rank);
    edges->init(N, 128);
    edges->update_edges(local, distant);
    return edges;
}

// ── DefaultView::size() ──────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, DeviceView_Size_EmptyGraph) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto sizes = device_read_sizes(handle.view_impl, N);
    ASSERT_EQ(sizes.size(), N);
    for (neuron_id_t i = 0; i < N; ++i) {
        EXPECT_EQ(sizes[i], 0U) << "neuron " << i;
    }
}

TEST_F(NetworkGraphGPUTest, DeviceView_Size_MatchesInsertedEdges) {
    constexpr neuron_id_t N = 5;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 1);
    local[0].emplace_back(NeuronID{ 4 }, 1);
    local[1].emplace_back(NeuronID{ 3 }, 1);
    distant[2].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 7 } }, 1);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto sizes = device_read_sizes(handle.view_impl, N);
    ASSERT_EQ(sizes.size(), N);
    EXPECT_EQ(sizes[0], 2U);
    EXPECT_EQ(sizes[1], 1U);
    EXPECT_EQ(sizes[2], 1U);
    EXPECT_EQ(sizes[3], 0U);
    EXPECT_EQ(sizes[4], 0U);
}

// ── DefaultView::get_other_neuron() ──────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, DeviceView_GetOtherNeuron_LocalEdges) {
    constexpr neuron_id_t N = 5;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 1);
    local[0].emplace_back(NeuronID{ 4 }, 1);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto neurons = device_read_other_neurons(handle.view_impl, 0, 2);
    ASSERT_EQ(neurons.size(), 2U);
    EXPECT_EQ(neurons[0], 2U);
    EXPECT_EQ(neurons[1], 4U);
}

TEST_F(NetworkGraphGPUTest, DeviceView_GetOtherNeuron_DistantEdges) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    distant[1].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 10 } }, 1);
    distant[1].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 20 } }, 1);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto neurons = device_read_other_neurons(handle.view_impl, 1, 2);
    ASSERT_EQ(neurons.size(), 2U);
    EXPECT_EQ(neurons[0], 10U);
    EXPECT_EQ(neurons[1], 20U);
}

// ── DefaultView::get_weight() with weights ────────────────────────────────────

TEST_F(NetworkGraphGPUTest, DeviceView_GetWeight_StoredWeightsReturned) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 7);
    local[0].emplace_back(NeuronID{ 3 }, -4);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto weights = device_read_weights(handle.view_impl, 0, 2);
    ASSERT_EQ(weights.size(), 2U);
    EXPECT_EQ(weights[0], static_cast<weight_t>(7));
    EXPECT_EQ(weights[1], static_cast<weight_t>(-4));
}

// ── DefaultView::get_weight() without weights → always 1 ─────────────────────

TEST_F(NetworkGraphGPUTest, DeviceView_GetWeight_NoWeightStorage_ReturnsOne) {
    // Build edges WITHOUT has_weights – the view's weights pointer will be null,
    // and DefaultView::get_weight() is specified to return 1 in that case.
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    GPUEdgesBase edges(LayoutType::MemoryPool, Features{
                                                   false,
                                                   true,
                                               },
                       R, my_rank);
    edges.init(N, 64);

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 5); // weight=5, but storage is unweighted → expanded to 5 copies
    edges.update_edges(local, distant);

    const EdgeHandle handle = edges.get_handle();

    // 5 copies of the edge → 5 calls to get_weight, each returning 1
    const auto weights = device_read_weights(handle.view_impl, 0, 5);
    ASSERT_EQ(weights.size(), 5U);
    for (const auto w : weights) {
        EXPECT_EQ(w, static_cast<weight_t>(1));
    }
}

// ── DefaultView::get_other_rank() ────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, DeviceView_GetOtherRank_LocalIsMyRank) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 1;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 1);
    local[0].emplace_back(NeuronID{ 3 }, 1);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto ranks = device_read_other_ranks(handle.view_impl, 0, 2);
    ASSERT_EQ(ranks.size(), 2U);
    EXPECT_EQ(ranks[0], static_cast<rank_t>(my_rank));
    EXPECT_EQ(ranks[1], static_cast<rank_t>(my_rank));
}

TEST_F(NetworkGraphGPUTest, DeviceView_GetOtherRank_DistantRankCorrect) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 4;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    distant[0].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 5 } }, 1);
    distant[0].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 3 }, NeuronID{ 9 } }, 1);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto ranks = device_read_other_ranks(handle.view_impl, 0, 2);
    ASSERT_EQ(ranks.size(), 2U);
    EXPECT_EQ(ranks[0], static_cast<rank_t>(1));
    EXPECT_EQ(ranks[1], static_cast<rank_t>(3));
}

TEST_F(NetworkGraphGPUTest, DeviceView_GetOtherRank_MixedLocalAndDistant) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 1);                                        // local  → rank 0
    distant[0].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 7 } }, 1); // distant → rank 2
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto ranks = device_read_other_ranks(handle.view_impl, 0, 2);
    ASSERT_EQ(ranks.size(), 2U);
    EXPECT_EQ(ranks[0], static_cast<rank_t>(my_rank)); // local stored first
    EXPECT_EQ(ranks[1], static_cast<rank_t>(2));
}

// ────────────────────────────────────────────────────────────────────────────
// DefaultView cursor-iteration tests
// These tests exercise the edge_begin / edge_end protocol directly on device.
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, DeviceView_IterateCursor_EmptyNeuron_NoEdges) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto result = device_iterate_edges_default(handle.view_impl, 0);
    EXPECT_TRUE(result.empty());
}

TEST_F(NetworkGraphGPUTest, DeviceView_IterateCursor_SingleLocalEdge_AllFieldsCorrect) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 1;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 7);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto result = device_iterate_edges_default(handle.view_impl, 0);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result[0].other_neuron, 2U);
    EXPECT_EQ(result[0].other_rank, static_cast<rank_t>(my_rank));
    EXPECT_EQ(result[0].weight, static_cast<weight_t>(7));
}

TEST_F(NetworkGraphGPUTest, DeviceView_IterateCursor_MultipleEdges_AllFieldsCorrect) {
    // Neuron 0 → local(2,w=3), local(4,w=-2), distant(rank=2,nid=10,w=5)
    constexpr neuron_id_t N = 5;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 3);
    local[0].emplace_back(NeuronID{ 4 }, -2);
    distant[0].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 10 } }, 5);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto result = device_iterate_edges_default(handle.view_impl, 0);
    ASSERT_EQ(result.size(), 3U);
    // local edges first (rank == my_rank), then distant
    EXPECT_EQ(result[0].other_neuron, 2U);
    EXPECT_EQ(result[0].other_rank, static_cast<rank_t>(my_rank));
    EXPECT_EQ(result[0].weight, static_cast<weight_t>(3));

    EXPECT_EQ(result[1].other_neuron, 4U);
    EXPECT_EQ(result[1].other_rank, static_cast<rank_t>(my_rank));
    EXPECT_EQ(result[1].weight, static_cast<weight_t>(-2));

    EXPECT_EQ(result[2].other_neuron, 10U);
    EXPECT_EQ(result[2].other_rank, static_cast<rank_t>(2));
    EXPECT_EQ(result[2].weight, static_cast<weight_t>(5));
}

TEST_F(NetworkGraphGPUTest, DeviceView_IterateCursor_NeuronCountMatchesSizeQuery) {
    // Verify that edge count from cursor == size() from device_read_sizes
    constexpr neuron_id_t N = 5;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 1 }, 1);
    local[0].emplace_back(NeuronID{ 3 }, 1);
    distant[2].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 9 } }, 1);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto sizes = device_read_sizes(handle.view_impl, N);
    for (neuron_id_t i = 0; i < N; ++i) {
        const auto iterated = device_iterate_edges_default(handle.view_impl, i);
        EXPECT_EQ(iterated.size(), sizes[i]) << "neuron " << i;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Large weight tests (MemoryPool weighted)
//
// The MemoryPool layout uses int8_t as the per-entry storage type (internal_weight_type).
// A weight whose absolute value exceeds INT8_MAX (127) is therefore split across multiple
// entries: ceil(|w| / 127) entries, each holding at most 127.  These tests verify that
// splitting and reconstruction.
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, DeviceView_GetWeight_LargePositiveWeight_SplitIntoMultipleEntries) {
    // weight=500: ceil(500/127)=4 entries [127, 127, 127, 119], sum=500
    constexpr neuron_id_t N = 3;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 1 }, 500);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto sizes = device_read_sizes(handle.view_impl, N);
    EXPECT_EQ(sizes[0], 4U); // ceil(500/127) = 4

    const auto weights = device_read_weights(handle.view_impl, 0, 4);
    ASSERT_EQ(weights.size(), 4U);
    int total = 0;
    for (const auto w : weights) {
        total += w;
    }
    EXPECT_EQ(total, 500);
    // First three entries saturated, last holds the remainder
    EXPECT_EQ(weights[0], static_cast<weight_t>(127));
    EXPECT_EQ(weights[1], static_cast<weight_t>(127));
    EXPECT_EQ(weights[2], static_cast<weight_t>(127));
    EXPECT_EQ(weights[3], static_cast<weight_t>(119)); // 500 - 3*127 = 119
}

TEST_F(NetworkGraphGPUTest, DeviceView_GetWeight_LargeNegativeWeight_SplitIntoMultipleEntries) {
    // weight=-300: ceil(300/128)=3 entries [-128, -128, -44], sum=-300
    constexpr neuron_id_t N = 3;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, -300);
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto sizes = device_read_sizes(handle.view_impl, N);
    EXPECT_EQ(sizes[0], 3U); // ceil(300/128) = 3

    const auto weights = device_read_weights(handle.view_impl, 0, 3);
    ASSERT_EQ(weights.size(), 3U);
    int total = 0;
    for (const auto w : weights) {
        total += w;
    }
    EXPECT_EQ(total, -300);
    EXPECT_EQ(weights[0], static_cast<weight_t>(-128));
    EXPECT_EQ(weights[1], static_cast<weight_t>(-128));
    EXPECT_EQ(weights[2], static_cast<weight_t>(-44)); // -300 + 2*128 = -44
}

TEST_F(NetworkGraphGPUTest, DeviceView_GetWeight_CursorLargeWeight_SumMatchesGetWeight) {
    // Cursor iteration and random-access get_weight are consistent for a split weight.
    // weight=254: ceil(254/127)=2 entries [127, 127], sum=254
    constexpr neuron_id_t N = 3;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 1 }, 254); // ceil(254/127)=2
    auto edges = make_edges_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto sizes = device_read_sizes(handle.view_impl, N);
    EXPECT_EQ(sizes[0], 2U);

    // Random-access weights
    const auto ra_weights = device_read_weights(handle.view_impl, 0, 2);
    ASSERT_EQ(ra_weights.size(), 2U);
    EXPECT_EQ(ra_weights[0], static_cast<weight_t>(127));
    EXPECT_EQ(ra_weights[1], static_cast<weight_t>(127));

    // Cursor yields the same values
    const auto cursor_edges = device_iterate_edges_default(handle.view_impl, 0);
    ASSERT_EQ(cursor_edges.size(), 2U);
    EXPECT_EQ(cursor_edges[0].weight, ra_weights[0]);
    EXPECT_EQ(cursor_edges[1].weight, ra_weights[1]);
    EXPECT_EQ(cursor_edges[0].other_neuron, 1U);
    EXPECT_EQ(cursor_edges[1].other_neuron, 1U); // same target, both entries
}

// ────────────────────────────────────────────────────────────────────────────
// MemoryPoolView (LayoutType::MemoryPool) device tests
// ────────────────────────────────────────────────────────────────────────────

// Helper: build a weighted MemoryPoolView GPUEdgesBase
static std::unique_ptr<GPUEdgesBase> make_edges_memory_pool_weighted(
    neuron_id_t N, rank_t R, rank_t my_rank,
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local,
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant) {
    auto e = std::make_unique<GPUEdgesBase>(
        LayoutType::MemoryPool, Features{
                                    true,
                                    false,
                                },
        R, my_rank);
    e->init(N, 256);
    e->update_edges(local, distant);
    return e;
}

TEST_F(NetworkGraphGPUTest, MemoryPoolView_Size_EmptyGraph_AllZero) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto sizes = device_memory_pool_view_sizes(handle.view_impl, N);
    ASSERT_EQ(sizes.size(), N);
    for (neuron_id_t i = 0; i < N; ++i) {
        EXPECT_EQ(sizes[i], 0U) << "neuron " << i;
    }
}

TEST_F(NetworkGraphGPUTest, MemoryPoolView_Size_AfterUpdateEdges_MatchesEdgeCounts) {
    // MemoryPool layout with weights stores one entry per synapse (direct int8_t store),
    // unlike unweighted MemoryPool storage which stores abs(weight) copies.
    constexpr neuron_id_t N = 5;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 2 }, 5); // weight=5 → one int8_t entry (value 5), NOT 5 copies
    local[1].emplace_back(NeuronID{ 3 }, 1);
    local[1].emplace_back(NeuronID{ 4 }, 2);
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto sizes = device_memory_pool_view_sizes(handle.view_impl, N);
    ASSERT_EQ(sizes.size(), N);
    EXPECT_EQ(sizes[0], 1U); // one entry, weight stored as int8_t(5)
    EXPECT_EQ(sizes[1], 2U);
    EXPECT_EQ(sizes[2], 0U);
    EXPECT_EQ(sizes[3], 0U);
    EXPECT_EQ(sizes[4], 0U);
}

TEST_F(NetworkGraphGPUTest, MemoryPoolView_IterateCursor_SingleEdge_AllFieldsCorrect) {
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 1;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 3 }, 7);
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto result = device_memory_pool_view_iterate_edges(handle.view_impl, 0);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result[0].other_neuron, 3U);
    // MemoryPoolView without has_ranks stores no rank; DynamicEdgeCursor falls back to my_rank when
    // has_ranks=false (see DynamicEdgeCursor<IdT>::next() in Views.cu).
    EXPECT_EQ(result[0].other_rank, static_cast<rank_t>(my_rank));
    EXPECT_EQ(result[0].weight, static_cast<weight_t>(7));
}

TEST_F(NetworkGraphGPUTest, MemoryPoolView_IterateCursor_MultipleEdges_AllFieldsCorrect) {
    constexpr neuron_id_t N = 5;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[0].emplace_back(NeuronID{ 1 }, 3);
    local[0].emplace_back(NeuronID{ 4 }, -2);
    local[0].emplace_back(NeuronID{ 2 }, 5);
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto result = device_memory_pool_view_iterate_edges(handle.view_impl, 0);
    ASSERT_EQ(result.size(), 3U);
    EXPECT_EQ(result[0].other_neuron, 1U);
    EXPECT_EQ(result[0].weight, static_cast<weight_t>(3));
    EXPECT_EQ(result[1].other_neuron, 4U);
    EXPECT_EQ(result[1].weight, static_cast<weight_t>(-2));
    EXPECT_EQ(result[2].other_neuron, 2U);
    EXPECT_EQ(result[2].weight, static_cast<weight_t>(5));
}

TEST_F(NetworkGraphGPUTest, MemoryPoolView_IterateCursor_EmptyNeuron_NoEdges) {
    constexpr neuron_id_t N = 3;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    local[1].emplace_back(NeuronID{ 2 }, 1); // only neuron 1 has an edge
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    const auto result = device_memory_pool_view_iterate_edges(handle.view_impl, 0); // neuron 0
    EXPECT_TRUE(result.empty());
}

// ────────────────────────────────────────────────────────────────────────────
// MemoryPoolView multi-entry weight saturation
//
// MemoryPoolView stores weights as int8_t per entry.  When the weight of an entry
// reaches INT8_MAX (127) or INT8_MIN (−127), add_synapse skips that entry and
// creates a new one.  After 200 excitatory add_synapse calls on the same pair
// the storage contains 2 entries: weight[0]=127, weight[1]=73, sum=200.
// ────────────────────────────────────────────────────────────────────────────

TEST_F(NetworkGraphGPUTest, MemoryPoolView_AddSynapses_ExactlySaturation_SingleEntry) {
    // 127 calls → entry 0 = 127; only one int8_t entry required.
    constexpr neuron_id_t N = 2;
    constexpr rank_t R = 1;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    device_memory_pool_view_add_synapse_n_times(handle.view_impl, 0, 1, 127, true);

    const auto sizes = device_memory_pool_view_sizes(handle.view_impl, N);
    EXPECT_EQ(sizes[0], 1U);

    const auto result = device_memory_pool_view_iterate_edges(handle.view_impl, 0);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result[0].other_neuron, 1U);
    EXPECT_EQ(result[0].weight, static_cast<weight_t>(127));
}

TEST_F(NetworkGraphGPUTest, MemoryPoolView_AddSynapses_OnePastSaturation_TwoEntries) {
    // 128 calls → entry 0 saturates at 127, entry 1 created with weight 1.
    constexpr neuron_id_t N = 2;
    constexpr rank_t R = 1;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    device_memory_pool_view_add_synapse_n_times(handle.view_impl, 0, 1, 128, true);

    const auto sizes = device_memory_pool_view_sizes(handle.view_impl, N);
    EXPECT_EQ(sizes[0], 2U); // two int8_t entries

    const auto result = device_memory_pool_view_iterate_edges(handle.view_impl, 0);
    ASSERT_EQ(result.size(), 2U);
    EXPECT_EQ(result[0].other_neuron, 1U);
    EXPECT_EQ(result[0].weight, static_cast<weight_t>(127));
    EXPECT_EQ(result[1].other_neuron, 1U);
    EXPECT_EQ(result[1].weight, static_cast<weight_t>(1));
}

TEST_F(NetworkGraphGPUTest, MemoryPoolView_AddSynapses_200Calls_TwoEntries_SumCorrect) {
    // 200 calls: entry 0 = 127, entry 1 = 73 → sum = 200.
    constexpr neuron_id_t N = 2;
    constexpr rank_t R = 1;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    device_memory_pool_view_add_synapse_n_times(handle.view_impl, 0, 1, 200, true);

    const auto sizes = device_memory_pool_view_sizes(handle.view_impl, N);
    EXPECT_EQ(sizes[0], 2U);

    const auto result = device_memory_pool_view_iterate_edges(handle.view_impl, 0);
    ASSERT_EQ(result.size(), 2U);
    const int total_weight = result[0].weight + result[1].weight;
    EXPECT_EQ(total_weight, 200);
    EXPECT_EQ(result[0].weight, static_cast<weight_t>(127));
    EXPECT_EQ(result[1].weight, static_cast<weight_t>(73));
}

TEST_F(NetworkGraphGPUTest, MemoryPoolView_AddSynapses_InhibitorySaturation_SingleEntry) {
    // Inhibitory saturation is at INT8_MIN = −128.
    // Calls 1–128: entry 0 goes from −1 to −128; still one entry.
    constexpr neuron_id_t N = 2;
    constexpr rank_t R = 1;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    device_memory_pool_view_add_synapse_n_times(handle.view_impl, 0, 1, 128, false);

    const auto sizes = device_memory_pool_view_sizes(handle.view_impl, N);
    EXPECT_EQ(sizes[0], 1U);

    const auto result = device_memory_pool_view_iterate_edges(handle.view_impl, 0);
    ASSERT_EQ(result.size(), 1U);
    EXPECT_EQ(result[0].weight, static_cast<weight_t>(-128));
}

TEST_F(NetworkGraphGPUTest, MemoryPoolView_AddSynapses_InhibitoryOnePastSaturation_TwoEntries) {
    // Call 129 finds entry 0 saturated at −128 and creates a second entry with −1.
    constexpr neuron_id_t N = 2;
    constexpr rank_t R = 1;
    constexpr rank_t my_rank = 0;
    const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local(N);
    const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant(N);
    auto edges = make_edges_memory_pool_weighted(N, R, my_rank, local, distant);
    const EdgeHandle handle = edges->get_handle();

    device_memory_pool_view_add_synapse_n_times(handle.view_impl, 0, 1, 129, false);

    const auto sizes = device_memory_pool_view_sizes(handle.view_impl, N);
    EXPECT_EQ(sizes[0], 2U);

    const auto result = device_memory_pool_view_iterate_edges(handle.view_impl, 0);
    ASSERT_EQ(result.size(), 2U);
    EXPECT_EQ(result[0].weight, static_cast<weight_t>(-128));
    EXPECT_EQ(result[1].weight, static_cast<weight_t>(-1));
    const int total_weight = result[0].weight + result[1].weight;
    EXPECT_EQ(total_weight, -129);
}

// ────────────────────────────────────────────────────────────────────────────
// Parametrized tests – all 4 NetworkGraphGPUBase configurations
// ────────────────────────────────────────────────────────────────────────────

struct GPUConfigSpec {
    const char* name;
    bool has_local_edges;
    LayoutType layout_local_in;
    Features features_local_in;
    LayoutType layout_local_out;
    Features features_local_out;
    LayoutType layout_distant_in;
    Features features_distant_in;
    LayoutType layout_distant_out;
    Features features_distant_out;
};

static NetworkGraphGPUParams make_gpu_params(const GPUConfigSpec& s,
                                             std::size_t N,
                                             std::size_t max_edges) {
    return NetworkGraphGPUParams{
        s.has_local_edges,
        s.layout_local_in, s.features_local_in,
        s.layout_local_out, s.features_local_out,
        s.layout_distant_in, s.features_distant_in,
        s.layout_distant_out, s.features_distant_out,
        max_edges, N
    };
}

static const GPUConfigSpec kGPUConfigs[] = {
    // Config 2 -- active production config (NetworkGPUType::MEMORY_POOL in NetworkGraph.h): unweighted
    // MemoryPool everywhere (edge count is expanded by |weight| instead of storing weights directly).
    {
        "MemoryPool_Unweighted_MemoryPool_MemoryPool",
        true,
        LayoutType::MemoryPool,
        Features{
            false,
            false,
        },
        LayoutType::MemoryPool,
        Features{
            false,
            false,
        },
        LayoutType::MemoryPool,
        Features{
            false,
            true,
        },
        LayoutType::MemoryPool,
        Features{
            false,
            true,
        },
    },
    // Config 3 – MemoryPool local edges with deletion flag
    {
        "MemoryPool_MemoryPool_deleted_MemoryPool_MemoryPool",
        true,
        LayoutType::MemoryPool,
        Features{
            true,
            false,
        },
        LayoutType::MemoryPool,
        Features{
            true,
            false,
        },
        LayoutType::MemoryPool,
        Features{ true, true },
        LayoutType::MemoryPool,
        Features{
            true,
            true,
        },
    },
    // Config 4 – no local edges, weighted MemoryPool distant
    {
        "NoLocal_MemoryPool_MemoryPool",
        false,
        LayoutType::MemoryPool,
        Features{},
        LayoutType::MemoryPool,
        Features{},
        LayoutType::MemoryPool,
        Features{ true, true },
        LayoutType::MemoryPool,
        Features{
            true,
            true,
        },
    },
};

class NetworkGraphGPUConfigTest
    : public RelearnTest,
      public testing::WithParamInterface<GPUConfigSpec> { };

TEST_P(NetworkGraphGPUConfigTest, EmptyEdges_HandlesNonNull) {
    const GPUConfigSpec& cfg = GetParam();
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;

    NetworkGraphGPUBase gpu(R, my_rank);
    gpu.init(make_gpu_params(cfg, N, 64));

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_in(N);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_out(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_in(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_out(N);
    EXPECT_NO_THROW(gpu.update_edges(local_in, local_out, distant_in, distant_out));

    const NetworkHandle h = gpu.get_handle();
    EXPECT_NE(h.incoming_distant_handle.view_impl, nullptr);
    EXPECT_NE(h.outgoing_distant_handle.view_impl, nullptr);
    if (cfg.has_local_edges) {
        EXPECT_NE(h.incoming_local_handle.view_impl, nullptr);
        EXPECT_NE(h.outgoing_local_handle.view_impl, nullptr);
    } else {
        EXPECT_EQ(h.incoming_local_handle.layout_type, LayoutType::Dummy);
        EXPECT_EQ(h.outgoing_local_handle.layout_type, LayoutType::Dummy);
        EXPECT_NE(h.incoming_local_handle.view_impl, nullptr);
        EXPECT_NE(h.outgoing_local_handle.view_impl, nullptr);
    }
}

TEST_P(NetworkGraphGPUConfigTest, LocalAndDistantEdges_HandlesNonNull) {
    const GPUConfigSpec& cfg = GetParam();
    constexpr neuron_id_t N = 6;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 1;

    NetworkGraphGPUBase gpu(R, my_rank);
    gpu.init(make_gpu_params(cfg, N, 128));

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_in(N);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_out(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_in(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_out(N);

    local_in[0].emplace_back(NeuronID{ 2 }, 1);
    local_in[1].emplace_back(NeuronID{ 3 }, -1);
    local_out[2].emplace_back(NeuronID{ 0 }, 1);
    local_out[3].emplace_back(NeuronID{ 1 }, -1);
    distant_in[4].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 10 } }, 1);
    distant_out[5].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 10 } }, -1);

    EXPECT_NO_THROW(gpu.update_edges(local_in, local_out, distant_in, distant_out));

    const NetworkHandle h = gpu.get_handle();
    EXPECT_NE(h.incoming_distant_handle.view_impl, nullptr);
    EXPECT_NE(h.outgoing_distant_handle.view_impl, nullptr);
    if (cfg.has_local_edges) {
        EXPECT_NE(h.incoming_local_handle.view_impl, nullptr);
        EXPECT_NE(h.outgoing_local_handle.view_impl, nullptr);
    }
}

TEST_P(NetworkGraphGPUConfigTest, Rebuild_MultipleTimes_HandlesStillValid) {
    const GPUConfigSpec& cfg = GetParam();
    constexpr neuron_id_t N = 6;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;

    NetworkGraphGPUBase gpu(R, my_rank);
    gpu.init(make_gpu_params(cfg, N, 128));

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_in(N);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_out(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_in(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_out(N);
    local_in[0].emplace_back(NeuronID{ 3 }, 1);
    local_out[0].emplace_back(NeuronID{ 3 }, 1);
    distant_in[2].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 7 } }, 1);
    distant_out[2].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 7 } }, -1);

    gpu.update_edges(local_in, local_out, distant_in, distant_out);

    EXPECT_NO_THROW(gpu.rebuild());
    EXPECT_NO_THROW(gpu.rebuild());

    const NetworkHandle h = gpu.get_handle();
    EXPECT_NE(h.incoming_distant_handle.view_impl, nullptr);
    EXPECT_NE(h.outgoing_distant_handle.view_impl, nullptr);
    if (cfg.has_local_edges) {
        EXPECT_NE(h.incoming_local_handle.view_impl, nullptr);
        EXPECT_NE(h.outgoing_local_handle.view_impl, nullptr);
    }
}

TEST_P(NetworkGraphGPUConfigTest, MultipleNeurons_SymmetricLocalEdges) {
    // Insert local edges both in and out for every neuron pair,
    // then verify the graph survives rebuild without crashing.
    const GPUConfigSpec& cfg = GetParam();
    constexpr neuron_id_t N = 5;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    NetworkGraphGPUBase gpu(R, my_rank);
    gpu.init(make_gpu_params(cfg, N, 64));

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_in(N);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_out(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_in(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_out(N);

    // neuron i ↔ neuron (i+1) % N
    for (neuron_id_t i = 0; i < N; ++i) {
        const neuron_id_t j = (i + 1) % N;
        local_in[i].emplace_back(NeuronID{ j }, 1);
        local_out[i].emplace_back(NeuronID{ j }, -1);
    }

    EXPECT_NO_THROW(gpu.update_edges(local_in, local_out, distant_in, distant_out));
    EXPECT_NO_THROW(gpu.rebuild());

    const NetworkHandle h = gpu.get_handle();
    EXPECT_NE(h.incoming_distant_handle.view_impl, nullptr);
    EXPECT_NE(h.outgoing_distant_handle.view_impl, nullptr);
}

// ────────────────────────────────────────────────────────────────────────────
// Device-query dispatch helpers
//
// Each helper dispatches to the right CUDA function based on the EdgeHandle's
// layout type, so parametrized tests can query any of the four handles without
// knowing the concrete view type.
// ────────────────────────────────────────────────────────────────────────────

static std::vector<std::size_t> query_device_sizes(const EdgeHandle& h,
                                                   neuron_id_t N) {
    switch (h.layout_type) {
    case LayoutType::MemoryPool:
        return device_memory_pool_view_sizes(h.view_impl, N);
    default:
        return std::vector<std::size_t>(N, 0);
    }
}

static std::vector<HostEdge> query_device_edges(const EdgeHandle& h,
                                                neuron_id_t nid) {
    switch (h.layout_type) {
    case LayoutType::MemoryPool:
        return device_memory_pool_view_iterate_edges(h.view_impl, nid);
    default:
        return {}; // OnlyOutgoing and Dummy: no per-edge iteration
    }
}

static bool can_iterate_edges(LayoutType lt) {
    return lt == LayoutType::MemoryPool;
}

// ────────────────────────────────────────────────────────────────────────────
// Parametrized device-query tests – all 4 configurations
// ────────────────────────────────────────────────────────────────────────────

// After a fully empty update_edges, every neuron's edge count should be 0 on
// the device for all four handles.
TEST_P(NetworkGraphGPUConfigTest, DeviceQuery_AllHandles_EmptySizesZero) {
    const GPUConfigSpec& cfg = GetParam();
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    NetworkGraphGPUBase gpu(R, my_rank);
    gpu.init(make_gpu_params(cfg, N, 64));

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_in(N);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_out(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_in(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_out(N);
    gpu.update_edges(local_in, local_out, distant_in, distant_out);

    const NetworkHandle h = gpu.get_handle();
    for (const EdgeHandle& handle :
         { h.incoming_local_handle, h.incoming_distant_handle,
           h.outgoing_local_handle, h.outgoing_distant_handle }) {
        const auto sizes = query_device_sizes(handle, N);
        ASSERT_EQ(sizes.size(), N);
        for (neuron_id_t i = 0; i < N; ++i) {
            EXPECT_EQ(sizes[i], 0U) << cfg.name << " neuron " << i;
        }
    }
}

// Add one local incoming edge (neuron 0 ← neuron 1) and one local outgoing edge
// (neuron 1 → neuron 0), then verify the counts and edge data on device.
TEST_P(NetworkGraphGPUConfigTest, DeviceQuery_LocalEdges_CountAndIterationCorrect) {
    const GPUConfigSpec& cfg = GetParam();
    if (!cfg.has_local_edges) {
        return;
    }

    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;

    NetworkGraphGPUBase gpu(R, my_rank);
    gpu.init(make_gpu_params(cfg, N, 64));

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_in(N);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_out(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_in(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_out(N);
    local_in[0].emplace_back(NeuronID{ 1 }, 1);  // neuron 0 receives from neuron 1
    local_out[1].emplace_back(NeuronID{ 0 }, 1); // neuron 1 sends to neuron 0
    gpu.update_edges(local_in, local_out, distant_in, distant_out);
    gpu.rebuild();

    const NetworkHandle h = gpu.get_handle();

    // ── incoming_local ──────────────────────────────────────────────────────
    {
        const auto sizes = query_device_sizes(h.incoming_local_handle, N);
        ASSERT_EQ(sizes.size(), N);
        EXPECT_EQ(sizes[0], 1U) << cfg.name << " incoming_local size[0]";
        for (neuron_id_t i = 1; i < N; ++i) {
            EXPECT_EQ(sizes[i], 0U) << cfg.name << " incoming_local size[" << i << "]";
        }

        if (can_iterate_edges(cfg.layout_local_in)) {
            const auto edges = query_device_edges(h.incoming_local_handle, 0);
            ASSERT_EQ(edges.size(), 1U) << cfg.name << " incoming_local edges";
            EXPECT_EQ(edges[0].other_neuron, 1U) << cfg.name;
            EXPECT_EQ(edges[0].weight, weight_t{ 1 }) << cfg.name;
            // MemoryPoolView::get_other_rank() falls back to the view's own my_rank when no rank
            // storage is present (has_ranks=false) -- all local edges are on this rank anyway.
            EXPECT_EQ(edges[0].other_rank, my_rank) << cfg.name;
        }
    }

    // ── outgoing_local ──────────────────────────────────────────────────────
    {
        const auto sizes = query_device_sizes(h.outgoing_local_handle, N);
        ASSERT_EQ(sizes.size(), N);
        EXPECT_EQ(sizes[1], 1U) << cfg.name << " outgoing_local size[1]";
        for (neuron_id_t i = 0; i < N; ++i) {
            if (i != 1U) {
                EXPECT_EQ(sizes[i], 0U) << cfg.name << " outgoing_local size[" << i << "]";
            }
        }

        if (can_iterate_edges(cfg.layout_local_out)) {
            const auto edges = query_device_edges(h.outgoing_local_handle, 1);
            ASSERT_EQ(edges.size(), 1U) << cfg.name << " outgoing_local edges";
            EXPECT_EQ(edges[0].other_neuron, 0U) << cfg.name;
            EXPECT_EQ(edges[0].weight, weight_t{ 1 }) << cfg.name;
        }
    }
}

// Add one distant incoming and one distant outgoing edge, then verify via device.
// All configurations have weighted distant edges with rank storage.
TEST_P(NetworkGraphGPUConfigTest, DeviceQuery_DistantEdges_CountAndIterationCorrect) {
    const GPUConfigSpec& cfg = GetParam();
    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 3;
    constexpr rank_t my_rank = 0;

    NetworkGraphGPUBase gpu(R, my_rank);
    gpu.init(make_gpu_params(cfg, N, 64));

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_in(N);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_out(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_in(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_out(N);
    distant_in[2].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 5 } }, 1);
    distant_out[2].emplace_back(RankNeuronId{ mpiPP::MPIRank{ 1 }, NeuronID{ 7 } }, 1);
    gpu.update_edges(local_in, local_out, distant_in, distant_out);
    gpu.rebuild();

    const NetworkHandle h = gpu.get_handle();

    // ── incoming_distant ────────────────────────────────────────────────────
    {
        const auto sizes = query_device_sizes(h.incoming_distant_handle, N);
        ASSERT_EQ(sizes.size(), N);
        EXPECT_EQ(sizes[2], 1U) << cfg.name << " incoming_distant size[2]";
        for (neuron_id_t i = 0; i < N; ++i) {
            if (i != 2U) {
                EXPECT_EQ(sizes[i], 0U) << cfg.name << " incoming_distant size[" << i << "]";
            }
        }

        if (can_iterate_edges(cfg.layout_distant_in)) {
            const auto edges = query_device_edges(h.incoming_distant_handle, 2);
            ASSERT_EQ(edges.size(), 1U) << cfg.name;
            EXPECT_EQ(edges[0].other_neuron, 5U) << cfg.name;
            EXPECT_EQ(edges[0].other_rank, rank_t{ 1 }) << cfg.name;
            EXPECT_EQ(edges[0].weight, weight_t{ 1 }) << cfg.name;
        }
    }

    // ── outgoing_distant ────────────────────────────────────────────────────
    {
        const auto sizes = query_device_sizes(h.outgoing_distant_handle, N);
        ASSERT_EQ(sizes.size(), N);
        EXPECT_EQ(sizes[2], 1U) << cfg.name << " outgoing_distant size[2]";
        for (neuron_id_t i = 0; i < N; ++i) {
            if (i != 2U) {
                EXPECT_EQ(sizes[i], 0U) << cfg.name << " outgoing_distant size[" << i << "]";
            }
        }

        if (can_iterate_edges(cfg.layout_distant_out)) {
            const auto edges = query_device_edges(h.outgoing_distant_handle, 2);
            ASSERT_EQ(edges.size(), 1U) << cfg.name;
            EXPECT_EQ(edges[0].other_neuron, 7U) << cfg.name;
            EXPECT_EQ(edges[0].other_rank, rank_t{ 1 }) << cfg.name;
            EXPECT_EQ(edges[0].weight, weight_t{ 1 }) << cfg.name;
        }
    }
}

// Add local edges with weight=254 (>int8_t max=127) and verify that the stored
// entry count on-device reflects the per-layout splitting semantics:
//   - OnlyOutgoing incoming: only stores a count, never splits → 1
//   - Unweighted MemoryPool (no weight storage): GPUEdges stores abs(weight) copies → W entries.
//   - Weighted MemoryPool: ceil(254/127) = 2 entries
TEST_P(NetworkGraphGPUConfigTest, DeviceQuery_LocalEdges_LargeWeight_EntrySplitCount) {
    const GPUConfigSpec& cfg = GetParam();
    if (!cfg.has_local_edges) {
        return;
    }

    constexpr neuron_id_t N = 4;
    constexpr rank_t R = 2;
    constexpr rank_t my_rank = 0;
    constexpr RelearnTypes::plastic_synapse_weight W = 254; // ceil(254/127) = 2

    NetworkGraphGPUBase gpu(R, my_rank);
    gpu.init(make_gpu_params(cfg, N, 64));

    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_in(N);
    std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>> local_out(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_in(N);
    std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>> distant_out(N);
    local_in[0].emplace_back(NeuronID{ 1 }, W);
    local_out[1].emplace_back(NeuronID{ 0 }, W);
    gpu.update_edges(local_in, local_out, distant_in, distant_out);

    const NetworkHandle h = gpu.get_handle();

    // incoming_local: number of stored entries depends on layout
    {
        const auto sizes = query_device_sizes(h.incoming_local_handle, N);
        // OnlyOutgoing uses incoming_count = synapse count → always 1 per logical edge.
        // Unweighted MemoryPool: GPUEdges stores abs(weight) copies → W entries.
        // Weighted MemoryPool: ceil(W / INT8_MAX) = ceil(254/127) = 2 entries.
        std::size_t expected = 0;
        if (!cfg.features_local_in.has_weights) {
            expected = static_cast<std::size_t>(std::abs(static_cast<int>(W)));
        } else {
            expected = 2U;
        }
        EXPECT_EQ(sizes[0], expected) << cfg.name << " incoming_local";
    }

    // outgoing_local: entry count depends on whether outgoing storage is weighted, same as incoming.
    {
        const auto sizes = query_device_sizes(h.outgoing_local_handle, N);
        const std::size_t expected = cfg.features_local_out.has_weights
                                         ? 2U
                                         : static_cast<std::size_t>(std::abs(static_cast<int>(W)));
        EXPECT_EQ(sizes[1], expected) << cfg.name << " outgoing_local";
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllNetworkGPUConfigs,
    NetworkGraphGPUConfigTest,
    testing::ValuesIn(kGPUConfigs),
    [](const testing::TestParamInfo<GPUConfigSpec>& param_info) {
        return std::string(param_info.param.name);
    });

#endif // RELEARN_CUDA_ENABLED
