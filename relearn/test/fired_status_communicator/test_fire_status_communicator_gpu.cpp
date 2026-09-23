/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_fire_status_communicator_gpu.h"

#ifdef RELEARN_CUDA_ENABLED
#include "Config.h"

#include "cuda/firing/FireStatusCommunicatorGPUUncompressed.h"
#include "neurons/NetworkGraph.h"
#include "neurons/input/ActivityInput.h"
#include "neurons/input/SynapticEquallyWeightedActivityInput.h"
#include "util/NeuronIDRange.h"

#include "factory/extra_info/extra_info_factory.h"
#include "factory/network_graph/network_graph_factory.h"

#include <cpp-utility/data/displacement.hpp>

// Sums the local and distant contributions get_input_arr_const() reports per neuron. Declared to
// take the base ActivityInput& (get_input_arr_const() is protected on the concrete
// SynapticActivityInput subclasses -- only ActivityInput exposes it publicly, matching the access
// pattern production code uses, e.g. NeuronModel::get_d_input_const()'s
// act_input->get_input_arr_const()[0]->get_device_ptr_const()).
static std::vector<ActivityInput::activity_type> combined_input(const ActivityInput& input, const std::size_t number_neurons) {
    // update_input(step, stream_wrapper) only enqueues work on that stream; unlike the
    // stream-less update_input(step) convenience overload, it does not itself synchronize before
    // returning, so the device work must be finished before host() below can read valid data.
    cudaDeviceSynchronize_bridge();
    std::vector<ActivityInput::activity_type> result(number_neurons, ActivityInput::activity_type{ 0 });
    for (const auto* arr : input.get_input_arr_const()) {
        const auto host_data = arr->host();
        for (std::size_t i = 0; i < number_neurons; ++i) {
            result[i] += host_data[i];
        }
    }
    return result;
}

template <typename T>
void test_fcm(std::shared_ptr<FiredStatusCommunicatorGPU<T>> fcm, std::function<void(std::shared_ptr<FiredStatusCommunicatorGPU<T>>)> simulate_transfer,
              NetworkGPUType graph_type = NetworkGPUType::MEMORY_POOL, bool do_binary_search = false) {
    // finalize() joins the background MPI thread; std::thread's destructor calls std::terminate()
    // if still joinable, so this must run on every exit path (including an ASSERT_EQ early return).
    const struct FinalizeGuard {
        std::shared_ptr<FiredStatusCommunicatorGPU<T>> fcm;
        ~FinalizeGuard() { fcm->finalize(); }
    } finalize_guard{ fcm };

    // update_distant_input() (the cross-rank path, exercised below via synapses to rank 2) picks
    // between SpikeMode::BinarySearch+NetworkMode::Sorted and SpikeMode::Set+NetworkMode::Default
    // based on this global flag (see SynapticEquallyWeightedActivityInput.cpp) -- restore it on
    // exit so this test's choice doesn't leak into other tests sharing the process.
    const auto previous_do_binary_search = Config::do_binary_search;
    Config::do_binary_search = do_binary_search;
    const struct BinarySearchGuard {
        bool previous;
        ~BinarySearchGuard() { Config::do_binary_search = previous; }
    } binary_search_guard{ previous_do_binary_search };

    const auto number_neurons = 10;
    const auto number_ranks = 4;
    const auto my_rank = 1;

    auto fsr = std::make_shared<FiredStatusRecorder>();
    auto network_graph = std::make_shared<NetworkGraph>(mpiPP::MPIRank{ my_rank }, number_ranks);
    network_graph->init(number_neurons, graph_type);
    network_graph->rebuild();

    fcm->init(number_neurons);
    fsr->init(number_neurons);
    fcm->set_network_graph(network_graph);
    fcm->set_fired_status_recorder(fsr);

    // Declared as the base ActivityInput pointer: get_input_arr()/get_input_arr_const() are
    // protected on the concrete SynapticActivityInput subclasses (only ActivityInput exposes them
    // publicly), matching the access pattern production code uses (e.g. NeuronModel::get_d_input()/
    // get_d_input_const()).
    std::unique_ptr<ActivityInput> input = std::make_unique<SynapticEquallyWeightedActivityInput>(number_ranks, fcm, 1.0F);
    auto extra_infos = NeuronsExtraInfoFactory::construct_extra_info();
    extra_infos->init(number_neurons);
    fcm->set_extra_infos(extra_infos);
    input->set_extra_infos(extra_infos);
    input->set_network_graph(network_graph);
    input->init(number_neurons);
    auto stream_wrapper = std::make_shared<StreamWrapper>();

    // No connections, nothing fired
    fcm->commit_local_fired_status(1);
    simulate_transfer(fcm);
    input->update_input(1, stream_wrapper);
    const auto input_arr = combined_input(*input, number_neurons);
    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        ASSERT_EQ(input_arr[neuron_id], 0);
    }

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        const auto fired = neuron_id % 3 == 0 ? FiredStatus::Fired : FiredStatus::Inactive;
        fsr->set_fired(neuron_id, fired);
    }

    // No connections, some neurons fired
    fcm->commit_local_fired_status(2);
    simulate_transfer(fcm);
    input->update_input(2, stream_wrapper);
    const auto input_arr2 = combined_input(*input, number_neurons);
    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        ASSERT_EQ(input_arr2[neuron_id], 0);
    }

    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        fsr->set_fired(neuron_id, FiredStatus::Inactive);
    }
    network_graph->add_synapse(PlasticDistantOutSynapse{ RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 1 } }, NeuronID{ 4 }, 1 });
    network_graph->add_synapse(PlasticDistantOutSynapse{ RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 2 } }, NeuronID{ 5 }, -1 });
    network_graph->add_synapse(PlasticDistantOutSynapse{ RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 3 } }, NeuronID{ 7 }, 1 });
    network_graph->add_synapse(PlasticDistantOutSynapse{ RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 4 } }, NeuronID{ 8 }, -1 });
    // We need symmetric edges otherwise our tests dont work
    network_graph->add_synapse(PlasticDistantInSynapse{ NeuronID{ 1 }, RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 4 } }, 1 });
    network_graph->add_synapse(PlasticDistantInSynapse{ NeuronID{ 2 }, RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 5 } }, -1 });
    network_graph->add_synapse(PlasticDistantInSynapse{ NeuronID{ 3 }, RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 7 } }, 1 });
    network_graph->add_synapse(PlasticDistantInSynapse{ NeuronID{ 4 }, RankNeuronId{ mpiPP::MPIRank{ 2 }, NeuronID{ 8 } }, -1 });
    network_graph->sync_with_gpu();
    network_graph->rebuild();

    // Some connections, nothing fired
    fcm->commit_local_fired_status(3);
    simulate_transfer(fcm);
    input->update_input(3, stream_wrapper);
    const auto input_arr3 = combined_input(*input, number_neurons);
    for (const auto neuron_id : NeuronIDRange::range_id(number_neurons)) {
        ASSERT_EQ(input_arr3[neuron_id], 0);
    }

    // Connections and fire
    fsr->set_fired(4, FiredStatus::Fired);
    fsr->set_fired(6, FiredStatus::Fired);
    fsr->set_fired(8, FiredStatus::Fired);
    fcm->commit_local_fired_status(4);
    simulate_transfer(fcm);

    input->update_input(4, stream_wrapper);
    const auto input_arr4 = combined_input(*input, number_neurons);
    ASSERT_EQ(input_arr4[0], 0);
    ASSERT_EQ(input_arr4[1], 1);
    ASSERT_EQ(input_arr4[2], 0);
    ASSERT_EQ(input_arr4[3], 0);
    ASSERT_EQ(input_arr4[4], -1);
    ASSERT_EQ(input_arr4[5], 0);
    ASSERT_EQ(input_arr4[6], 0);
    ASSERT_EQ(input_arr4[7], 0);
    ASSERT_EQ(input_arr4[8], 0);
    ASSERT_EQ(input_arr4[9], 0);
}

template <typename T>
void simulate_transfer_fun(std::shared_ptr<FiredStatusCommunicatorGPU<T>> fcm) {
    const auto& out = fcm->get_outgoing_data();
    const auto& out_sizes = fcm->get_outgoing_data_size_per_rank();
    const auto begin = out_sizes[0] + out_sizes[1];
    const auto end = begin + out_sizes[2];
    const auto h_out = out.get_device_data();
    const auto h_in = std::vector<T>(h_out.data() + begin, h_out.data() + end);
    DeviceArray<T> incoming_bitvector(h_in);
    const auto displ = utility::calculate_offsets(std::span<const int>(out_sizes.cbegin(), out_sizes.cend()));
    DeviceArray<int> incoming_displ{ displ };
    fcm->set_incoming_data(std::move(incoming_bitvector), std::move(incoming_displ));
}

TEST_F(FiredStatusCommunicatorGPUTest, testUncompressed) {
    const auto number_ranks = 4;
    const auto my_rank = 1;
    auto fcm = std::make_shared<FireStatusCommunicatorGPUUncompressed>(mpiPP::MPIRank{ my_rank }, number_ranks);
    test_fcm<CudaConfig::number_neurons_type>(fcm, &simulate_transfer_fun<CudaConfig::number_neurons_type>);
}

// SynapticEquallyWeightedActivityInput's update_local_input always uses the same fixed
// LaunchConfig (LocalVector spikes, Default network, Weighted), already covered above; but the
// network-graph *layout* it dispatches over inside launch() (MemoryPool/OnlyOutgoing, wide vs. narrow ids,
// weighted vs. unweighted storage) depends on the actual NetworkGPUType the graph was built with.
// Running the exact same numeric scenario across every NetworkGPUType exercises those layout
// branches without needing to hand-construct LaunchHandles directly.
class FiredStatusCommunicatorGPUNetworkTypeTest : public RelearnTest, public ::testing::WithParamInterface<NetworkGPUType> { };

TEST_P(FiredStatusCommunicatorGPUNetworkTypeTest, testUncompressedAcrossNetworkGraphTypes) {
    const auto number_ranks = 4;
    const auto my_rank = 1;
    auto fcm = std::make_shared<FireStatusCommunicatorGPUUncompressed>(mpiPP::MPIRank{ my_rank }, number_ranks);
    test_fcm<CudaConfig::number_neurons_type>(fcm, &simulate_transfer_fun<CudaConfig::number_neurons_type>, GetParam());
}

INSTANTIATE_TEST_SUITE_P(AllSupportedNetworkGraphTypes, FiredStatusCommunicatorGPUNetworkTypeTest,
                         ::testing::Values(NetworkGPUType::MEMORY_POOL, NetworkGPUType::MEMORY_POOL_WEIGHTED),
                         [](const ::testing::TestParamInfo<NetworkGPUType>& param_info) {
                             switch (param_info.param) {
                             case NetworkGPUType::MEMORY_POOL:
                                 return std::string("MEMORY_POOL");
                             case NetworkGPUType::MEMORY_POOL_WEIGHTED:
                                 return std::string("MEMORY_POOL_WEIGHTED");
                             default:
                                 return std::string("Unknown");
                             }
                         });

// update_distant_input (the cross-rank path) picks between SpikeMode::BinarySearch and
// SpikeMode::Set based on Config::do_binary_search; the suite above only exercises the default
// (false, i.e. Set). This covers the BinarySearch branch.
TEST_F(FiredStatusCommunicatorGPUTest, testUncompressedWithBinarySearchDistantLookup) {
    const auto number_ranks = 4;
    const auto my_rank = 1;
    auto fcm = std::make_shared<FireStatusCommunicatorGPUUncompressed>(mpiPP::MPIRank{ my_rank }, number_ranks);
    test_fcm<CudaConfig::number_neurons_type>(fcm, &simulate_transfer_fun<CudaConfig::number_neurons_type>,
                                              NetworkGPUType::MEMORY_POOL, /*do_binary_search=*/true);
}

#endif