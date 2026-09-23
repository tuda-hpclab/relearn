#pragma once
#include "algorithm/Connector.h"
#include "algorithm/Internal/ExchangingAlgorithm.h"
#include "cuda/mpi/MPICuda.h"
#include "cuda/util/LinearizedTreeDeviceHandle.h"
#include "cuda/util/Util.h"

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

struct TargetNeuronSearchResult;
struct TargetNeuronSearchResultBothSignalTypes;
/**
 * This class manages the exchange of requests and responses, and their distribution on all MPI ranks
 *      It connects from axons to dendrites
 * @tparam RequestType The type of creation requests
 * @tparam ResponseType The type of creation responses
 */
template <typename RequestType, typename ResponseType>
class ForwardGPUAlgorithm : public Algorithm {
public:
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new object
     */
    ForwardGPUAlgorithm()
        : Algorithm() { }

    [[nodiscard]] ConnectivityUpdateResult update_connectivity([[maybe_unused]] const number_neurons_type number_neurons) final {
#ifdef RELEARN_CUDA_ENABLED
        Timers::start(TimerRegion::CREATE_SYNAPSES);

        auto total_number_created_synapses = 0UL;

        Timers::start(TimerRegion::FIND_TARGET_NEURONS);
        const auto [synapse_creation_requests_exc, synapse_creation_requests_inh] = find_target_neurons(number_neurons);
        Timers::stop_and_add(TimerRegion::FIND_TARGET_NEURONS);

        auto helper = [&total_number_created_synapses, this](const target_neuron_result_type& synapse_creation_requests, const SignalType signal_type) {
            Timers::start(TimerRegion::EXCHANGE_CREATION_REQUESTS);
            const auto number_ranks = mpiPP::MPIInfo::get_number_ranks_cast();

            const auto& [h_sizes, source_ids_out, source_positions_out, target_ids_out] = synapse_creation_requests;

            const auto h_source_ids_out = source_ids_out.get_device_data();
            const auto h_target_ids_out = target_ids_out.get_device_data();

            const auto& [recv_source_ids, recv_sizes_source_id, recv_displ_source_id] = cuda_all_to_all(source_ids_out,
                                                                                                        h_sizes,
                                                                                                        number_ranks);
            const auto& [recv_source_pos, recv_sizes_source_pos, recv_displ_source_pos] = cuda_all_to_all(
                source_positions_out, h_sizes, number_ranks);
            const auto& [recv_target_ids, recv_sizes_target_ids, recv_displ_target_ids] = cuda_all_to_all(
                target_ids_out, h_sizes, number_ranks);

            Timers::stop_and_add(TimerRegion::EXCHANGE_CREATION_REQUESTS);

            Timers::start(TimerRegion::PROCESS_CREATION_REQUESTS);
            auto out = process_requests(recv_sizes_source_id, recv_displ_source_id, recv_source_ids,
                                        recv_source_pos, recv_target_ids, signal_type);
            auto& [responses_outgoing, target_ids_outgoing, _number_created_synapses] = out;
            total_number_created_synapses += _number_created_synapses;
            Timers::stop_and_add(TimerRegion::PROCESS_CREATION_REQUESTS);

            Timers::start(TimerRegion::CREATE_CREATION_RESPONSES);
            auto res1 = cuda_all_to_all<SynapseCreationResponse>(responses_outgoing, recv_sizes_source_id,
                                                                 number_ranks);
            auto& [resp_incoming, resp_sizes, resp_displ] = res1;
            auto res2 = cuda_all_to_all(target_ids_outgoing, recv_sizes_source_id, number_ranks);
            auto& [resp_target_incoming, resp_target_sizes, resp_target_displ] = res2;
            Timers::stop_and_add(TimerRegion::CREATE_CREATION_RESPONSES);

            Timers::start(TimerRegion::PROCESS_CREATION_RESPONSES);
            process_responses(resp_incoming, source_ids_out, resp_target_incoming, resp_sizes,
                              resp_displ, signal_type);
            Timers::stop_and_add(TimerRegion::PROCESS_CREATION_RESPONSES);
        };

        helper(synapse_creation_requests_exc, SignalType::Excitatory);
        helper(synapse_creation_requests_inh, SignalType::Inhibitory);

        cudaDeviceSynchronize_bridge();

        Timers::stop_and_add(TimerRegion::CREATE_SYNAPSES);

        return {
            total_number_created_synapses,
            PlasticLocalSynapses{},
            PlasticDistantInSynapses{},
            PlasticDistantOutSynapses{},
        };
#else
        CUDA_NOT_SUPPORTED
#endif
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        Algorithm::record_memory_footprint(footprint);

        const auto my_footprint = sizeof(*this) - sizeof(Algorithm);
        footprint->emplace("ForwardAlgorithm", my_footprint);
    }

protected:
    using target_neuron_result_type = TargetNeuronSearchResult;
    using target_neuron_both_result_type = TargetNeuronSearchResultBothSignalTypes;

    /**
     * @brief Returns a collection of proposed synapse creations for each neuron
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. Does not send those requests to the other MPI ranks.
     */

    [[nodiscard]] virtual target_neuron_both_result_type find_target_neurons(number_neurons_type number_neurons) = 0;

    virtual ProcessRequestsAwareResult
    process_requests([[maybe_unused]] const std::vector<int>& counts_full, [[maybe_unused]] const std::vector<int>& offsets_full, [[maybe_unused]] const DeviceArray<CudaConfig::number_neurons_type>& source_ids, [[maybe_unused]] const DeviceArray<SimpleVec3d>& source_positions, [[maybe_unused]] const DeviceArray<CudaConfig::number_neurons_type>& target_ids,
                     [[maybe_unused]] const SignalType dendrite_type_needed) = 0;

    virtual void
    process_responses([[maybe_unused]] const DeviceArray<SynapseCreationResponse>& responses, [[maybe_unused]] const DeviceArray<CudaConfig::number_neurons_type>& source_ids, [[maybe_unused]] const DeviceArray<CudaConfig::number_neurons_type>& target_ids, [[maybe_unused]] const std::vector<int>& sizes, [[maybe_unused]] const std::vector<int>& offset, [[maybe_unused]] SignalType signal_type) = 0;
};