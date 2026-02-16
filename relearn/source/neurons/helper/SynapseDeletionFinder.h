#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"
#include "Types3.h"

#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"

#include "mpi-wrapper/MPIRank.h"

#include <boost/dynamic_bitset.hpp>
#include <fmt/ostream.h>

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

class FiredStatusRecorder;
class NetworkGraph;
class NeuronsExtraInfo;
class SynapticElements;

/**
 * This enums lists all types of synapse deletion finders
 */
enum class SynapseDeletionFinderType : char {
    Random,
    InverseLength,
    CoActivation,
};

/**
 * @brief Pretty-prints the synapse deletion finder type to the chosen stream
 * @param out The stream to which to print the synapse deletion finder
 * @param synapse_deletion_type The synapse deletion finder to print
 * @return The argument out, now altered with the synapse deletion finder
 */
inline std::ostream& operator<<(std::ostream& out, const SynapseDeletionFinderType& synapse_deletion_type) {
    if (synapse_deletion_type == SynapseDeletionFinderType::Random) {
        return out << "Random";
    }

    if (synapse_deletion_type == SynapseDeletionFinderType::InverseLength) {
        return out << "InverseLength";
    }

    if (synapse_deletion_type == SynapseDeletionFinderType::CoActivation) {
        return out << "CoActivation";
    }

    return out;
}

template <>
struct fmt::formatter<SynapseDeletionFinderType> : ostream_formatter { };

/**
 * This class encapsulates the logic of finding and deleting synapses
 * based on the synaptic elements. It provides the communication via MPI
 * and other house keeping, as well as a virtual method to implement.
 */
class SynapseDeletionFinder {
public:
    SynapseDeletionFinder() = default;

    SynapseDeletionFinder(const SynapseDeletionFinder&) = default;
    SynapseDeletionFinder& operator=(const SynapseDeletionFinder&) = default;

    SynapseDeletionFinder(SynapseDeletionFinder&&) = delete;
    SynapseDeletionFinder& operator=(SynapseDeletionFinder&&) = delete;

    virtual ~SynapseDeletionFinder() = default;

    /**
     * @brief Sets the network graph that stores the synapses
     * @param ng The new network graph, must not be empty
     * @exception Throws a RelearnException if ng is empty
     */
    void set_network_graph(std::shared_ptr<NetworkGraph> ng) {
        const auto full = ng != nullptr;
        RelearnException::check(full, "SynapseDeletionFinder::set_network_graph: The network graph is empty");

        network_graph = std::move(ng);
    }

    void set_synaptic_elements(std::shared_ptr<SynapticElements> synaptic_elements_ptr) {
        const auto full = synaptic_elements_ptr != nullptr;
        RelearnException::check(full, "SynapseDeletionFinder::set_synaptic_elements: The synaptic elements is empty");

        synaptic_elements = std::move(synaptic_elements_ptr);
    }

    /**
     * @brief Sets the extra information
     * @param new_extra_info The extra information, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info);

    /**
     * @brief Sets the fired status recorder
     * @param new_fired_status_recorder The fired status recorder, must not be empty
     * @exception Throws a RelearnException if new_fired_status_recorder is empty
     */
    void set_fired_status_recorder(std::shared_ptr<FiredStatusRecorder> new_fired_status_recorder);

    /**
     * @brief Commits the updates for the synaptic elements, deletes synapses in the network graph,
     *      exchanges the deletions between MPI ranks, and commits the deletions from other ranks as well
     * @return The number of deleted synapses that are initiated by (1) the local axons and (2) the local dendrites
     */
    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> delete_synapses();

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        const auto my_footprint = sizeof(*this);
        footprint->emplace("SynapseDeletionFinder", my_footprint);
    }

protected:
    [[nodiscard]] virtual RelearnTypes::comm_map_deletion<SynapseDeletionRequest> find_synapses_to_delete(ElementType element_type, std::span<const SignalType> signal_types,
                                                                                                          std::span<const unsigned int> number_deletions);

    [[nodiscard]] virtual RelearnTypes::comm_map_deletion<SynapseDeletionRequest> find_synapses_to_delete(ElementType element_type, SignalType signal_types,
                                                                                                          std::span<const unsigned int> number_deletions);

    [[nodiscard]] virtual std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) = 0;

    [[nodiscard]] std::uint64_t commit_deletions(const RelearnTypes::comm_map_deletion<SynapseDeletionRequest>& deletions, mpiPP::MPIRank my_rank);

    [[nodiscard]] std::vector<RankNeuronId> register_synapses(NeuronID neuron_id, ElementType element_type, SignalType signal_type);

    [[nodiscard]] std::vector<mpiPP::MPIRank> build_rank_whitelist() const;

    std::shared_ptr<SynapticElements> synaptic_elements{};
    std::shared_ptr<NetworkGraph> network_graph{};
    std::shared_ptr<NeuronsExtraInfo> extra_info{};
    std::shared_ptr<FiredStatusRecorder> fired_status_recorder{};
};

/**
 * This class deletes synapses based on randomness, i.e., it picks the
 * synapses to delete uniformely at random.
 */
class RandomSynapseDeletionFinder : public SynapseDeletionFinder {
public:
    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(SynapseDeletionFinder);
        footprint->emplace("RandomSynapseDeletionFinder", my_footprint);

        SynapseDeletionFinder::record_memory_footprint(footprint);
    }

protected:
    [[nodiscard]] std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) override;
};

class CoActivationSynapseDeletionFinder : public SynapseDeletionFinder {
public:
    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(SynapseDeletionFinder);
        footprint->emplace("CoActivationSynapseDeletionFinder", my_footprint);

        SynapseDeletionFinder::record_memory_footprint(footprint);
    }

protected:
    [[nodiscard]] std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) override;

private:
    [[nodiscard]] static double calculate_co_activation(const boost::dynamic_bitset<>& pre_synaptic, const boost::dynamic_bitset<>& post_synaptic);
};

/**
 * This class deletes synapses based on their length, i.e., it picks the
 * shorted synapses more likely (linearly dependent on the length)
 */
class InverseLengthSynapseDeletionFinder : public SynapseDeletionFinder {
public:
    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_easy_footprint = sizeof(*this) - sizeof(SynapseDeletionFinder);

        auto my_hard_footprint = uint64_t{ 0 };
        for (const auto& rank : mpiPP::MPIRank::range(partners.get_number_ranks())) {
            my_hard_footprint += partners.get_size_in_bytes(rank) + positions.get_size_in_bytes(rank);
        }

        footprint->emplace("InverseLengthSynapseDeletionFinder", my_hard_footprint + my_easy_footprint);

        SynapseDeletionFinder::record_memory_footprint(footprint);
    }

protected:
    [[nodiscard]] RelearnTypes::comm_map_deletion<SynapseDeletionRequest> find_synapses_to_delete(ElementType element_type, std::span<const SignalType> signal_types,
                                                                                                  std::span<const unsigned int> number_deletions) override;

    [[nodiscard]] RelearnTypes::comm_map_deletion<SynapseDeletionRequest> find_synapses_to_delete(ElementType element_type, SignalType signal_types,
                                                                                                  std::span<const unsigned int> number_deletions) override;

    [[nodiscard]] std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) override;

private:
    [[nodiscard]] RelearnTypes::comm_map_deletion<NeuronID> find_partners_to_locate(ElementType element_type, std::span<const SignalType> signal_types,
                                                                                    std::span<const unsigned int> number_deletions);

    [[nodiscard]] RelearnTypes::comm_map_deletion<NeuronID> find_partners_to_locate(ElementType element_type, SignalType signal_types,
                                                                                    std::span<const unsigned int> number_deletions);

    RelearnTypes::comm_map_deletion<NeuronID> partners{ 1, 1 };
    RelearnTypes::comm_map_deletion<RelearnTypes::position_type> positions{ 1, 1 };
};
