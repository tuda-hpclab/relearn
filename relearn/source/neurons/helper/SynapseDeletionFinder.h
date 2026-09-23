#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2023-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/NeuronsExtraInfo.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "neurons/synaptic_elements/SynapticElements.h"
#include "types/BasicTypes.h"
#include "types/CommunicationTypes.h"
#include "types/SpaceTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/MemoryFootprint.hpp>

#include <fmt/ostream.h>

#include <mpi-wrapper/core/MPIRank.h>
#include <mpi-wrapper/core/MPIRankRange.h>

#include <memory>
#include <span>
#include <utility>
#include <vector>

class NetworkGraph;

/**
 * This enums lists all types of synapse deletion finders
 */
enum class SynapseDeletionFinderType : char {
    Random,
    InverseLength,
    // CoActivation,
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

    // if (synapse_deletion_type == SynapseDeletionFinderType::CoActivation) {
    //     return out << "CoActivation";
    // }

    return out;
}

template <>
struct fmt::formatter<SynapseDeletionFinderType> : ostream_formatter { };

/**
 * @brief SynapseDeletionFinder resolves, at compile time, to the synapse deletion finder
 *      implementation for this build: SynapseDeletionFinderGPU when RELEARN_CUDA_ENABLED,
 *      SynapseDeletionFinderCPU otherwise. A build only ever compiles one of the two, so this is a
 *      plain type alias rather than a runtime choice. RandomSynapseDeletionFinder and
 *      InverseLengthSynapseDeletionFinder below are the genuinely runtime-polymorphic axis (chosen
 *      by user config, see SynapseDeletionFinderType) and derive from this alias.
 */
#ifdef RELEARN_CUDA_ENABLED
#include "cuda/deletion/SynapseDeletionFinderGPU.h"
using SynapseDeletionFinder = SynapseDeletionFinderGPU;
#else
#include "SynapseDeletionFinderCPU.h"
using SynapseDeletionFinder = SynapseDeletionFinderCPU;
#endif

/**
 * This class deletes synapses based on randomness, i.e., it picks the
 * synapses to delete uniformely at random.
 */
class RandomSynapseDeletionFinder : public SynapseDeletionFinder {
public:
    RandomSynapseDeletionFinder();

    RandomSynapseDeletionFinder(const RandomSynapseDeletionFinder&) = default;
    RandomSynapseDeletionFinder& operator=(const RandomSynapseDeletionFinder&) = default;

    RandomSynapseDeletionFinder(RandomSynapseDeletionFinder&&) = delete;
    RandomSynapseDeletionFinder& operator=(RandomSynapseDeletionFinder&&) = delete;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(SynapseDeletionFinder);
        footprint->emplace("RandomSynapseDeletionFinder", my_footprint);

        SynapseDeletionFinder::record_memory_footprint(footprint);
    }

    void init(RelearnTypes::number_neurons_type number_neurons) override;

    ~RandomSynapseDeletionFinder() override = default;

protected:
    [[nodiscard]] std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, counter_type num_synapses_to_delete) override;
};
//
// class CoActivationSynapseDeletionFinder : public SynapseDeletionFinder {
// public:
//     CoActivationSynapseDeletionFinder() = default;
//
//     CoActivationSynapseDeletionFinder(const CoActivationSynapseDeletionFinder&) = default;
//     CoActivationSynapseDeletionFinder& operator=(const CoActivationSynapseDeletionFinder&) = default;
//
//     CoActivationSynapseDeletionFinder(CoActivationSynapseDeletionFinder&&) = delete;
//     CoActivationSynapseDeletionFinder& operator=(CoActivationSynapseDeletionFinder&&) = delete;
//
//     /**
//      * @brief Records the memory footprint of the current object
//      * @param footprint Where to store the current footprint
//      */
//     void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
//         const auto my_footprint = sizeof(*this) - sizeof(SynapseDeletionFinder);
//         footprint->emplace("CoActivationSynapseDeletionFinder", my_footprint);
//
//         SynapseDeletionFinder::record_memory_footprint(footprint);
//     }
//
//     ~CoActivationSynapseDeletionFinder() = default;
//
// protected:
//     [[nodiscard]] std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, counter_type num_synapses_to_delete) override;
//
// private:
//     [[nodiscard]] static double calculate_co_activation(const boost::dynamic_bitset<>& pre_synaptic, const boost::dynamic_bitset<>& post_synaptic);
// };

/**
 * This class deletes synapses based on their length, i.e., it picks the
 * shorted synapses more likely (linearly dependent on the length)
 */
class InverseLengthSynapseDeletionFinder : public SynapseDeletionFinder {
public:
    InverseLengthSynapseDeletionFinder() = default;

    InverseLengthSynapseDeletionFinder(const InverseLengthSynapseDeletionFinder&) = default;
    InverseLengthSynapseDeletionFinder& operator=(const InverseLengthSynapseDeletionFinder&) = default;

    InverseLengthSynapseDeletionFinder(InverseLengthSynapseDeletionFinder&&) = delete;
    InverseLengthSynapseDeletionFinder& operator=(InverseLengthSynapseDeletionFinder&&) = delete;

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) override {
        const auto my_easy_footprint = sizeof(*this) - sizeof(SynapseDeletionFinder);

        auto my_hard_footprint = uint64_t{ 0 };
        for (const auto& rank : mpiPP::MPIRankRange::range(partners.get_number_ranks())) {
            my_hard_footprint += partners.get_size_in_bytes(rank) + positions.get_size_in_bytes(rank);
        }

        footprint->emplace("InverseLengthSynapseDeletionFinder", my_hard_footprint + my_easy_footprint);

        SynapseDeletionFinder::record_memory_footprint(footprint);
    }

    ~InverseLengthSynapseDeletionFinder() override = default;

protected:
#ifndef RELEARN_CUDA_ENABLED
    [[nodiscard]] RelearnTypes::comm_map_deletion<SynapseDeletionRequest> find_synapses_to_delete(ElementType element_type, std::span<const SignalType> signal_types,
                                                                                                  std::span<const counter_type> number_deletions) override;

    [[nodiscard]] RelearnTypes::comm_map_deletion<SynapseDeletionRequest> find_synapses_to_delete(ElementType element_type, SignalType signal_types,
                                                                                                  std::span<const counter_type> number_deletions) override;
#endif

    [[nodiscard]] std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, counter_type num_synapses_to_delete) override;

private:
    [[nodiscard]] RelearnTypes::comm_map_deletion<NeuronID> find_partners_to_locate(ElementType element_type, std::span<const SignalType> signal_types,
                                                                                    std::span<const counter_type> number_deletions);

    [[nodiscard]] RelearnTypes::comm_map_deletion<NeuronID> find_partners_to_locate(ElementType element_type, SignalType signal_types,
                                                                                    std::span<const counter_type> number_deletions);

    RelearnTypes::comm_map_deletion<NeuronID> partners{ 1, 1 };
    RelearnTypes::comm_map_deletion<RelearnTypes::position_type> positions{ 1, 1 };
};
