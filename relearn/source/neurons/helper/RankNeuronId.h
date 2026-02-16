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

#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "mpi-wrapper/MPIRank.h"

#include <boost/functional/hash.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <compare>
#include <ostream>
#include <utility>

/**
 * Identifies a neuron by the MPI rank of its owner
 * and its neuron id on the owner, i.e., the pair <rank, neuron_id>
 */
class RankNeuronId {
public:
    /**
     * @brief Constructs a new RankNeuronId with invalid rank and id.
     *      This constructor is present for resizing vectors, etc.
     */
    constexpr RankNeuronId() = default;

    /**
     * @brief Constructs a new RankNeuronId with specified inputs (not validated)
     * @param _rank The MPI rank
     * @param _neuron_id The neuron id
     */
    constexpr RankNeuronId(const mpiPP::MPIRank _rank, const NeuronID _neuron_id) noexcept
        : rank(_rank)
        , neuron_id(_neuron_id) {
    }

    /**
     * @brief Provides a lexicographical ordering of RankNeuronId
     * @param first The first object
     * @param second The second object
     * @return A strong ordering on RankNeuronId
     */
    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const RankNeuronId& first, const RankNeuronId& second) noexcept = default;

    /**
     * @brief Returns the associated MPI rank
     * @return The MPI rank, must be initialized
     * @exception Throws a RelearnException if the rank is not initialized
     */
    [[nodiscard]] constexpr mpiPP::MPIRank get_rank() const {
        RelearnException::check(rank.is_initialized(), "RankNeuronId::get_rank: The rank was not initialized");
        return rank;
    }

    /**
     * @brief Returns the associated neuron id
     * @return The neuron id
     * @exception Throws a RelearnException if the id is not initialized
     */
    [[nodiscard]] constexpr NeuronID get_neuron_id() const {
        RelearnException::check(neuron_id.is_initialized(), "RankNeuronId::get_neuron_id: neuron_id is not initialized");
        return neuron_id;
    }

    /**
     * @brief Prints the object's rank and id; inserts \n
     * @param os The out-stream in which the object is printed
     * @return The argument os to allow chaining
     */
    friend std::ostream& operator<<(std::ostream& os, const RankNeuronId& rni) {
        os << "Rank: " << rni.get_rank() << "\t id: " << rni.get_neuron_id() << '\n';
        return os;
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto& get() & {
        if constexpr (Index == 0) {
            return rank;
        }
        if constexpr (Index == 1) {
            return neuron_id;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr const auto& get() const& {
        if constexpr (Index == 0) {
            return rank;
        }
        if constexpr (Index == 1) {
            return neuron_id;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] constexpr auto&& get() && {
        if constexpr (Index == 0) {
            return rank;
        }
        if constexpr (Index == 1) {
            return neuron_id;
        }
    }

private:
    mpiPP::MPIRank rank{}; // MPI rank of the owner
    NeuronID neuron_id{};  // Neuron id on the owner
};

template <>
struct fmt::formatter<RankNeuronId> : ostream_formatter { };

namespace std {
template <>
struct tuple_size<::RankNeuronId> {
    static constexpr size_t value = 2;
};

template <>
struct tuple_element<0, ::RankNeuronId> {
    using type = mpiPP::MPIRank;
};

template <>
struct tuple_element<1, ::RankNeuronId> {
    using type = NeuronID;
};

} // namespace std

namespace std {
template <>
struct hash<RankNeuronId> {
    using argument_type = RankNeuronId;
    using result_type = std::size_t;

    result_type operator()(const argument_type& rni) const {
        const auto& [rank, neuron_id] = rni;

        const auto rank_hash = std::hash<mpiPP::MPIRank>{}(rank);
        const auto neuron_id_hash = std::hash<NeuronID>{}(neuron_id);

        std::size_t total_hash = rank_hash;
        boost::hash_combine(total_hash, neuron_id_hash);
        return total_hash;
    }
};
} // namespace std

// for use of boost::hash for SynapticIndividuallyWeightedActivityInput::weight_map_type
[[nodiscard]] inline std::size_t hash_value(const RankNeuronId& rni) {
    return std::hash<RankNeuronId>{}(rni);
}
