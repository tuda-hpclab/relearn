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

#include "io/LogFiles.h"
#include "neurons/helper/RankNeuronId.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/ranges/Functional.hpp>

#include <mpi-wrapper/core/MPIRank.h>

#include <range/v3/action/sort.hpp>
#include <range/v3/action/transform.hpp>
#include <range/v3/action/unique.hpp>
#include <range/v3/algorithm/find.hpp>
#include <range/v3/iterator/operations.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/cache1.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

/**
 * This class provides an interface to parse the neuron ids that shall be monitored from a std::string.
 * It also provides the functionality to sort them and remove duplicates.
 */
class NeuronIdParser {
public:
    /**
     * @brief Parses a RankNeuronId from a description. Format must be:
     *      <mpi_rank>:<neuron_id>
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, my_rank is used instead.
     *      <neuron_id> is in input format, i.e., "+1".
     * @param description The description to parse
     * @param my_rank The default MPI rank
     * @exception Throws a RelearnException if a <neuron_id> is 0
     * @return An optional that contains the parsed RankNeuronId. Is empty if parsing failed or my_rank is not initialized
     */
    [[nodiscard]] static std::optional<RankNeuronId> parse_description(const std::string_view description, const mpiPP::MPIRank my_rank) {
        if (!my_rank.is_initialized()) {
            return {};
        }

        const auto colon_position = description.find(':');
        if (colon_position == std::string::npos) {
            return {};
        }

        const auto& mpi_rank_string = description.substr(0, colon_position);
        const auto& neuron_id_string = description.substr(colon_position + 1, description.size() - colon_position);

        auto parsed_mpi_rank = int{};
        const auto& [mpi_rank_ptr, mpi_rank_err] = std::from_chars(mpi_rank_string.data(), mpi_rank_string.data() + mpi_rank_string.size(), parsed_mpi_rank);

        if (parsed_mpi_rank == -1) {
            parsed_mpi_rank = my_rank.get_rank();
        }

        auto neuron_id = NeuronID::value_type{};
        const auto& [neuron_id_ptr, neuron_id_err] = std::from_chars(neuron_id_string.data(), neuron_id_string.data() + neuron_id_string.size(), neuron_id);

        const auto mpi_rank_ok = (mpi_rank_err == std::errc{}) && (mpi_rank_ptr == mpi_rank_string.data() + mpi_rank_string.size()) && parsed_mpi_rank >= 0;
        const auto neuron_id_ok = (neuron_id_err == std::errc{}) && (neuron_id_ptr == neuron_id_string.data() + neuron_id_string.size());

        if (mpi_rank_ok && neuron_id_ok) {
            // Check here so we can use the previous error codes correctly
            RelearnException::check(neuron_id > 0, "NeuronIdParser::parse_description: A parsed NeuronID is 0, but the input is 1-based: {}", description);

            return RankNeuronId{ mpiPP::MPIRank(parsed_mpi_rank), NeuronID(neuron_id - 1) };
        }

        LogFiles::print_message_rank(mpiPP::MPIRank::root_rank(), "Failed to parse string to match the pattern <mpi_rank>:<neuron_id> : {}", description);
        return {};
    }

    /**
     * @brief Parses multiple RankNeuronIds from a description. Format is:
     *      <mpi_rank>:<neuron_id> with ; separating the RankNeuronIds
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, my_rank is used instead.
     *      <neuron_id> is in input format, i.e., "+1".
     * @param description The description of the RankNeuronIds
     * @param my_rank The default MPI rank, must be initialized
     * @exception Throws a RelearnException if my_rank is not initialized or a <neuron_id> is 0
     * @return A vector with all successfully parsed RankNeuronIds
     */
    [[nodiscard]] static std::vector<RankNeuronId> parse_multiple_description(const std::string_view description, const mpiPP::MPIRank my_rank) {
        RelearnException::check(my_rank.is_initialized(), "NeuronIdParser::parse_multiple_description: my_rank is not initialized.", my_rank);

        auto parsed_ids = std::vector<RankNeuronId>{};
        // The first description is at least 3 chars long, the following at least 4
        parsed_ids.reserve((description.size() >> 2U) + 1U);

        auto current_position = std::string::size_type{ 0 };

        while (true) {
            auto semicolon_position = description.find(';', current_position);
            if (semicolon_position == std::string_view::npos) {
                semicolon_position = description.size();
            }

            const auto substring = description.substr(current_position, semicolon_position - current_position);
            const auto opt_rank_neuron_id = parse_description(substring, my_rank);

            if (opt_rank_neuron_id.has_value()) {
                parsed_ids.emplace_back(opt_rank_neuron_id.value());
            }

            if (semicolon_position == description.size()) {
                break;
            }

            current_position = semicolon_position + 1;
        }

        return parsed_ids;
    }

    /**
     * @brief Parses multiple RankNeuronIds from a description. Format is:
     *      <mpi_rank>:<neuron_id> with ; separating the RankNeuronIds
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, my_rank is used instead.
     *      <neuron_id> is in input format, i.e., "+1".
     * @param descriptions The description of the RankNeuronIds
     * @param my_rank The default MPI rank, must be initialized
     * @exception Throws a RelearnException if my_rank is not initialized or a <neuron_id> is 0
     * @return A vector with all successfully parsed RankNeuronIds
     */
    [[nodiscard]] static std::vector<RankNeuronId> parse_multiple_description(const std::vector<std::string>& descriptions, const mpiPP::MPIRank my_rank) {
        RelearnException::check(my_rank.is_initialized(), "NeuronIdParser::parse_multiple_description: my_rank is not initialized.", my_rank);

        auto parsed_ids = std::vector<RankNeuronId>{};
        parsed_ids.reserve(descriptions.size());
        for (const auto& description : descriptions) {
            const auto opt_rank_neuron_id = parse_description(description, my_rank);

            if (opt_rank_neuron_id.has_value()) {
                parsed_ids.emplace_back(opt_rank_neuron_id.value());
            }
        }
        return parsed_ids;
    }

    /**
     * @brief Extracts all NeuronIDs from the RankNeuronIds that belong to the given rank.
     * @param rank_neuron_ids The rank neuron ids
     * @param my_rank The current MPI rank, must be initialized
     * @exception Throws a RelearnException if my_rank is not initialized or if a NeuronID in rank_neuron_ids is 0
     * @return A vector with all successfully parsed RankNeuronIds
     */
    [[nodiscard]] static std::vector<NeuronID> extract_my_ids(const std::vector<RankNeuronId>& rank_neuron_ids, const mpiPP::MPIRank my_rank) {
        RelearnException::check(my_rank.is_initialized(), "NeuronIdParser::extract_my_ids: my_rank is not initialized.", my_rank);

        return rank_neuron_ids
               | ranges::views::filter(utility::equal_to(my_rank), &RankNeuronId::get_rank)
               | ranges::views::transform([](const auto& neuron_id) { return NeuronID{ neuron_id.get_neuron_id().get_neuron_id() }; })
               | ranges::to_vector;
    }

    /**
     * @brief Removes duplicate NeuronID from the parameter and sorts the result.
     *      Requires that all NeuronIDs are actual ids, i.e., neither virtual nor uninitialized.
     * @param neuron_ids NeuronIDs to check for duplicates
     * @exception Throws a RelearnException if a NeuronID was virtual or uninitialized
     * @return The unique and sorted NeuronIDs
     */
    [[nodiscard]] static std::vector<NeuronID> remove_duplicates_and_sort(std::vector<NeuronID> neuron_ids) {
        return std::move(neuron_ids)
               | ranges::actions::transform([](NeuronID neuron_id) {
                     RelearnException::check(neuron_id.is_initialized(), "neuron_id is uninitialized");
                     RelearnException::check(!neuron_id.is_virtual(), "neuron_id is virtual");
                     return neuron_id;
                 })
               | ranges::actions::sort
               | ranges::actions::unique;
    }
};
