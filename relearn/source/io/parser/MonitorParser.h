#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "io/parser/NeuronIdParser.h"
#include "neurons/LocalGroupTranslator.h"
#include "neurons/helper/RankNeuronId.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <cpp-utility/StringUtil.hpp>
#include <cpp-utility/ranges/Functional.hpp>

#include <mpi-wrapper/core/MPIRank.h>

#include <range/v3/action/insert.hpp>
#include <range/v3/iterator/operations.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/cache1.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/transform.hpp>

#include <memory>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <vector>

/**
 * This class provides an interface to parse neuron ids with descriptions that can contain group names.
 */
class MonitorParser {
public:
    /**
     * @brief Parses a descriptor string for the neuron monitors. If it contains an group name (a string without ':' and not only containing digits),
     *      uses this to get the associated neuron ids from the local_group_translator (discards those that are not present)
     * @param description The string that will be parsed
     * @param local_group_translator Translates between the local group id on the current mpi rank and its group name
     * @return List of group ids found in the string
     */
    [[nodiscard]] static std::vector<RelearnTypes::group_id> parse_group_names(const std::string_view description,
                                                                               const std::shared_ptr<const LocalGroupTranslator>& local_group_translator) {
        const auto& vector = utility::split_string(std::string(description), ';');
        return parse_group_names(vector, local_group_translator);
    }

    /**
     * @brief Parses a vectgor of descriptor string for the neuron monitors. If it contains an group name (a string without ':' and not only containing digits),
     *      uses this to get the associated neuron ids from the local_group_translator (discards those that are not present)
     * @param vector The vector of strings that will be parsed
     * @param local_group_translator Translates between the local group id on the current mpi rank and its group name
     * @return List of group ids found in the string
     */
    [[nodiscard]] static std::vector<RelearnTypes::group_id> parse_group_names(const std::vector<std::string>& vector,
                                                                               const std::shared_ptr<const LocalGroupTranslator>& local_group_translator) {
        const auto& known_group_names = local_group_translator->get_all_group_names();

        return vector | ranges::views::filter([](const auto& desc) {
                   return !(desc.find(':') != std::string::npos || utility::is_number(desc));
               })
               | ranges::views::transform([&known_group_names](const auto& parsed_group_name) -> std::set<std::string> {
                     std::set<RelearnTypes::group_name> matching_group_names{};
                     for (const auto& known_group_name : known_group_names) {
                         std::smatch match;
                         const bool is_match = std::regex_match(known_group_name, match, std::regex(parsed_group_name));
                         if (is_match) {
                             matching_group_names.insert(known_group_name);
                         }
                     }
                     return matching_group_names;
                 })
               | ranges::views::cache1
               | ranges::views::join
               | ranges::views::transform([&known_group_names](const auto& parsed_group_name) {
                     return std::find(known_group_names.begin(), known_group_names.end(), parsed_group_name);
                 })
               | ranges::views::transform([&known_group_names](const auto& parsed_group_name_iter) -> RelearnTypes::group_id {
                     return static_cast<RelearnTypes::group_id>(ranges::distance(known_group_names.begin(),
                                                                                 parsed_group_name_iter));
                 })
               | ranges::to_vector;
    }

    /**
     * @brief Extracts all to be monitored NeuronIDs that belong to the current rank. Format is:
     *      <mpi_rank>:<neuron_id> with ; separating the RankNeuronIds
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, my_rank is used instead.
     *      Alternatively, it can also contain
     *      <group_name>
     *      which then translates to all NeuronIDs within the groups
     * @param description The description of the RankNeuronIds
     * @param my_rank The current MPI rank, must be initialized
     * @param local_group_translator Translates the group names to the associated NeuronIDs
     * @exception Throws a RelearnException if my_rank is not initialized
     * @return A vector with all NeuronIDs that shall be monitored at the current rank, sorted and unique
     */
    [[nodiscard]] static std::vector<NeuronID> parse_my_ids(const std::string_view description, const mpiPP::MPIRank my_rank,
                                                            const std::shared_ptr<const LocalGroupTranslator>& local_group_translator) {
        const auto& descriptions = utility::split_string(description, ';');
        return parse_my_ids(descriptions, my_rank, local_group_translator);
    }

    /**
     * @brief Extracts all to be monitored NeuronIDs that belong to the current rank. Format is:
     *      <mpi_rank>:<neuron_id> with ; separating the RankNeuronIds
     *      with a non-negative MPI rank. However, if -1 is parsed as the MPI rank, my_rank is used instead.
     *      Alternatively, it can also contain
     *      <group_name>
     *      which then translates to all NeuronIDs within the groups
     * @param descriptions The description of the RankNeuronIds
     * @param my_rank The current MPI rank, must be initialized
     * @param local_group_translator Translates the group names to the associated NeuronIDs
     * @exception Throws a RelearnException if my_rank is not initialized
     * @return A vector with all NeuronIDs that shall be monitored at the current rank, sorted and unique
     */
    [[nodiscard]] static std::vector<NeuronID> parse_my_ids(const std::vector<std::string>& descriptions, const mpiPP::MPIRank my_rank,
                                                            const std::shared_ptr<const LocalGroupTranslator>& local_group_translator) {

        const auto& rank_neuron_ids = NeuronIdParser::parse_multiple_description(descriptions, my_rank);
        auto neuron_ids = NeuronIdParser::extract_my_ids(rank_neuron_ids, my_rank);

        const auto& group_ids = parse_group_names(descriptions, local_group_translator);
        const auto& neurons_in_groups = local_group_translator->get_neuron_ids_in_groups(group_ids);

        ranges::insert(neuron_ids, neuron_ids.end(), neurons_in_groups);
        return NeuronIdParser::remove_duplicates_and_sort(std::move(neuron_ids));
    }
};
