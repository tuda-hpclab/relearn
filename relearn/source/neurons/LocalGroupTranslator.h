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

#include "Config.h"
#include "Types.h"

#include "io/NeuronIO.h"
#include "util/File.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "cpp-utility/MemoryFootprint.hpp"
#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIInfo.h"

#include <range/v3/algorithm/any_of.hpp>
#include <range/v3/algorithm/contains.hpp>
#include <range/v3/algorithm/find.hpp>
#include <range/v3/iterator/operations.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

/**
 * Over all mpi ranks, neurons can be assigned to the same group. This group is identified by a name (string). Each mpi rank assigns this group an individual id.
 * Hence, the same group (same name) can have different ids on two mpi ranks.
 * This class helps to convert between the group id on this mpi rank and its unique name.
 */
class LocalGroupTranslator {
public:
    /**
     * Constructs a LocalGroupTranslator with the given group_id_to_group_name and neuron_id_to_group_ids
     * @param _group_id_to_group_name Maps the id of a group to its name
     * @param _neuron_id_to_group_ids Map the neuron id to its assigned group ids
     * @exception Throws a RelearnException if the group_id_to_group_name or neuron_id_to_group_ids is empty
     *     or if any group id is out of bounds or if a neuron has no group ids or if a group name occurs more than once
     */
    LocalGroupTranslator(RelearnTypes::group_names _group_id_to_group_name,
                         std::vector<RelearnTypes::group_ids> _neuron_id_to_group_ids)
        : group_id_to_group_name(std::move(_group_id_to_group_name))
        , neuron_id_to_group_ids(std::move(_neuron_id_to_group_ids)) {

        RelearnException::check(!group_id_to_group_name.empty(), "LocalGroupTranslator::group_id_to_group_name is empty");
        RelearnException::check(!neuron_id_to_group_ids.empty(), "LocalGroupTranslator::neuron_id_to_group_id is empty");

        for (const auto& group_ids : neuron_id_to_group_ids) {
            RelearnException::check(!group_ids.empty(), "LocalGroupTranslator:: The group ids from at least one neuron are empty, but they should't!");
            for (const auto& group_id : group_ids) {
                RelearnException::check(group_id < group_id_to_group_name.size(), "LocalGroupTranslator:: Invalid group id {}. Must be between 0 and {}", group_id, group_id_to_group_name.size());
            }
        }

        const auto num_group_names = group_id_to_group_name.size();
        const auto set = group_id_to_group_name | ranges::to<std::set>;
        RelearnException::check(num_group_names == set.size(), "LocalGroupTranslator::Group name must be unique on single mpi rank");

        neuron_id_to_group_ids_unordered = neuron_id_to_group_ids
                                           | ranges::views::transform([&](const RelearnTypes::group_ids& group_ids) {
                                                 return group_ids
                                                        | ranges::to<std::unordered_set>;
                                             })
                                           | ranges::to<std::vector>;
    }

    /**
     * Constructs a LocalGroupTranslator with number_neurons many neurons. Uses only the default group and puts every neuron in it
     * @param number_neurons The number of neurons
     * @exception Throws a RelearnException if number_neurons <= 0
     */
    explicit LocalGroupTranslator(RelearnTypes::number_neurons_type number_neurons) {
        RelearnException::check(number_neurons > 0, "LocalGroupTranslator: number neurons must be over 0! Actually was: {}", number_neurons);
        neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>{ number_neurons, { Constants::default_group_id } };
        neuron_id_to_group_ids_unordered = neuron_id_to_group_ids
                                           | ranges::views::transform([&](const RelearnTypes::group_ids& group_ids) {
                                                 return group_ids
                                                        | ranges::to<std::unordered_set>;
                                             })
                                           | ranges::to<std::vector>;
    }

    /**
     * Constructs a LocalGroupTranslator from a given file with the neuron groups. Uses NeuronIO::read_neuron_groups for that
     * @param file_path_groups The file path to the groups
     * @param number_neurons The number neurons
     * @exception Throws a RelearnException if the read neuron_id_to_group_ids or group_id_to_group_name are empty or if the group ids from at least
     * one neuron are empty. Also throws a RelearnException if a group id is out of bounds or if a group name occurs more than once
     */
    LocalGroupTranslator(const std::filesystem::path& file_path_groups, RelearnTypes::number_neurons_type number_neurons) {
        const auto path_to_file = Util::find_file_for_rank(file_path_groups, mpiPP::MPIInfo::get_my_rank(), "rank_", "_groups.txt");
        auto [_neuron_id_to_group_ids, _group_id_to_group_name] = NeuronIO::read_neuron_groups(path_to_file, number_neurons);

        neuron_id_to_group_ids = std::move(_neuron_id_to_group_ids);
        group_id_to_group_name = std::move(_group_id_to_group_name);

        RelearnException::check(!group_id_to_group_name.empty(), "LocalGroupTranslator::group_id_to_group_name is empty");
        RelearnException::check(!neuron_id_to_group_ids.empty(), "LocalGroupTranslator::neuron_id_to_group_id is empty");

        for (const auto& group_ids : neuron_id_to_group_ids) {
            RelearnException::check(!group_ids.empty(), "LocalGroupTranslator:: The group ids from at least one neuron are empty, but they should't!");
            for (const auto& group_id : group_ids) {
                RelearnException::check(group_id < group_id_to_group_name.size(), "LocalGroupTranslator:: Invalid group id {}. Must be between 0 and {}", group_id, group_id_to_group_name.size());
            }
        }

        const auto num_group_names = group_id_to_group_name.size();
        const auto set = group_id_to_group_name | ranges::to<std::set>;
        RelearnException::check(num_group_names == set.size(), "LocalGroupTranslator::Group name must be unique on single mpi rank");

        neuron_id_to_group_ids_unordered = neuron_id_to_group_ids
                                           | ranges::views::transform([&](const RelearnTypes::group_ids& group_ids) {
                                                 return group_ids
                                                        | ranges::to<std::unordered_set>;
                                             })
                                           | ranges::to<std::vector>;
    }

    /**
     * Returns the group ids in which the neuron lays
     * @param neuron_id Id of the neuron
     * @return Group ids
     */
    [[nodiscard]] const RelearnTypes::group_ids& get_group_ids_for_neuron_id(RelearnTypes::neuron_id neuron_id) const {
        RelearnException::check(neuron_id < neuron_id_to_group_ids.size(), "LocalGroupTranslator::get_group_id_for_neuron_id: Neuron id is too large. {} < {}", neuron_id, neuron_id_to_group_ids.size());
        return neuron_id_to_group_ids[neuron_id];
    }

    /**
     * Returns the unordered group ids in which the neuron lays
     * @param neuron_id Id of the neuron
     * @return Unordered group ids
     */
    [[nodiscard]] const RelearnTypes::group_ids_unordered& get_group_ids_for_neuron_id_unordered(RelearnTypes::neuron_id neuron_id) const {
        RelearnException::check(neuron_id < neuron_id_to_group_ids.size(), "LocalGroupTranslator::get_group_id_for_neuron_id_unordered: Neuron id is too large. {} < {}", neuron_id, neuron_id_to_group_ids.size());
        return neuron_id_to_group_ids_unordered[neuron_id];
    }

    /**
     * Returns the group name of a group id
     * @param group_id The id of the group
     * @return Name of the group
     */
    [[nodiscard]] const RelearnTypes::group_name& get_group_name_for_group_id(RelearnTypes::group_id group_id) const {
        RelearnException::check(group_id < group_id_to_group_name.size(), "LocalGroupTranslator::get_group_name_for_group_id: Group id is too large. {} < {}", group_id, group_id_to_group_name.size());
        return group_id_to_group_name[group_id];
    }

    /**
     * Returns the group names of group ids
     * @param group_ids The ids of the groups
     * @return The names of the groups in the same order
     */
    [[nodiscard]] RelearnTypes::group_names get_group_names_for_group_ids(RelearnTypes::group_ids group_ids) const {
        return group_ids
               | ranges::views::transform([&group_id_to_group_name = group_id_to_group_name](const RelearnTypes::group_id& group_id) {
                     RelearnException::check(group_id < group_id_to_group_name.size(), "LocalGroupTranslator::get_group_names_for_group_ids: Group id is too large. {} < {}", group_id, group_id_to_group_name.size());
                     return group_id_to_group_name[group_id];
                 })
               | ranges::to<RelearnTypes::group_names>;
    }

    /**
     * Returns the unordered group names of unordered group ids
     * @param group_ids The unordered ids of the groups
     * @return The unordered names of the groups
     */
    [[nodiscard]] RelearnTypes::group_names_unordered get_group_names_for_group_ids_unordered(const RelearnTypes::group_ids_unordered& group_ids) const {
        auto group_names = RelearnTypes::group_names_unordered{};
        group_names.reserve(group_ids.size()); // Optional, falls RelearnTypes::group_names_unordered = unordered_set

        for (const auto& group_id : group_ids) {
            RelearnException::check(group_id < group_id_to_group_name.size(), "LocalGroupTranslator::get_group_names_for_group_ids_unordered: Group id is too large. {} < {}", group_id, group_id_to_group_name.size());
            group_names.insert(group_id_to_group_name[group_id]);
        }

        return group_names;
    }

    /**
     * Returns the names of the groups in which the neuron lays
     * @param neuron_id Id of the neuron
     * @return Names of the groups
     */
    [[nodiscard]] RelearnTypes::group_names get_group_names_for_neuron_id(RelearnTypes::neuron_id neuron_id) const {
        return get_group_names_for_group_ids(get_group_ids_for_neuron_id(neuron_id));
    }

    /**
     * Returns the unordered names of the groups in which the neuron lays
     * @param neuron_id Id of the neuron
     * @return Unordered names of the groups
     */
    [[nodiscard]] RelearnTypes::group_names_unordered get_group_names_for_neuron_id_unordered(RelearnTypes::neuron_id neuron_id) const {
        return get_group_names_for_group_ids_unordered(get_group_ids_for_neuron_id_unordered(neuron_id));
    }

    /**
     * Returns the group id on this mpi rank for the group name
     * @param group_name Name of the group
     * @return Id of the group
     */
    [[nodiscard]] RelearnTypes::group_id get_group_id_for_group_name(const RelearnTypes::group_name& group_name) const {
        const auto iter = ranges::find(group_id_to_group_name, group_name);
        RelearnException::check(iter != group_id_to_group_name.end(), "LocalGroupTranslator::get_group_id_for_group_name: Group name {} is unknown", group_name);
        return static_cast<RelearnTypes::group_id>(ranges::distance(group_id_to_group_name.begin(), iter));
    }

    /**
     * Returns if the group name is known on this mpi rank
     * @param group_name Name of the group
     * @return True, if group name exists on this mpi rank
     */
    [[nodiscard]] bool knows_group_name(const RelearnTypes::group_name& group_name) const noexcept {
        return ranges::contains(group_id_to_group_name, group_name);
    }

    /**
     * Returns the number of neurons on this mpi rank (over all local groups)
     * @return Number of neurons on this mpi rank
     */
    [[nodiscard]] RelearnTypes::number_neurons_type get_number_neurons_in_total() const noexcept {
        return neuron_id_to_group_ids.size();
    }

    /**
     * Vector with all used group names on this mpi rank
     * @return Vector of group names
     */
    [[nodiscard]] const std::vector<RelearnTypes::group_name>& get_all_group_names() const noexcept {
        return group_id_to_group_name;
    }

    /**
     * Returns number of groups available on this mpi rank
     * @return Number of groups
     */
    [[nodiscard]] std::size_t get_number_of_groups() const noexcept {
        return group_id_to_group_name.size();
    }

    /**
     * @brief Return number of neurons placed with a certain group name
     * @return Number of neurons currently stored under the given group name
     */
    [[nodiscard]] RelearnTypes::number_neurons_type get_number_neurons_in_group(const RelearnTypes::group_id& group_id) const {
        auto count = RelearnTypes::number_neurons_type{ 0 };
        for (const auto& group_ids : neuron_id_to_group_ids) {
            if (ranges::contains(group_ids, group_id)) {
                ++count;
            }
        }
        return count;
    }

    /**
     * @brief Returns all neuron ids in a specific group on this rank
     * @param my_group_id Vector of group ids in which the neuron ids must lay
     * @return Vector of neuron ids within the specified group in my_group_ids
     */
    [[nodiscard]] std::unordered_set<NeuronID> get_neuron_ids_in_group(RelearnTypes::group_id my_group_id) const {
        RelearnException::check(my_group_id < group_id_to_group_name.size(), "LocalGroupTranslator::get_neuron_ids_in_group: Group id {} is too large", my_group_id);

        auto contains_group_id = [&my_group_id](const RelearnTypes::group_ids_unordered& group_ids) {
            // return ranges::find(group_ids, my_group_id) != ranges::end(group_ids);
            return ranges::contains(group_ids, my_group_id);
        };

        return NeuronID::range(neuron_id_to_group_ids.size())
               | ranges::views::filter(contains_group_id, utility::lookup(neuron_id_to_group_ids_unordered, &NeuronID::get_neuron_id))
               | ranges::to<std::unordered_set>;
    }

    /**
     * @brief Returns all neuron ids in specific groups on this rank
     * @param my_group_ids Vector of group ids in which the neuron ids must lay
     * @return Vector of neuron ids within the specified group in my_group_ids
     */
    [[nodiscard]] std::vector<NeuronID> get_neuron_ids_in_groups(const RelearnTypes::group_ids& my_group_ids) const {
        const auto is_id_in_my_ranks_groups = [my_group_ids, &neuron_id_to_group_ids = neuron_id_to_group_ids](const NeuronID& neuron_id) {
            // check if my_group_ids and neuron_id_to_group_ids[neuron_id.get_neuron_id()] intersect
            const auto id = neuron_id.get_neuron_id();
            for (const auto& group_id : my_group_ids) {
                for (const auto& neuron_group_id : neuron_id_to_group_ids[id]) {
                    if (neuron_group_id == group_id) {
                        return true;
                    }
                }
            }
            return false;
        };

        return NeuronID::range(neuron_id_to_group_ids.size())
               | ranges::views::filter(is_id_in_my_ranks_groups)
               | ranges::to_vector;
    }

    /**
     * @brief Returns all neuron ids that are in the specified groups
     * @param my_group_ids The groups to look in
     * @return The neuron ids
     */
    [[nodiscard]] std::unordered_set<NeuronID> get_neuron_ids_in_groups(const RelearnTypes::group_ids_unordered& my_group_ids) const {
        const auto is_id_in_my_ranks_groups = [&my_group_ids, &neuron_id_to_group_ids_unordered = neuron_id_to_group_ids_unordered](const NeuronID& neuron_id) {
            const auto& neuron_groups = neuron_id_to_group_ids_unordered[neuron_id.get_neuron_id()];

            return ranges::any_of(my_group_ids.begin(), my_group_ids.end(), [&](const auto& group_id) {
                return ranges::contains(neuron_groups, group_id);
            });
        };

        auto result = std::unordered_set<NeuronID>{};
        for (const auto& id : NeuronID::range(neuron_id_to_group_ids_unordered.size())) {
            if (is_id_in_my_ranks_groups(id)) {
                result.insert(id);
            }
        }
        return result;
    }

    /**
     * @brief Returns the group ids for all neurons
     * @return The group ids
     */
    [[nodiscard]] const std::vector<RelearnTypes::group_ids>& get_neuron_ids_to_group_ids() const noexcept {
        return neuron_id_to_group_ids;
    }

    /**
     * @brief Returns the unordered group ids for all neurons
     * @return The unordered group ids
     */
    [[nodiscard]] const std::vector<RelearnTypes::group_ids_unordered>& get_neuron_ids_to_group_ids_unordered() const noexcept {
        return neuron_id_to_group_ids_unordered;
    }

    /**
     * @brief Returns the ids for all specified names (duplicates are kept)
     * @param group_names The names to translate
     * @return The ids
     */
    [[nodiscard]] std::vector<RelearnTypes::group_id> translate_group_names_to_group_ids_ordered(const std::vector<std::string>& group_names) const {
        auto group_ids = RelearnTypes::group_ids{};
        group_ids.reserve(group_names.size());
        for (const auto& name : group_names) {
            const auto& id_ = get_group_id_for_group_name(name);
            group_ids.push_back(id_);
        }
        return group_ids;
    }

    /**
     * @brief Returns the ids for all specified names (duplicates are removed)
     * @param group_names The names to translate
     * @return The ids
     */
    [[nodiscard]] RelearnTypes::group_ids_unordered translate_group_names_to_group_ids(const RelearnTypes::group_names_unordered& group_names) const {
        auto group_ids = RelearnTypes::group_ids_unordered{};
        group_ids.reserve(group_names.size());
        for (const auto& name : group_names) {
            const auto& id_ = get_group_id_for_group_name(name);
            group_ids.insert(id_);
        }
        return group_ids;
    }

    /**
     * @brief Returns all group names that match the given regex
     * @param group_regex The regex
     * @return The matching group names
     */
    [[nodiscard]] RelearnTypes::group_names_unordered get_matching_group_names(const std::string& group_regex) const {
        auto matching_group_names = RelearnTypes::group_names_unordered{};
        const auto& names = get_all_group_names();
        for (const auto& known_group_name : names) {
            auto match = std::smatch{};
            const bool is_match = std::regex_match(known_group_name, match, std::regex(group_regex));
            if (is_match) {
                matching_group_names.insert(known_group_name);
            }
        }
        return matching_group_names;
    }

    /**
     * @brief Returns all group ids that match the given regex
     * @param group_regex The regex
     * @return The matching group ids
     */
    [[nodiscard]] RelearnTypes::group_ids_unordered get_group_ids_for_matching_group_names(const std::string& group_regex) const {
        const auto& group_names = get_matching_group_names(group_regex);
        return translate_group_names_to_group_ids(group_names);
    }

    /**
     * @brief Creates the number of neurons of default group id
     * @param created_neurons The number of to-be-created neurons
     * @exception Throws a RelearnExcepttion if created_neurons == 0 or if there are no previous neurons
     */
    void create_neurons(RelearnTypes::number_neurons_type created_neurons) {
        RelearnException::check(!neuron_id_to_group_ids.empty(), "LocalGroupTranslator::create_neurons: Was not initialized");
        RelearnException::check(created_neurons > 0, "LocalGroupTranslator::create_neurons: Cannot create 0 neurons");
        const auto old_size = neuron_id_to_group_ids.size();
        neuron_id_to_group_ids.resize(old_size + created_neurons, { Constants::default_group_id });
        neuron_id_to_group_ids_unordered.resize(old_size + created_neurons, { Constants::default_group_id });
    }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<utility::MemoryFootprint>& footprint) {
        const auto my_easy_footprint = sizeof(*this)
                                       + (group_id_to_group_name.capacity() * sizeof(RelearnTypes::group_name))
                                       + (neuron_id_to_group_ids.capacity() * sizeof(RelearnTypes::group_ids));

        auto my_hard_footprint = std::uint64_t{ 0 };
        for (const auto& str : group_id_to_group_name) {
            my_hard_footprint += str.capacity() * sizeof(RelearnTypes::group_name::value_type);
        }

        footprint->emplace("LocalGroupTranslator", my_hard_footprint + my_easy_footprint);
    }

private:
    RelearnTypes::group_names group_id_to_group_name{ std::string{ Constants::default_group_name } };
    std::vector<RelearnTypes::group_ids> neuron_id_to_group_ids;
    std::vector<RelearnTypes::group_ids_unordered> neuron_id_to_group_ids_unordered;
};
