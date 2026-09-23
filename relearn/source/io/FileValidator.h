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

#include "util/NeuronID.h"

#include <cpp-utility/ranges/Functional.hpp>
#include <cpp-utility/ranges/views/IO.hpp>

#include <range/v3/view/getlines.hpp>

#include <algorithm>
#include <filesystem>
#include <istream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class FileValidator {
public:
    /**
     * @brief Checks if the specified file contains only synapses between neurons with specified ids (only works locally).
     * @param path_synapses The path to the file in which the synapses are stored (with the ids starting at 1)
     * @param neuron_ids The neuron ids between which the synapses should be formed. Must be sorted ascendingly
     * @tparam synapse_weight The type of synapse weight
     * @exception Throws an exception if the allocation of memory fails
     * @return Returns true iff the file has the correct format and only ids in neuron_ids are present
     */
    template <typename synapse_weight>
    static bool check_edges_from_file(const std::filesystem::path& path_synapses, const std::vector<NeuronID::value_type>& neuron_ids) {
        auto file_synapses = std::ifstream(path_synapses, std::ios::binary | std::ios::in);

        auto ids_in_file = std::set<NeuronID::value_type>{};

        for (const auto& line : ranges::getlines(file_synapses) | utility::views::filter_not_comment_not_empty_line) {
            auto source_id = NeuronID::value_type{ 0 };
            auto target_id = NeuronID::value_type{ 0 };
            auto weight = synapse_weight{ 0 };

            auto sstream = std::stringstream{ line };
            const auto success = (sstream >> source_id) && (sstream >> target_id) && (sstream >> weight);

            if (!success) {
                return false;
            }

            // The neurons start with 1
            source_id--;
            target_id--;

            ids_in_file.insert(source_id);
            ids_in_file.insert(target_id);
        }

        return std::ranges::all_of(ids_in_file, [&neuron_ids](const NeuronID::value_type val) {
            return std::ranges::binary_search(neuron_ids, val);
        });
    }
};
