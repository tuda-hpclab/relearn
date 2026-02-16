/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronIO.h"

#include "Config.h"
#include "Types.h"

#include "neurons/LocalGroupTranslator.h"
#include "neurons/enums/SynapticElementType.h"
#include "sim/LoadedNeuron.h"
#include "sim/file/AdditionalPositionInformation.h"
#include "structure/Partition.h"
#include "util/NeuronFilePaths.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"
#include "util/StringUtil.h"

#include "cpp-utility/ranges/views/IO.hpp"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include <boost/lexical_cast.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <range/v3/algorithm/find.hpp>
#include <range/v3/iterator/operations.hpp>
#include <range/v3/view/getlines.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

std::vector<std::string> NeuronIO::read_comments(const std::filesystem::path& file_path) {
    auto file = std::ifstream(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_comments: Opening the file '{}' was not successful", file_path);

    auto comments = std::vector<std::string>{};

    for (auto line = std::string{}; std::getline(file, line);) {
        if (line.empty()) {
            continue;
        }

        if (line[0] == '#') {
            comments.emplace_back(std::move(line));
            continue;
        }

        break;
    }

    return comments;
}

AdditionalPositionInformation NeuronIO::parse_additional_position_information(const std::vector<std::string>& comments) {
    const auto search_multiple = [&comments](const std::string& specifier) -> std::vector<std::string> {
        auto results = std::vector<std::string>{};

        for (const auto& comment : comments) {
            const auto position = comment.find(specifier);
            if (position == std::string::npos) {
                continue;
            }
            results.push_back(comment.substr(specifier.size()));
        }

        return results;
    };

    const auto search = [&comments](const std::string& specifier) -> double {
        for (const auto& comment : comments) {
            const auto position = comment.find(specifier);
            if (position != 0) {
                continue;
            }
            return std::stod(comment.substr(specifier.size()));
        }

        RelearnException::fail("MultipleSubdomainsFromFile::read_neurons_from_file: Did not find comment containing {}", specifier);
    };

    const auto search_contains = [&comments](const std::string& specifier) -> std::string {
        for (const auto& comment : comments) {
            const auto position = comment.find(specifier);
            if (position == std::string::npos) {
                continue;
            }
            return comment;
        }

        RelearnException::fail("MultipleSubdomainsFromFile::read_neurons_from_file: Did not find comment containing {}", specifier);
    };

    const auto parse_coordinates = [](std::string str) -> RelearnTypes::position_type {
        std::ranges::replace(str, '(', ' ');
        std::ranges::replace(str, ')', ' ');
        const auto coords = StringUtil::split_string(str, ',');
        RelearnException::check(coords.size() == 3, "NeuronIO::parse_additional_position_information: Subdomains have invalid coordinates: {}", str);
        const auto x = std::stod(coords[0]);
        const auto y = std::stod(coords[1]);
        const auto z = std::stod(coords[2]);
        return { x, y, z };
    };

    const auto min_x = search("# Minimum x:");
    const auto min_y = search("# Minimum y:");
    const auto min_z = search("# Minimum z:");
    const auto max_x = search("# Maximum x:");
    const auto max_y = search("# Maximum y:");
    const auto max_z = search("# Maximum z:");
    const auto sim_box = RelearnTypes::bounding_box_type{ Vec3d{ min_x, min_y, min_z }, Vec3d{ max_x, max_y, max_z } };

    auto subdomain_strings = search_multiple("# Local subdomain ");
    auto subdomains = std::vector<RelearnTypes::bounding_box_type>{};
    auto subdomain_id_expected = 0;

    for (auto i = 0U; i < subdomain_strings.size(); i++) {
        const auto& subdomain_string = subdomain_strings[i];
        const auto tokens = StringUtil::split_string(subdomain_string, ' ');
        if (!StringUtil::is_number(tokens[0])) {
            continue;
        }

        const auto subdomain_id = std::stoi(tokens[0]);
        RelearnException::check(subdomain_id_expected == subdomain_id, "NeuronIO::parse_additional_position_information: Expected subdomain id {} not {}", i, subdomain_id);
        subdomain_id_expected++;

        const auto i1 = subdomain_string.find('(');
        const auto i2 = subdomain_string.find(')');
        const auto i3 = subdomain_string.find('(', i2 + 1);
        const auto i4 = subdomain_string.find(')', i3 + 1);

        const auto min_subdomain = parse_coordinates(subdomain_string.substr(i1 + 1, i2 - i1));
        const auto max_subdomain = parse_coordinates(subdomain_string.substr(i3 + 1, i4 - i3));

        RelearnException::check(min_subdomain.get_x() < max_subdomain.get_x(), "NeuronIO::parse_additional_position_information: Min subdomain larger or equal than max subdomain");
        RelearnException::check(min_subdomain.get_y() < max_subdomain.get_y(), "NeuronIO::parse_additional_position_information: Min subdomain larger or equal than max subdomain");
        RelearnException::check(min_subdomain.get_z() < max_subdomain.get_z(), "NeuronIO::parse_additional_position_information: Min subdomain larger or equal than max subdomain");

        subdomains.emplace_back(min_subdomain, max_subdomain);
    }

    const auto neurons_str = search_contains(" of ");
    const auto i1 = neurons_str.find(" of ");
    const auto local_neurons = std::stoi(neurons_str.substr(2, i1 - 2));
    const auto total_neurons = std::stoi(neurons_str.substr(i1 + 4));

    return { .sim_size = sim_box, .subdomain_sizes = subdomains, .total_neurons = static_cast<number_neurons_type>(total_neurons), .local_neurons = static_cast<number_neurons_type>(local_neurons) };
}

bool NeuronIO::is_valid_group_name(const RelearnTypes::group_name& group_name) {
    const auto is_valid_char = [](const unsigned char character) {
        return (std::isalpha(character)) || (std::isdigit(character)) || character == '_';
    };

    return std::ranges::find_if(group_name,
                                [&](const unsigned char character) { return !is_valid_char(character); })
               == group_name.end()
           && std::ranges::any_of(group_name,
                                  [&](const unsigned char character) { return std::isalpha(character); });
}

bool NeuronIO::is_valid_neuron_id_value(const std::string& neuron_id) {
    return std::ranges::find_if(neuron_id,
                                [&](const unsigned char character) { return !std::isdigit(character); })
           == neuron_id.end();
}

std::tuple<std::vector<LoadedNeuron>, std::vector<RelearnTypes::group_ids>, RelearnTypes::group_names, LoadedNeuronsInfo, AdditionalPositionInformation> NeuronIO::read_neurons_all_information(const NeuronFilePaths& paths) {
    const auto& [file_path_positions_signals, file_path_groups] = paths;
    auto [neurons, loaded_neurons_info, additional_position_information] = read_neuron_positions_and_signals(file_path_positions_signals);
    const auto number_neurons = neurons.size();
    const auto& [neuron_id_to_group_ids, group_id_to_group_name] = read_neuron_groups(file_path_groups, number_neurons);

    return std::tuple{ neurons, neuron_id_to_group_ids, group_id_to_group_name, loaded_neurons_info, additional_position_information };
}

std::tuple<std::vector<NeuronID>, std::vector<NeuronIO::position_type>, std::vector<RelearnTypes::group_ids>, RelearnTypes::group_names, std::vector<SignalType>, LoadedNeuronsInfo>
NeuronIO::read_neurons_all_information_componentwise(const NeuronFilePaths& paths) {
    const auto& [path_signals_positions, path_groups] = paths;

    auto [ids, positions, signal_types, additional_information] = read_neuron_positions_and_signals_componentwise(path_signals_positions);
    const auto number_neurons = ids.size();
    const auto [neuron_id_to_group_ids, group_id_to_group_name] = read_neuron_groups(path_groups, number_neurons);

    return std::tuple{ ids, positions, neuron_id_to_group_ids, group_id_to_group_name, signal_types, additional_information };
}

std::tuple<std::vector<LoadedNeuron>, LoadedNeuronsInfo, AdditionalPositionInformation> NeuronIO::read_neuron_positions_and_signals(const std::filesystem::path& file_path) {
    RelearnException::check(std::filesystem::is_regular_file(file_path), "NeuronIO::read_neuron_positions_and_signals: Path '{}' is not a file", file_path);
    auto file = std::ifstream(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_neuron_positions_and_signals: Opening the file '{}' was not successful", file_path);

    auto minimum = position_type(std::numeric_limits<position_type::value_type>::max());
    auto maximum = position_type(std::numeric_limits<position_type::value_type>::min());

    auto found_ex_neurons = number_neurons_type{ 0 };
    auto found_in_neurons = number_neurons_type{ 0 };

    auto nodes = std::vector<LoadedNeuron>{};

    auto expected_id = NeuronID::value_type{ 0 };

    auto comments = std::vector<std::string>{};

    for (const auto& line : ranges::getlines(file)) {

        // Skip line with comments
        if (line.empty()) {
            continue;
        }
        if ('#' == line[0]) {
            comments.emplace_back(line);
            continue;
        }

        auto id = NeuronID::value_type{};
        auto pos_x = position_type::value_type{};
        auto pos_y = position_type::value_type{};
        auto pos_z = position_type::value_type{};
        auto signal_type = std::string{};

        auto sstream = std::stringstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> signal_type);

        if (!success) {
            if (!line.starts_with('#')) {
                spdlog::info("Skipping line: {}", line);
            }
            continue;
        }

        RelearnException::check(pos_x >= 0, "NeuronIO::read_neuron_positions_and_signals: x position of neuron {} was negative: {}", id, pos_x);
        RelearnException::check(pos_y >= 0, "NeuronIO::read_neuron_positions_and_signals: y position of neuron {} was negative: {}", id, pos_y);
        RelearnException::check(pos_z >= 0, "NeuronIO::read_neuron_positions_and_signals: z position of neuron {} was negative: {}", id, pos_z);
        RelearnException::check(id >= 1, "NeuronIO::read_neuron_positions_and_signals: neuron id is too small {}", id);

        id--;
        RelearnException::check(id == expected_id, "NeuronIO::read_neuron_positions_and_signals: Loaded neuron with id {} but expected: {}", id, expected_id);

        expected_id++;

        const auto position = position_type{ pos_x, pos_y, pos_z };

        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);

        if (signal_type == "in") {
            found_in_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Inhibitory);
        } else {
            found_ex_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Excitatory);
        }
    }

    const auto additional_position_information = parse_additional_position_information(comments);

    return { std::move(nodes), LoadedNeuronsInfo{ .minimum = minimum, .maximum = maximum, .number_excitatory_neurons = found_ex_neurons, .number_inhibitory_neurons = found_in_neurons }, additional_position_information };
}

std::tuple<std::vector<NeuronID>, std::vector<NeuronIO::position_type>, std::vector<SignalType>, LoadedNeuronsInfo>
NeuronIO::read_neuron_positions_and_signals_componentwise(const std::filesystem::path& file_path) {

    auto file = std::ifstream(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_neuron_positions_and_signals_componentwise: Opening the file '{}' was not successful", file_path);

    auto minimum = position_type(std::numeric_limits<position_type::value_type>::max());
    auto maximum = position_type(std::numeric_limits<position_type::value_type>::min());

    auto found_ex_neurons = number_neurons_type{ 0 };
    auto found_in_neurons = number_neurons_type{ 0 };

    auto ids = std::vector<NeuronID>{};
    auto positions = std::vector<position_type>{};
    auto signal_types = std::vector<SignalType>{};

    auto expected_id = NeuronID::value_type{ 0 };

    for (const auto& line : ranges::getlines(file) | utility::views::filter_not_comment_not_empty_line) {
        auto id = NeuronID::value_type{};
        auto pos_x = position_type::value_type{};
        auto pos_y = position_type::value_type{};
        auto pos_z = position_type::value_type{};
        auto signal_type = std::string{};

        auto sstream = std::stringstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> signal_type);

        if (!success) {
            if (!line.starts_with('#')) {
                spdlog::info("Skipping line: {}", line);
            }
            continue;
        }

        RelearnException::check(pos_x >= 0, "NeuronIO::read_neuron_positions_and_signals_componentwise: x position of neuron {} was negative: {}", id, pos_x);
        RelearnException::check(pos_y >= 0, "NeuronIO::read_neuron_positions_and_signals_componentwise: y position of neuron {} was negative: {}", id, pos_y);
        RelearnException::check(pos_z >= 0, "NeuronIO::read_neuron_positions_and_signals_componentwise: z position of neuron {} was negative: {}", id, pos_z);

        id--;

        RelearnException::check(id == expected_id, "NeuronIO::read_neuron_positions_and_signals_componentwise: Loaded neuron with id {} but expected: {}", id, expected_id);

        expected_id++;

        const auto position = position_type{ pos_x, pos_y, pos_z };

        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);

        ids.emplace_back(false, id);
        positions.emplace_back(position);

        if (signal_type == "in") {
            found_in_neurons++;
            signal_types.emplace_back(SignalType::Inhibitory);
        } else {
            found_ex_neurons++;
            signal_types.emplace_back(SignalType::Excitatory);
        }
    }

    return { std::move(ids), std::move(positions), std::move(signal_types), LoadedNeuronsInfo{ .minimum = minimum, .maximum = maximum, .number_excitatory_neurons = found_ex_neurons, .number_inhibitory_neurons = found_in_neurons } };
}

std::tuple<std::vector<RelearnTypes::group_ids>, RelearnTypes::group_names> NeuronIO::read_neuron_groups(const std::filesystem::path& file_path, const RelearnTypes::number_neurons_type number_neurons) {
    auto file = std::ifstream(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_neuron_groups: Opening the file '{}' was not successful", file_path);

    auto neuron_id_to_group_ids = std::vector<RelearnTypes::group_ids>(number_neurons, { Constants::default_group_id });
    auto group_id_to_group_name = RelearnTypes::group_names{ std::string{ Constants::default_group_name } };
    for (const auto& line : ranges::getlines(file) | utility::views::filter_not_comment_not_empty_line) {
        if (line.starts_with('#')) {
            continue;
        }
        const auto& data = StringUtil::split_string(line, ' ');
        if (!is_valid_group_name(data[0])) {
            RelearnException::check(is_valid_neuron_id_value(data[0]), "NeuronIO::read_neuron_groups: First token in line is neither group name nor neuron id! ({})", data[0]);
            const auto neuron_id = boost::lexical_cast<NeuronID::value_type>(data[0]) - 1;
            RelearnException::check(neuron_id < number_neurons, "NeuronIO::read_neuron_groups: Read neuron id is too large! Must be smaller than number neurons: (neuron id after decrement: {}, number neurons: {})", neuron_id, number_neurons);

            for (auto i = 1UL; i < data.size(); ++i) {
                const auto& group_name = data[i];
                const auto group_id_it = ranges::find(group_id_to_group_name, group_name);
                auto group_id = RelearnTypes::group_id{ 0 };
                if (group_id_it == group_id_to_group_name.end()) {
                    RelearnException::check(is_valid_group_name(group_name), "NeuronIO::read_neuron_groups: Group name {} is not valid.", group_name);
                    group_id_to_group_name.push_back(group_name);
                    group_id = group_id_to_group_name.size() - 1;
                } else {
                    group_id = static_cast<RelearnTypes::group_id>(ranges::distance(group_id_to_group_name.begin(), group_id_it));
                }
                neuron_id_to_group_ids[neuron_id].push_back(group_id);
            }
        } else {
            const auto group_name = data[0];
            const auto group_id_it = ranges::find(group_id_to_group_name, group_name);

            auto group_id = RelearnTypes::group_id{ 0 };
            if (group_id_it == group_id_to_group_name.end()) {
                group_id_to_group_name.push_back(group_name);
                group_id = group_id_to_group_name.size() - 1;
            } else {
                group_id = static_cast<RelearnTypes::group_id>(ranges::distance(group_id_to_group_name.begin(), group_id_it));
            }

            for (auto i = 1UL; i < data.size(); ++i) {
                RelearnException::check(is_valid_neuron_id_value(data[i]), "NeuronIO::read_neuron_groups: First token in line is neither group name nor neuron id! ({})", data[i]);
                const auto neuron_id = boost::lexical_cast<NeuronID::value_type>(data[i]) - 1;
                RelearnException::check(neuron_id < number_neurons, "NeuronIO::read_neuron_groups: Read neuron id is too large! Must be smaller than number neurons: (neuron id after decrement: {}, number neurons: {})", neuron_id, number_neurons);
                neuron_id_to_group_ids[neuron_id].push_back(group_id);
            }
        }
    }
    return { neuron_id_to_group_ids, group_id_to_group_name };
}

void NeuronIO::write_neurons(const std::vector<LoadedNeuron>& neurons, const NeuronFilePaths& paths, const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    write_neurons(neurons, paths, local_group_translator, nullptr);
}

void NeuronIO::write_neurons(const std::vector<LoadedNeuron>& neurons, const NeuronFilePaths& paths, const std::shared_ptr<LocalGroupTranslator>& local_group_translator, const std::shared_ptr<Partition>& partition) {
    const auto& [file_path_positions_signals, file_path_groups] = paths;
    write_neuron_positions_and_signals(neurons, file_path_positions_signals, partition);
    write_neuron_groups(file_path_groups, local_group_translator);
}

void NeuronIO::write_neuron_positions_and_signals(const std::vector<LoadedNeuron>& neurons, std::stringstream& sstream, const std::shared_ptr<Partition>& partition) {
    auto sim_box = RelearnTypes::bounding_box_type{};
    auto subdomain_boxes = std::vector<RelearnTypes::bounding_box_type>{};
    auto total_number_neurons = RelearnTypes::number_neurons_type{ 0 };

    auto number_local_subdomains = std::size_t{ 0 };
    auto first_local_subdomain_index = std::size_t{ 0 };
    auto last_local_subdomain_index = std::size_t{ 0 };

    if (partition != nullptr) {
        total_number_neurons = partition->get_total_number_neurons();

        sim_box = partition->get_simulation_box_size();

        number_local_subdomains = partition->get_number_local_subdomains();
        first_local_subdomain_index = partition->get_local_subdomain_id_start();
        last_local_subdomain_index = partition->get_local_subdomain_id_end();

        for (auto i = first_local_subdomain_index; i <= last_local_subdomain_index; i++) {
            subdomain_boxes.push_back(partition->get_subdomain_boundaries(i));
        }

    } else {
        auto min_x = std::numeric_limits<double>::max();
        auto min_y = std::numeric_limits<double>::max();
        auto min_z = std::numeric_limits<double>::max();
        auto max_x = std::numeric_limits<double>::min();
        auto max_y = std::numeric_limits<double>::min();
        auto max_z = std::numeric_limits<double>::min();

        for (const auto& neuron : neurons) {
            const auto& [x, y, z] = neuron.pos;
            min_x = std::min(x, min_x);
            max_x = std::max(x, max_x);
            min_y = std::min(y, min_y);
            max_y = std::max(y, max_y);
            min_z = std::min(z, min_z);
            max_z = std::max(z, max_z);
        }

        if (min_x - max_x < Constants::eps) {
            max_x++;
        }
        if (min_y - max_y < Constants::eps) {
            max_y++;
        }
        if (min_y - max_y < Constants::eps) {
            max_z++;
        }

        sim_box = RelearnTypes::bounding_box_type{ { min_x, min_y, min_z }, { max_x, max_y, max_z } };
        subdomain_boxes.push_back(sim_box);

        number_local_subdomains = 1;
        first_local_subdomain_index = 0;
        last_local_subdomain_index = 0;

        total_number_neurons = neurons.size();
    }

    sstream << std::setprecision(std::numeric_limits<double>::digits10);

    // Write total number of neurons to log file
    sstream << "# " << neurons.size() << " of " << total_number_neurons << '\n';
    sstream << "# Minimum x: " << sim_box.get_minimum().get_x() << '\n';
    sstream << "# Minimum y: " << sim_box.get_minimum().get_y() << '\n';
    sstream << "# Minimum z: " << sim_box.get_minimum().get_z() << '\n';
    sstream << "# Maximum x: " << sim_box.get_maximum().get_x() << '\n';
    sstream << "# Maximum y: " << sim_box.get_maximum().get_y() << '\n';
    sstream << "# Maximum z: " << sim_box.get_maximum().get_z() << '\n';
    sstream << "# <local id> <pos x> <pos y> <pos z> <type>\n";

    sstream << "# Local subdomain index start: " << first_local_subdomain_index << "\n";
    sstream << "# Local subdomain index end: " << last_local_subdomain_index << "\n";
    sstream << "# Number of local subdomains: " << number_local_subdomains << "\n";

    for (auto i = 0U; i < subdomain_boxes.size(); i++) {
        const auto local_subdomain_index = first_local_subdomain_index + i;
        const auto& subdomain_bb = subdomain_boxes[i];
        sstream << "# Local subdomain " << local_subdomain_index << " boundaries (" << subdomain_bb.get_minimum().get_x() << ", " << subdomain_bb.get_minimum().get_y() << ", " << subdomain_bb.get_minimum().get_z() << ") - (";
        sstream << subdomain_bb.get_maximum().get_x() << ", " << subdomain_bb.get_maximum().get_y() << ", " << subdomain_bb.get_maximum().get_z() << ")\n";
    }

    for (const auto& neuron : neurons) {
        const auto& [x, y, z] = neuron.pos;
        const auto& signal_type_name = (neuron.signal_type == SignalType::Excitatory) ? "ex" : "in";

        sstream << fmt::format("{1:<} {2:<.{0}} {3:<.{0}} {4:<.{0}} {5:<}",
                               Constants::print_precision, (neuron.id.get_neuron_id() + 1), x, y, z, signal_type_name)
                << '\n';
    }
}

void NeuronIO::write_neuron_positions_and_signals(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path, const std::shared_ptr<Partition>& partition) {
    auto sstream = std::stringstream{};
    auto ofs = std::ofstream(file_path, std::ios::binary | std::ios::out);

    const auto is_good = ofs.good();
    const auto is_bad = ofs.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons: The output file was bad, {}", file_path);

    write_neuron_positions_and_signals(neurons, sstream, partition);

    ofs << sstream.str();
    ofs.close();
}

void NeuronIO::write_neuron_groups(std::stringstream& sstream, const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    sstream << "# <group_name> <neuron_ids>\n";
    const auto& group_names = local_group_translator->get_all_group_names();
    for (auto group_id = 1UL; group_id < group_names.size(); ++group_id) { // start in group 1 because group 0 is default group which we ignore
        const auto& group_name = group_names[group_id];
        const auto& neuron_ids = local_group_translator->get_neuron_ids_in_group(group_id);

        sstream << group_name;

        for (const auto neuron_id : neuron_ids) {
            sstream << " " << (neuron_id.get_neuron_id() + 1);
        }

        sstream << '\n';
    }
}

void NeuronIO::write_neuron_groups(const std::filesystem::path& file_path, const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    auto sstream = std::stringstream{};
    auto ofs = std::ofstream(file_path, std::ios::binary | std::ios::out);

    const auto is_good = ofs.good();
    const auto is_bad = ofs.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neuron_groups: The file is bad, {}", file_path);

    write_neuron_groups(sstream, local_group_translator);

    ofs << sstream.str();
    ofs.close();
}

void NeuronIO::write_neuron_groups_of_specific_neurons(const std::span<const NeuronID> ids, std::stringstream& sstream, const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    sstream << "# <group_name> <neuron_ids>\n";
    const auto& group_names = local_group_translator->get_all_group_names();
    for (auto group_id = 1UL; group_id < group_names.size(); ++group_id) { // start in group 1 because group 0 is default group which we ignore
        const auto& group_name = group_names[group_id];
        const auto& neuron_ids = local_group_translator->get_neuron_ids_in_group(group_id);

        sstream << group_name;

        for (const auto neuron_id : neuron_ids) {
            if (ranges::find(ids.begin(), ids.end(), neuron_id) != ids.end()) {
                sstream << " " << (neuron_id.get_neuron_id() + 1);
            }
        }

        sstream << '\n';
    }
}

void NeuronIO::write_neuron_groups_of_specific_neurons(const std::span<const NeuronID> ids, const std::filesystem::path& file_path, const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    auto sstream = std::stringstream{};
    auto ofs = std::ofstream(file_path, std::ios::binary | std::ios::out);

    const auto is_good = ofs.good();
    const auto is_bad = ofs.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neuron_groups_of_specific_neurons: The file is bad, {}", file_path);

    write_neuron_groups_of_specific_neurons(ids, sstream, local_group_translator);

    ofs << sstream.str();
    ofs.close();
}

void NeuronIO::write_group_names(std::stringstream& sstream, const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    sstream << "# <group id>\t<group_name>\t<num_neurons_in_group>\n";

    const auto num_groups = local_group_translator->get_number_of_groups();

    for (auto group_id = 0U; group_id < num_groups; group_id++) {
        const auto& group_name = local_group_translator->get_group_name_for_group_id(group_id);

        sstream << group_id << '\t' << group_name << '\t' << local_group_translator->get_number_neurons_in_group(group_id) << '\n';
    }
}

void NeuronIO::write_group_name_to_file_name(std::stringstream& sstream, const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    sstream << "# <group name>\t<file name>\n";

    const auto& my_rank_str = mpiPP::MPIInfo::get_my_rank_str();

    const auto num_groups = local_group_translator->get_number_of_groups();

    for (auto group_id = 0U; group_id < num_groups; group_id++) {
        const auto& group_name = local_group_translator->get_group_name_for_group_id(group_id);

        sstream << group_name << '\t' << my_rank_str << "_group_" << group_id << ".csv\n";
    }
}

void NeuronIO::write_neuron_positions_and_signals_componentwise(const std::span<const NeuronID> ids, const std::span<const position_type> positions,
                                                                const std::span<const SignalType> signal_types, std::stringstream& sstream, std::size_t total_number_neurons, RelearnTypes::bounding_box_type simulation_box,
                                                                std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries) {

    const auto size_ids = ids.size();
    const auto size_positions = positions.size();
    const auto size_signal_types = signal_types.size();

    if (simulation_box.get_minimum().get_x() == simulation_box.get_maximum().get_x()) {
        auto min_x = std::numeric_limits<double>::max();
        auto min_y = std::numeric_limits<double>::max();
        auto min_z = std::numeric_limits<double>::max();
        auto max_x = std::numeric_limits<double>::min();
        auto max_y = std::numeric_limits<double>::min();
        auto max_z = std::numeric_limits<double>::min();

        for (const auto& [x, y, z] : positions) {
            min_x = std::min(x, min_x);
            max_x = std::max(x, max_x);
            min_y = std::min(y, min_y);
            max_y = std::max(y, max_y);
            min_z = std::min(z, min_z);
            max_z = std::max(z, max_z);
        }
        simulation_box = RelearnTypes::bounding_box_type{ { min_x, min_y, min_z }, { max_x, max_y, max_z } };
    }

    if (local_subdomain_boundaries.empty()) {
        local_subdomain_boundaries.push_back(simulation_box);
    }

    const auto all_same_size = size_ids == size_positions && size_ids == size_signal_types;

    RelearnException::check(all_same_size, "NeuronIO::write_neuron_positions_and_signals_componentwise: The vectors had different sizes.");

    // Write total number of neurons to log file
    if (total_number_neurons > 0) {
        sstream << "# " << ids.size() << " of " << total_number_neurons << '\n';
    }

    sstream << std::setprecision(std::numeric_limits<double>::digits10);
    const auto& [simulation_box_min, simulation_box_max] = simulation_box;
    const auto& [min_x, min_y, min_z] = simulation_box_min;
    const auto& [max_x, max_y, max_z] = simulation_box_max;

    sstream << "# Minimum x: " << min_x << '\n';
    sstream << "# Minimum y: " << min_y << '\n';
    sstream << "# Minimum z: " << min_z << '\n';
    sstream << "# Maximum x: " << max_x << '\n';
    sstream << "# Maximum y: " << max_y << '\n';
    sstream << "# Maximum z: " << max_z << '\n';
    sstream << "# <local id> <pos x> <pos y> <pos z> <type>\n";

    const auto number_local_subdomains = local_subdomain_boundaries.size();
    sstream << "# Number of local subdomains: " << number_local_subdomains << "\n";

    for (auto local_subdomain_index = 0U; local_subdomain_index < number_local_subdomains; local_subdomain_index++) {
        const auto& [subdomain_bounding_box_min, subdomain_bounding_box_max] = local_subdomain_boundaries[local_subdomain_index];
        sstream << "# Local subdomain " << local_subdomain_index << " boundaries (" << subdomain_bounding_box_min.get_x() << ", " << subdomain_bounding_box_min.get_y() << ", " << subdomain_bounding_box_min.get_z() << ") - (";
        sstream << subdomain_bounding_box_max.get_x() << ", " << subdomain_bounding_box_max.get_y() << ", " << subdomain_bounding_box_max.get_z() << ")\n";
    }

    for (const auto& neuron_id : ids) {
        RelearnException::check(neuron_id.get_neuron_id() < ids.size(), "NeuronIO::write_neuron_positions_and_signals_componentwise: Neuron id {} is too large", neuron_id);
        const auto& [x, y, z] = positions[neuron_id.get_neuron_id()];
        const std::string_view signal_type_name = (signal_types[neuron_id.get_neuron_id()] == SignalType::Excitatory) ? "ex" : "in";
        sstream << (neuron_id.get_neuron_id() + 1) << " " << x << " " << y << " " << z << " " << signal_type_name << '\n';
    }
}

void NeuronIO::write_neuron_positions_and_signals_componentwise(const std::span<const NeuronID> ids, const std::span<const position_type> positions,
                                                                const std::span<const SignalType> signal_types, const std::filesystem::path& file_path, std::size_t total_number_neurons, RelearnTypes::bounding_box_type simulation_box,
                                                                std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries) {
    auto sstream = std::stringstream{};
    write_neuron_positions_and_signals_componentwise(ids, positions, signal_types, sstream, total_number_neurons, simulation_box, std::move(local_subdomain_boundaries));
    auto ofs = std::ofstream(file_path, std::ios::binary | std::ios::out);

    const auto is_good = ofs.good();
    const auto is_bad = ofs.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neuron_positions_and_signals_componentwise: The file is bad, {}", file_path);

    ofs << sstream.str();
    ofs.close();
}

void NeuronIO::write_neuron_positions_and_signals_componentwise(const std::span<const NeuronID> ids, const std::span<const position_type> positions,
                                                                const std::span<const SignalType> signal_types, const std::filesystem::path& file_path) {
    write_neuron_positions_and_signals_componentwise(ids, positions, signal_types, file_path, 0, {}, {});
}

void NeuronIO::write_neurons_componentwise(const std::span<const NeuronID> ids, const std::span<const position_type> positions,
                                           const std::shared_ptr<LocalGroupTranslator>& local_group_translator, const std::span<const SignalType> signal_types, const NeuronFilePaths& paths) {
    const auto& [file_path_positions_signals, file_path_groups] = paths;
    write_neuron_positions_and_signals_componentwise(ids, positions, signal_types, file_path_positions_signals, 0, {}, {});
    write_neuron_groups_of_specific_neurons(ids, file_path_groups, local_group_translator);
}

void NeuronIO::write_neurons_componentwise(std::span<const NeuronID> ids, std::span<const position_type> positions,
                                           const std::shared_ptr<LocalGroupTranslator>& local_group_translator, std::span<const SignalType> signal_types, const NeuronFilePaths& paths,
                                           std::size_t total_number_neurons, RelearnTypes::bounding_box_type simulation_box,
                                           std::vector<RelearnTypes::bounding_box_type> local_subdomain_boundaries) {
    const auto& [file_path_positions_signals, file_path_groups] = paths;
    write_neuron_positions_and_signals_componentwise(ids, positions, signal_types, file_path_positions_signals, total_number_neurons, simulation_box, std::move(local_subdomain_boundaries));
    write_neuron_groups_of_specific_neurons(ids, file_path_groups, local_group_translator);
}

std::optional<std::vector<NeuronID>> NeuronIO::read_neuron_ids(const std::filesystem::path& file_path) {
    auto local_file = std::ifstream(file_path);

    const auto file_is_good = local_file.good();
    const auto file_is_not_good = local_file.fail() || local_file.eof();

    if (!file_is_good || file_is_not_good) {
        return {};
    }

    auto ids = std::vector<NeuronID>{};

    for (const auto& line : ranges::getlines(local_file) | utility::views::filter_not_comment_not_empty_line) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        auto id = NeuronID::value_type{};
        auto pos_x = position_type::value_type{};
        auto pos_y = position_type::value_type{};
        auto pos_z = position_type::value_type{};
        auto signal_type = std::string{};

        auto sstream = std::stringstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> signal_type);

        if (!success) {
            return {};
        }

        id--;

        if (!ids.empty()) {
            const auto last_id = ids[ids.size() - 1].get_neuron_id();

            if (last_id + 1 != id) {
                return {};
            }
        }

        ids.emplace_back(false, id);
    }

    return ids;
}

NeuronIO::InSynapses NeuronIO::read_in_synapses(const std::filesystem::path& file_path,
                                                number_neurons_type number_local_neurons, mpiPP::MPIRank my_rank, std::size_t number_mpi_ranks) {
    auto local_in_synapses_static = StaticLocalSynapses{};
    auto distant_in_synapses_static = StaticDistantInSynapses{};
    auto local_in_synapses_plastic = PlasticLocalSynapses{};
    auto distant_in_synapses_plastic = PlasticDistantInSynapses{};

    auto file_synapses = std::ifstream(file_path, std::ios::binary | std::ios::in);

    const auto is_good = file_synapses.good();
    const auto is_bad = file_synapses.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::read_in_synapses: The ofstream failed to open '{}'", file_path.string());

    for (const auto& line : ranges::getlines(file_synapses) | utility::views::filter_not_comment_not_empty_line) {
        auto read_target_rank = 0;
        auto read_target_id = NeuronID::value_type{ 0 };

        auto read_source_rank = 0;
        auto read_source_id = NeuronID::value_type{ 0 };

        auto weight = RelearnTypes::static_synapse_weight{ 0 };
        auto plastic = false;

        auto sstream = std::stringstream(line);
        const auto success = (sstream >> read_target_rank) && (sstream >> read_target_id) && (sstream >> read_source_rank) && (sstream >> read_source_id) && (sstream >> weight) && (sstream >> plastic);

        if (!success) {
            if (!line.starts_with('#')) {
                spdlog::info("Skipping line: {}", line);
            }
            continue;
        }

        RelearnException::check(read_target_rank == my_rank.get_rank(), "NeuronIO::read_in_synapses: target_rank is not equal to my_rank: {} vs {}", read_target_rank, my_rank);
        RelearnException::check(read_target_id > 0 && read_target_id <= number_local_neurons, "NeuronIO::read_in_synapses: target_id was not from [1, {}]: {}", number_local_neurons, read_target_id);

        RelearnException::check(read_source_rank < static_cast<int>(number_mpi_ranks), "NeuronIO::read_in_synapses: source rank is not smaller than the number of mpi ranks: {} vs {}", read_source_rank, number_mpi_ranks);

        RelearnException::check(weight != 0, "NeuronIO::read_in_synapses: weight was 0");

        // The neurons start with 1
        --read_source_id;
        --read_target_id;

        auto source_id = NeuronID{ false, read_source_id };
        auto target_id = NeuronID{ false, read_target_id };

        if (read_source_rank != my_rank.get_rank()) {
            if (plastic) {
                distant_in_synapses_plastic.emplace_back(target_id, RankNeuronId{ mpiPP::MPIRank(read_source_rank), source_id }, static_cast<RelearnTypes::plastic_synapse_weight>(weight));
            } else {
                distant_in_synapses_static.emplace_back(target_id, RankNeuronId{ mpiPP::MPIRank(read_source_rank), source_id }, weight);
            }
        } else {
            // if (target_id == source_id) {
            //     spdlog::info("Skipping line: {}", line);
            //     continue;
            // }

            if (plastic) {
                local_in_synapses_plastic.emplace_back(target_id, source_id, static_cast<RelearnTypes::plastic_synapse_weight>(weight));
            } else {
                local_in_synapses_static.emplace_back(target_id, source_id, weight);
            }
        }
    }

    return { { local_in_synapses_static, distant_in_synapses_static }, { local_in_synapses_plastic, distant_in_synapses_plastic } };
}

NeuronIO::OutSynapses NeuronIO::read_out_synapses(const std::filesystem::path& file_path,
                                                  number_neurons_type number_local_neurons, mpiPP::MPIRank my_rank, std::size_t number_mpi_ranks) {
    auto local_out_synapses_static = StaticLocalSynapses{};
    auto distant_out_synapses_static = StaticDistantOutSynapses{};
    auto local_out_synapses_plastic = PlasticLocalSynapses{};
    auto distant_out_synapses_plastic = PlasticDistantOutSynapses{};

    auto file_synapses = std::ifstream(file_path, std::ios::binary | std::ios::in);

    const auto is_good = file_synapses.good();
    const auto is_bad = file_synapses.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::read_out_synapses: The ofstream failed to open");

    for (const auto& line : ranges::getlines(file_synapses) | utility::views::filter_not_comment_not_empty_line) {
        auto read_target_rank = 0;
        auto read_target_id = NeuronID::value_type{ 0 };

        auto read_source_rank = 0;
        auto read_source_id = NeuronID::value_type{ 0 };

        auto weight = RelearnTypes::static_synapse_weight{ 0 };
        auto plastic = false;

        auto sstream = std::stringstream(line);
        const auto success = (sstream >> read_target_rank) && (sstream >> read_target_id) && (sstream >> read_source_rank) && (sstream >> read_source_id) && (sstream >> weight) && (sstream >> plastic);

        if (!success) {
            if (!line.starts_with('#')) {
                spdlog::info("Skipping line: {}", line);
            }
            continue;
        }

        RelearnException::check(read_source_rank == my_rank.get_rank(), "NeuronIO::read_out_synapses: source_rank is not equal to my_rank: {} vs {}", read_target_rank, my_rank);
        RelearnException::check(read_source_id > 0 && read_source_id <= number_local_neurons, "NeuronIO::read_out_synapses: source_id was not from [1, {}]: {}", number_local_neurons, read_source_id);

        RelearnException::check(read_target_rank < static_cast<int>(number_mpi_ranks), "NeuronIO::read_out_synapses: target rank is not smaller than the number of mpi ranks: {} vs {}", read_source_rank, number_mpi_ranks);

        RelearnException::check(weight != 0, "NeuronIO::read_out_synapses: weight was 0");

        // The neurons start with 1
        --read_source_id;
        --read_target_id;

        auto source_id = NeuronID{ false, read_source_id };
        auto target_id = NeuronID{ false, read_target_id };

        if (read_target_rank != my_rank.get_rank()) {
            if (plastic) {
                distant_out_synapses_plastic.emplace_back(RankNeuronId{ mpiPP::MPIRank(read_target_rank), target_id }, source_id, static_cast<RelearnTypes::plastic_synapse_weight>(weight));
            } else {
                distant_out_synapses_static.emplace_back(RankNeuronId{ mpiPP::MPIRank(read_target_rank), target_id }, source_id, weight);
            }
        } else {
            // if (target_id == source_id) {
            //     spdlog::info("Skipping line: {}", line);
            //     continue;
            // }

            if (plastic) {
                local_out_synapses_plastic.emplace_back(target_id, source_id, static_cast<RelearnTypes::plastic_synapse_weight>(weight));
            } else {
                local_out_synapses_static.emplace_back(target_id, source_id, weight);
            }
        }
    }

    return { { local_out_synapses_static, distant_out_synapses_static }, { local_out_synapses_plastic, distant_out_synapses_plastic } };
}

void NeuronIO::write_out_synapses(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>& local_out_edges_static,
                                  const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>& distant_out_edges_static,
                                  const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_out_edges_plastic,
                                  const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_out_edges_plastic,
                                  const mpiPP::MPIRank my_rank, const std::size_t mpi_ranks, const RelearnTypes::number_neurons_type number_local_neurons, const RelearnTypes::number_neurons_type number_total_neurons,
                                  std::stringstream& sstream, const std::size_t step) {
    const auto is_good = sstream.good();
    const auto is_bad = sstream.bad();

    const auto my_rank_int = my_rank.get_rank();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_distant_out_synapses: The ofstream failed to open");

    sstream << "# Total number neurons: " << number_total_neurons << '\n';
    sstream << "# Local number neurons: " << number_local_neurons << '\n';
    sstream << "# Number MPI ranks: " << mpi_ranks << '\n';
    sstream << "# Current simulation step: " << step << '\n';
    sstream << "# <target rank> <target neuron id>\t<source rank> <source neuron id>\t<weight>\t<plastic>\n";

    for (const auto& source_local_id : NeuronID::range_id(number_local_neurons)) {
        for (const auto& [target_id, weight] : local_out_edges_static[source_local_id]) {
            const auto& target_local_id = target_id.get_neuron_id();

            sstream << my_rank_int << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '0' << '\n';
        }

        for (const auto& [target_neuron, weight] : distant_out_edges_static[source_local_id]) {
            const auto& [target_rank, target_id] = target_neuron;
            const auto& target_local_id = target_id.get_neuron_id();

            RelearnException::check(target_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);
            sstream << target_rank.get_rank() << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '0' << '\n';
        }

        for (const auto& [target_id, weight] : local_out_edges_plastic[source_local_id]) {
            const auto& target_local_id = target_id.get_neuron_id();

            sstream << my_rank_int << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '1' << '\n';
        }

        for (const auto& [target_neuron, weight] : distant_out_edges_plastic[source_local_id]) {
            const auto& [target_rank, target_id] = target_neuron;
            const auto& target_local_id = target_id.get_neuron_id();

            RelearnException::check(target_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);
            sstream << target_rank.get_rank() << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '1' << '\n';
        }
    }
}

void NeuronIO::write_in_synapses(const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>& local_in_edges_static,
                                 const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>& distant_in_edges_static,
                                 const std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>& local_in_edges_plastic,
                                 const std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>& distant_in_edges_plastic,
                                 const mpiPP::MPIRank my_rank, const std::size_t mpi_ranks, const RelearnTypes::number_neurons_type number_local_neurons, const RelearnTypes::number_neurons_type number_total_neurons,
                                 std::stringstream& sstream, const std::size_t step) {
    const auto is_good = sstream.good();
    const auto is_bad = sstream.bad();

    const auto my_rank_int = my_rank.get_rank();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_distant_out_synapses: The ofstream failed to open");

    sstream << "# Total number neurons: " << number_total_neurons << '\n';
    sstream << "# Local number neurons: " << number_local_neurons << '\n';
    sstream << "# Number MPI ranks: " << mpi_ranks << '\n';
    sstream << "# Current simulation step: " << step << '\n';
    sstream << "# <target rank> <target neuron id>\t<source rank> <source neuron id>\t<weight>\t<plastic>\n";

    for (const auto& target_local_id : NeuronID::range_id(number_local_neurons)) {
        for (const auto& [source_id, weight] : local_in_edges_static[target_local_id]) {
            const auto& source_local_id = source_id.get_neuron_id();

            sstream << my_rank_int << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '0' << '\n';
        }

        for (const auto& [source_neuron, weight] : distant_in_edges_static[target_local_id]) {
            const auto& [source_rank, source_id] = source_neuron;
            const auto& source_local_id = source_id.get_neuron_id();

            RelearnException::check(source_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);
            sstream << my_rank_int << ' ' << (target_local_id + 1) << '\t' << source_rank.get_rank() << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '0' << '\n';
        }

        for (const auto& [source_id, weight] : local_in_edges_plastic[target_local_id]) {
            const auto& source_local_id = source_id.get_neuron_id();

            sstream << my_rank_int << ' ' << (target_local_id + 1) << '\t' << my_rank_int << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '1' << '\n';
        }

        for (const auto& [source_neuron, weight] : distant_in_edges_plastic[target_local_id]) {
            const auto& [source_rank, source_id] = source_neuron;
            const auto& source_local_id = source_id.get_neuron_id();

            RelearnException::check(source_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);
            sstream << my_rank_int << ' ' << (target_local_id + 1) << '\t' << source_rank.get_rank() << ' ' << (source_local_id + 1) << '\t' << weight << '\t' << '1' << '\n';
        }
    }
}

void NeuronIO::write_out_synapses(const StaticLocalSynapses& local_out_synapses_static, const StaticDistantOutSynapses& distant_out_synapses_static, const PlasticLocalSynapses& local_out_synapses_plastic,
                                  const PlasticDistantOutSynapses& distant_out_synapses_plastic, mpiPP::MPIRank my_rank, RelearnTypes::number_neurons_type number_neurons, const std::filesystem::path& file_path) {
    auto local_neighborhood_static = std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>{};
    local_neighborhood_static.resize(number_neurons);
    auto distant_neighborhood_static = std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>{};
    distant_neighborhood_static.resize(number_neurons);
    auto local_neighborhood_plastic = std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>{};
    local_neighborhood_plastic.resize(number_neurons);
    auto distant_neighborhood_plastic = std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>{};
    distant_neighborhood_plastic.resize(number_neurons);

    for (const auto& [target_id, source_id, weight] : local_out_synapses_static) {
        local_neighborhood_static[source_id.get_neuron_id()].emplace_back(std::make_pair(target_id.get_neuron_id(), weight));
    }
    for (const auto& [target_id, source_id, weight] : local_out_synapses_plastic) {
        local_neighborhood_plastic[source_id.get_neuron_id()].emplace_back(std::make_pair(target_id.get_neuron_id(), weight));
    }
    for (const auto& [target_id, source_id, weight] : distant_out_synapses_static) {
        distant_neighborhood_static[source_id.get_neuron_id()].emplace_back(std::make_pair(target_id, weight));
    }
    for (const auto& [target_id, source_id, weight] : distant_out_synapses_plastic) {
        distant_neighborhood_plastic[source_id.get_neuron_id()].emplace_back(std::make_pair(target_id, weight));
    }

    auto sstream = std::stringstream{};
    write_out_synapses(local_neighborhood_static, distant_neighborhood_static, local_neighborhood_plastic, distant_neighborhood_plastic, my_rank, 1, number_neurons, number_neurons, sstream, 0);
    auto ofs = std::ofstream(file_path, std::ios::binary | std::ios::out);
    const auto is_good = ofs.good();
    const auto is_bad = ofs.bad();
    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons_to_file: The ofstream failed to open '{}'", file_path);
    ofs << sstream.str();
    ofs.close();
}

void NeuronIO::write_in_synapses(const StaticLocalSynapses& local_in_synapses_static, const StaticDistantInSynapses& distant_in_synapses_static, const PlasticLocalSynapses& local_in_synapses_plastic,
                                 const PlasticDistantInSynapses& distant_in_synapses_plastic, mpiPP::MPIRank my_rank, RelearnTypes::number_neurons_type number_neurons, const std::filesystem::path& file_path) {
    auto local_neighborhood_static = std::vector<std::vector<std::pair<NeuronID, RelearnTypes::static_synapse_weight>>>{};
    local_neighborhood_static.resize(number_neurons);
    auto distant_neighborhood_static = std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>>{};
    distant_neighborhood_static.resize(number_neurons);
    auto local_neighborhood_plastic = std::vector<std::vector<std::pair<NeuronID, RelearnTypes::plastic_synapse_weight>>>{};
    local_neighborhood_plastic.resize(number_neurons);
    auto distant_neighborhood_plastic = std::vector<std::vector<std::pair<RankNeuronId, RelearnTypes::plastic_synapse_weight>>>{};
    distant_neighborhood_plastic.resize(number_neurons);

    for (const auto& [target_id, source_id, weight] : local_in_synapses_static) {
        local_neighborhood_static[target_id.get_neuron_id()].emplace_back(std::make_pair(source_id.get_neuron_id(), weight));
    }
    for (const auto& [target_id, source_id, weight] : local_in_synapses_plastic) {
        local_neighborhood_plastic[target_id.get_neuron_id()].emplace_back(std::make_pair(source_id.get_neuron_id(), weight));
    }
    for (const auto& [target_id, source_id, weight] : distant_in_synapses_static) {
        distant_neighborhood_static[target_id.get_neuron_id()].emplace_back(std::make_pair(source_id, weight));
    }
    for (const auto& [target_id, source_id, weight] : distant_in_synapses_plastic) {
        distant_neighborhood_plastic[target_id.get_neuron_id()].emplace_back(std::make_pair(source_id, weight));
    }

    auto sstream = std::stringstream{};
    write_in_synapses(local_neighborhood_static, distant_neighborhood_static, local_neighborhood_plastic, distant_neighborhood_plastic, my_rank, 1, number_neurons, number_neurons, sstream, 0);
    auto ofs = std::ofstream(file_path, std::ios::binary | std::ios::out);
    const auto is_good = ofs.good();
    const auto is_bad = ofs.bad();
    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons_to_file: The ofstream failed to open '{}'", file_path);
    ofs << sstream.str();
    ofs.close();
}
