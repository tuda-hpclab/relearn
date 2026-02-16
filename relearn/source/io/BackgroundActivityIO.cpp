/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "BackgroundActivityIO.h"

#include "Types.h"

#include "io/parser/MonitorParser.h"
#include "neurons/helper/CachedChoiceFunction.h"
#include "neurons/input/ConstantActivityInput.h"
#include "neurons/input/NormalActivityInput.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/ranges/views/IO.hpp"

#include "mpi-wrapper/MPIRank.h"

#include <boost/lexical_cast.hpp>
#include <range/v3/action/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/transform.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

std::pair<std::vector<std::shared_ptr<ActivityInput>>, std::unique_ptr<ChoiceFunction>> BackgroundActivityIO::load_background_activity(const std::filesystem::path& file_path, const mpiPP::MPIRank my_rank, const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    auto file = std::ifstream{ file_path };

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    const auto& parse_line = [my_rank, local_group_translator](const std::string& line) {
        auto sstream = std::stringstream(line);

        RelearnTypes::step_type begin{};
        std::int64_t end{};
        std::string input_descr{};

        const auto success = (sstream >> begin) && (sstream >> end) && (sstream >> input_descr);
        RelearnException::check(success, "BackgroundActivityIO::load_background_activity: Did not succeed in parsing.");

        auto descriptions = std::vector<std::string>{};

        for (auto current_value = std::string{}; sstream >> current_value;) {
            descriptions.emplace_back(current_value);
        }
        const auto& parsed_ids = MonitorParser::parse_my_ids(descriptions, my_rank, local_group_translator);

        if (end < 0) {
            return std::tuple{ begin, std::numeric_limits<RelearnTypes::step_type>::max(), input_descr, parsed_ids };
        }

        return std::tuple{ begin, utility::save_cast<RelearnTypes::step_type>(end), input_descr, parsed_ids };
    };

    RelearnException::check(file_is_good && !file_is_not_good,
                            "BackgroundActivityIO::load_background_activity: Opening the file '{}' was not successful", file_path);

    auto background_activities = ranges::getlines(file)
                                 | utility::views::filter_not_comment_not_empty_line
                                 | ranges::views::transform(parse_line)
                                 | ranges::to_vector
                                 | ranges::actions::sort([](const auto& t1, const auto& t2) {
                                       return std::get<0>(t1) < std::get<0>(t2);
                                   })
                                 | ranges::to_vector;

    // Extract the set of input keys
    // There has to be an easier way of doing this
    std::vector<std::string> input_keys = background_activities | ranges::views::transform([](const auto& p) { return std::get<2>(p); }) | ranges::to_vector;
    std::sort(input_keys.begin(), input_keys.end());
    auto last = std::unique(input_keys.begin(), input_keys.end());
    input_keys.erase(last, input_keys.end());

    std::vector<std::shared_ptr<ActivityInput>> inputs{};
    inputs.reserve(input_keys.size());

    auto background_activities_with_indices = std::vector<std::tuple<RelearnTypes::step_type, RelearnTypes::step_type, std::size_t, std::vector<NeuronID>>>{};

    for (const auto& [begin, end, input_descr, parsed_ids] : background_activities) {
        const auto it = std::find(input_keys.begin(), input_keys.end(), input_descr);
        RelearnException::check(it != input_keys.end(), "BackgroundActivityIO::load_background_activity: Input {} does not exist", input_descr);
        const auto index = std::distance(input_keys.begin(), it);
        background_activities_with_indices.emplace_back(begin, end, index, parsed_ids);
    }

    for (const auto& key : input_keys) {
        inputs.push_back(create_input(key));
    }

    std::unique_ptr<ChoiceFunction> function = std::make_unique<CachedChoiceFunction>(std::move(background_activities_with_indices), local_group_translator->get_number_neurons_in_total());
    return std::make_pair(inputs, std::move(function));
}

std::shared_ptr<ActivityInput> BackgroundActivityIO::create_input(std::string key) {
    StringUtil::to_lower(key);
    const auto i1 = key.find(':');
    const auto prefix = key.substr(0, i1);
    const auto params = StringUtil::split_string(key.substr(i1 + 1), ',');

    if (prefix == "normal") {
        // normal background
        RelearnException::check(params.size() == 2,
                                "FlexibleBackgroundActivityCalculator::create_input: Needs 2 parameters but {} were given ({})",
                                params.size(), key);

        const auto new_calculator = std::make_shared<NormalActivityInput>(boost::lexical_cast<double>(params[0]), boost::lexical_cast<double>(params[1]));
        return new_calculator;
    }

    if (prefix == "fastnormal") {
        // normal background
        RelearnException::check(params.size() == 2,
                                "FlexibleBackgroundActivityCalculator::create_input: Needs 2 parameters but {} were given ({})",
                                params.size(), key);

        const auto new_calculator = std::make_shared<FastNormalActivityInput>(boost::lexical_cast<double>(params[0]), boost::lexical_cast<double>(params[1]), 1000);
        return new_calculator;
    }

    if (prefix == "constant") {
        // normal background
        RelearnException::check(params.size() == 1,
                                "FlexibleBackgroundActivityCalculator::create_input: Needs 1 parameters but {} were given ({})",
                                params.size(), key);

        const auto new_calculator = std::make_shared<ConstantActivityInput>(boost::lexical_cast<double>(params[0]));
        return new_calculator;
    }

    RelearnException::fail(
        "FlexibleBackgroundActivityCalculator::create_input: Unknown activity input {}", prefix);
}
