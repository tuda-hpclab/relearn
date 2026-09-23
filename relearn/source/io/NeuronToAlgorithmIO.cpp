/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2025-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "io/NeuronToAlgorithmIO.h"

#include "algorithm/AlgorithmEnum.h"
#include "algorithm/Kernel/Gamma.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/KernelBase.h"
#include "algorithm/Kernel/KernelType.h"
#include "algorithm/Kernel/Linear.h"
#include "algorithm/Kernel/Weibull.h"
#include "io/parser/MonitorParser.h"
#include "types/AlgorithmTypes.h"
#include "types/BasicTypes.h"
#include "util/NeuronID.h"
#include "util/RelearnException.h"

#include <boost/lexical_cast.hpp>

#include <cpp-utility/StringUtil.hpp>
#include <cpp-utility/data/intersection.hpp>
#include <cpp-utility/ranges/views/IO.hpp>

#include <fmt/std.h>

#include <mpi-wrapper/core/MPIRank.h>

#include <range/v3/action/sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/transform.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

[[nodiscard]] std::pair<RelearnTypes::AlgorithmIndexWithNeuronsType, RelearnTypes::AlgorithmConfigs> NeuronToAlgorithmIO::read_descriptions(const std::filesystem::path& file_path, const mpiPP::MPIRank my_rank,
                                                                                                                                            const std::shared_ptr<LocalGroupTranslator>& local_group_translator) {
    auto file = std::ifstream{ file_path };

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good,
                            "NeuronToAlgorithmIO::readDescriptions: Opening the file '{}' was not successful", file_path);

    auto algorithms = RelearnTypes::AlgorithmConfigs{}; // list of used algorithms

    auto already_seen_neuron_ids = std::unordered_set<NeuronID>{}; // keep track of already seen NeuronIDs

    const auto& parse_line
        = [my_rank, local_group_translator, &algorithms, &already_seen_neuron_ids](const std::string& line) {
              auto sstream = std::stringstream(line);

              auto algorithm = std::string{};

              const auto success = static_cast<bool>(sstream >> algorithm);
              RelearnException::check(success, "NeuronToAlgorithmIO::readDescriptions: Did not succeed in parsing.");

              auto algorithm_config = get_algorithm_config(algorithm); // copy elision done here

              const auto index = algorithms.size(); // new algorithm will have index algorithms.size

              algorithms.emplace_back(std::move(algorithm_config)); // place algorithm in list of used algorithms. each algorithm only shows up once in the file, so it is safe that the algorithm is inserted for the first and also last time.

              auto descriptions = std::vector<std::string>{};

              for (auto current_value = std::string{}; sstream >> current_value;) {
                  descriptions.emplace_back(current_value);
              }
              const auto& parsed_ids = MonitorParser::parse_my_ids(descriptions, my_rank, local_group_translator);

              const auto id_already_seen = utility::containers_intersect(already_seen_neuron_ids, parsed_ids);

              RelearnException::check(!id_already_seen, "NeuronToAlgorithmIO::read_descriptions: A neuron is not supposed to use multiple different algorithm configs. Each neuron should only get assigned to one!");

              already_seen_neuron_ids.insert(parsed_ids.begin(), parsed_ids.end());
              return std::make_pair(index, parsed_ids);
          };

    // this is a vector of tuples of index and std::vector<NeuronID>
    const RelearnTypes::AlgorithmIndexWithNeuronsType algorithmidxes_and_ids = ranges::getlines(file)
                                                                               | utility::views::filter_not_comment_not_empty_line
                                                                               | ranges::views::transform(parse_line)
                                                                               | ranges::to_vector;

    file.close();

    auto number_neuron_ids = RelearnTypes::number_neurons_type{ 0 };
    for (const auto& [_algorithm_config, neuron_ids] : algorithmidxes_and_ids) {
        number_neuron_ids += neuron_ids.size();
    }
    RelearnException::check(number_neuron_ids == local_group_translator->get_number_neurons_in_total(), "NeuronToAlgorithmIO::read_descriptions: Some neurons are missing or duplicated. Number of neurons assigned {} != Number of neurons on this rank {}", number_neuron_ids, local_group_translator->get_number_neurons_in_total());

    return std::make_pair(algorithmidxes_and_ids, std::move(algorithms));
}

// barnes-hut:gaussian(1,5);1.5
[[nodiscard]] AlgorithmConfig NeuronToAlgorithmIO::get_algorithm_config(std::string description) {
    utility::to_lower(description);
    const auto ind1 = description.find(':');
    const auto algorithm_type = description.substr(0, ind1);
    const auto params = utility::split_string(description.substr(ind1 + 1), ';');

    const auto string_to_algorithm = std::map<std::string, AlgorithmEnum>{
        { "naive", AlgorithmEnum::Naive },
        { "barnes-hut", AlgorithmEnum::BarnesHut },
        { "barnes-hut-inverted", AlgorithmEnum::BarnesHutInverted },
        { "barnes-hut-location-aware", AlgorithmEnum::BarnesHutLocationAware },
    };

    RelearnException::check(string_to_algorithm.contains(algorithm_type), "NeuronToAlgorithmIO::get_algorithm_config: Unknown algorithm {}", algorithm_type);

    const auto algorithm_enum = string_to_algorithm.at(algorithm_type);

    const auto& kernel_description = params[0]; // "gaussian(1,5)"

    if (!is_barnes_hut(algorithm_enum)) {
        RelearnException::check(params.size() < 2, "NeuronToAlgorithmIO::get_algorithm_config: Algorithms other than Barnes-Hut types should not have additional parameters! However, {} did, ({})", stringify(algorithm_enum), description);
    }

    // read kernel
    const auto ind2 = kernel_description.find('(');

    std::string kernel;
    auto kernel_params = std::vector<std::string>{};
    if (ind2 == std::string::npos) {
        kernel = kernel_description;
    } else {
        kernel = kernel_description.substr(0, ind2); // "gaussian"
        const auto ind3 = kernel_description.find(')');
        RelearnException::check((ind3 != std::string::npos && ind3 > ind2), "NeuronToAlgorithmIO::get_algorithm_config: Closing brackets missing! ({})", description);
        const auto& kernel_params_str = kernel_description.substr(ind2 + 1, ind3 - ind2 - 1); // "1,5"
        if (!kernel_params_str.empty()) {
            kernel_params = utility::split_string(kernel_params_str, ','); // ["1", "5"]
        }
    }

    const auto string_to_kernel = std::map<std::string, KernelType>{
        { "gaussian", KernelType::Gaussian },
        { "linear", KernelType::Linear },
        { "gamma", KernelType::Gamma },
        { "weibull", KernelType::Weibull }
    };

    const auto kernel_type = string_to_kernel.at(kernel);

    std::unique_ptr<KernelBase> kernel_to_use;
    switch (kernel_type) {
    case KernelType::Gaussian:
        if (kernel_params.empty()) {
            kernel_to_use = std::make_unique<GaussianDistributionKernel>();
        } else if (kernel_params.size() == 2) {
            kernel_to_use = std::make_unique<GaussianDistributionKernel>(boost::lexical_cast<RelearnTypes::attraction_type>(kernel_params[0]), boost::lexical_cast<RelearnTypes::attraction_type>(kernel_params[1]));
        } else {
            RelearnException::fail("NeuronToAlgorithmIO::get_algorithm_config: Expected 0 or 2 parameters for gaussian kernel, but got {}, ({})", kernel_params.size(), description);
        }
        break;
    case KernelType::Linear:
        if (kernel_params.empty()) {
            kernel_to_use = std::make_unique<LinearDistributionKernel>();
        } else if (kernel_params.size() == 1) {
            kernel_to_use = std::make_unique<LinearDistributionKernel>(boost::lexical_cast<RelearnTypes::attraction_type>(kernel_params[0]));
        } else {
            RelearnException::fail("NeuronToAlgorithmIO::get_algorithm_config: Expected 0 or 1 parameters for linear kernel, but got {}, ({})", kernel_params.size(), description);
        }
        break;
    case KernelType::Gamma:
        if (kernel_params.empty()) {
            kernel_to_use = std::make_unique<GammaDistributionKernel>();
        } else if (kernel_params.size() == 2) {
            kernel_to_use = std::make_unique<GammaDistributionKernel>(boost::lexical_cast<RelearnTypes::attraction_type>(kernel_params[0]), boost::lexical_cast<RelearnTypes::attraction_type>(kernel_params[1]));
        } else {
            RelearnException::fail("NeuronToAlgorithmIO::get_algorithm_config: Expected 0 or 2 parameters for gamma kernel, but got {}, ({})", kernel_params.size(), description);
        }
        break;
    case KernelType::Weibull:
        if (kernel_params.empty()) {
            kernel_to_use = std::make_unique<WeibullDistributionKernel>();
        } else if (kernel_params.size() == 2) {
            kernel_to_use = std::make_unique<WeibullDistributionKernel>(boost::lexical_cast<RelearnTypes::attraction_type>(kernel_params[0]), boost::lexical_cast<RelearnTypes::attraction_type>(kernel_params[1]));
        } else {
            RelearnException::fail("NeuronToAlgorithmIO::get_algorithm_config: Expected 0 or 2 parameters for weibull kernel, but got {}, ({})", kernel_params.size(), description);
        }
        break;
    default:
        RelearnException::fail("NeuronToAlgorithmIO::get_algorithm_config: Unknown kernel type! ({})", kernel_type);
    }

    // read theta
    // if not barnes-hut or theta is just not given, we don't need additional parameters
    if (!is_barnes_hut(algorithm_enum) || params.size() < 2) {
        return { algorithm_enum, std::move(kernel_to_use) };
    }

    const auto& theta = params[1];

    return { algorithm_enum, std::move(kernel_to_use), boost::lexical_cast<RelearnTypes::acceptance_criterion_type>(theta) };
}
