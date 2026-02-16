#include "neurons_factory.h"

#include "Config.h"
#include "Types.h"

#include "io/NeuronIO.h"
#include "neurons/LocalGroupTranslator.h"
#include "neurons/enums/SynapticElementType.h"
#include "sim/random/SubdomainFromNeuronDensity.h"
#include "structure/Partition.h"
#include "util/NeuronFilePaths.h"
#include "util/NeuronID.h"
#include "util/RelearnAllocator.h"
#include "util/Vec3.h"
#include "util/shuffle/shuffle.h"

#include "cpp-utility/ranges/Functional.hpp"

#include "mpi-wrapper/MPIRank.h"

#include "factory/neuron_id/neuron_id_factory.h"
#include "factory/random/random_factory.h"
#include "factory/simulation/simulation_factory.h"

#include <range/v3/algorithm/contains.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/generate.hpp>
#include <range/v3/view/generate_n.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

void NeuronsFactory::generate_random_neurons(std::vector<RelearnTypes::position_type>& positions,
                                             std::vector<RelearnTypes::group_ids>& neuron_id_to_group_ids,
                                             RelearnTypes::group_names& group_id_to_group_name,
                                             std::vector<SignalType>& types,
                                             std::mt19937& mt,
                                             const NeuronOptFilePaths& paths) {

    const auto& [path_to_positions, path_to_groups] = paths;

    generate_random_neuron_positions_and_signals(positions, types, mt, path_to_positions);

    const auto number_neurons = types.size();
    generate_random_neuron_groups(neuron_id_to_group_ids, group_id_to_group_name, number_neurons, mt, path_to_groups);
}

void NeuronsFactory::generate_random_neuron_positions_and_signals(std::vector<RelearnTypes::position_type>& positions,
                                                                  std::vector<SignalType>& types,
                                                                  std::mt19937& mt, const std::optional<std::filesystem::path>& path) {
    const auto number_neurons = NeuronIdFactory::get_random_number_neurons(mt);
    const auto fraction_excitatory_neurons = RandomFactory::get_random_percentage<double>(mt);
    const auto um_per_neuron = RandomFactory::get_random_double(1.0, 100.0, mt);

    const auto part = std::make_shared<Partition>(1, mpiPP::MPIRank(0));
    part->set_total_number_neurons(number_neurons);
    auto sfnd = SubdomainFromNeuronDensity{ number_neurons, fraction_excitatory_neurons, um_per_neuron, part };

    sfnd.initialize();

    positions = sfnd.get_neuron_positions_in_subdomains();
    types = sfnd.get_neuron_types_in_subdomains();

    if (path.has_value()) {
        sfnd.write_neuron_positions_and_signals_to_file(path.value());
    }
}

void NeuronsFactory::generate_random_neuron_groups(std::vector<RelearnTypes::group_ids>& neuron_id_to_group_ids,
                                                   RelearnTypes::group_names& group_id_to_group_name,
                                                   const RelearnTypes::number_neurons_type number_neurons,
                                                   std::mt19937& mt, const std::optional<std::filesystem::path>& path,
                                                   const std::optional<std::size_t>& _max_groups_except_default,
                                                   const std::size_t _max_num_groups_per_neuron_except_default) {
    const auto real_max_groups_except_default = _max_groups_except_default.value_or(number_neurons);
    group_id_to_group_name = get_random_group_names(real_max_groups_except_default, mt);
    neuron_id_to_group_ids = get_random_group_ids({ group_id_to_group_name.size(), number_neurons }, mt, _max_num_groups_per_neuron_except_default);

    if (path.has_value()) {
        NeuronIO::write_neuron_groups(path.value(), std::make_shared<LocalGroupTranslator>(group_id_to_group_name, neuron_id_to_group_ids));
    }
}

std::vector<RelearnTypes::group_ids> NeuronsFactory::get_random_group_ids(const GroupConfig& group_config, std::mt19937& mt, const std::size_t _min_num_groups_per_neuron_except_default,  const std::size_t _max_num_groups_per_neuron_except_default) {
    return ranges::views::generate_n(
               [&group_config, _max_num_groups_per_neuron_except_default, _min_num_groups_per_neuron_except_default, &mt]() {
                   const auto& number_groups = group_config.number_groups;
                   const auto num_groups_for_neuron = RandomFactory::get_random_integer<std::size_t>(_min_num_groups_per_neuron_except_default, _max_num_groups_per_neuron_except_default, mt);
                   auto group_ids = RelearnTypes::group_ids{ Constants::default_group_id }; // default group is 0 and every neuron belongs to default group

                   if (num_groups_for_neuron >= number_groups) {
                       group_ids.reserve(number_groups);
                       for (auto i = 1UL; i < number_groups; ++i) {
                           group_ids.push_back(i);
                       }
                       return group_ids;
                   }
                   group_ids.reserve(num_groups_for_neuron + 1);
                   for (auto i = 0UL; i < num_groups_for_neuron; ++i) {
                       auto candidate_group_id = RandomFactory::get_random_integer<RelearnTypes::group_id>(1, number_groups - 1, mt);
                       while (ranges::contains(group_ids, candidate_group_id)) {
                           candidate_group_id = RandomFactory::get_random_integer<RelearnTypes::group_id>(1, number_groups - 1, mt);
                       }
                       group_ids.push_back(candidate_group_id);
                   }
                   return group_ids;
               },
               group_config.number_neurons)
           | ranges::to_vector;
}

RelearnTypes::group_names NeuronsFactory::get_random_group_names(const std::size_t max_groups_except_default, std::mt19937& mt) {
    const auto number_groups = RandomFactory::get_random_integer<std::size_t>(1, max_groups_except_default, mt);
    return get_random_group_names_specific(number_groups, mt);
}

RelearnTypes::group_name NeuronsFactory::get_random_group_name(std::mt19937& mt) {
    const auto group_name_length = 10;
    auto candidate = RandomFactory::get_random_string(group_name_length, mt);
    while (!std::ranges::any_of(candidate,
                                [&](const unsigned char character) { return std::isalpha(character); })) {
        candidate = RandomFactory::get_random_string(group_name_length, mt);
    }
    return candidate;
}

RelearnTypes::group_names NeuronsFactory::get_random_group_names_specific(const std::size_t number_groups_except_default, std::mt19937& mt) {
    auto group_names = RelearnTypes::group_names{ std::string{ Constants::default_group_name } };
    group_names.reserve(number_groups_except_default + 1);

    for (auto group_id = 0ULL; group_id < number_groups_except_default; group_id++) {
        RelearnTypes::group_name name = get_random_group_name(mt);

        while (name.empty() || ranges::contains(group_names, name)) {
            name = get_random_group_name(mt);
        }

        group_names.emplace_back(std::move(name));
    }

    return group_names;
}

std::vector<RelearnTypes::group_names> NeuronsFactory::get_neuron_id_vs_group_names(const std::vector<RelearnTypes::group_ids>& neuron_id_vs_group_ids,
                                                                                    const RelearnTypes::group_names& group_id_vs_group_name) {
    return neuron_id_vs_group_ids
           | ranges::views::transform([&](const RelearnTypes::group_ids& group_ids) {
                 return group_ids
                        | ranges::views::transform(utility::lookup(group_id_vs_group_name))
                        | ranges::to_vector;
             })
           | ranges::to_vector;
}

std::vector<RelearnTypes::group_names_unordered> NeuronsFactory::get_neuron_id_vs_group_names_unordered(const std::vector<RelearnTypes::group_ids_unordered>& neuron_id_to_group_ids_unordered,
                                                                                                        const RelearnTypes::group_names& group_id_to_group_name) {
    return neuron_id_to_group_ids_unordered
           | ranges::views::transform([&](const RelearnTypes::group_ids_unordered& group_ids) {
                 return group_ids
                        | ranges::views::transform(utility::lookup(group_id_to_group_name))
                        | ranges::to<std::unordered_set>;
             })
           | ranges::to_vector;
}

std::vector<RelearnTypes::group_ids_unordered> NeuronsFactory::get_neuron_id_to_group_ids_unordered(const std::vector<RelearnTypes::group_ids>& neuron_id_to_group_ids) {
    return neuron_id_to_group_ids
           | ranges::views::transform([&](const RelearnTypes::group_ids& group_ids) {
                 return group_ids
                        | ranges::to<std::unordered_set>;
             })
           | ranges::to_vector;
}

std::string NeuronsFactory::get_invalid_group_name(const RelearnTypes::group_names& group_id_to_group_name, std::mt19937& mt) {
    RelearnTypes::group_name group_name = get_random_group_name(mt);
    while (ranges::contains(group_id_to_group_name, group_name)) {
        group_name = get_random_group_name(mt);
    }
    return group_name;
}

std::vector<std::pair<Vec3d, NeuronID>> NeuronsFactory::generate_random_neurons(const Vec3d& min, const Vec3d& max, const std::size_t max_id, std::mt19937& mt) {
    auto ids = NeuronID::range(max_id) | ranges::to_vector | actions::shuffle(mt);

    return ranges::views::zip(
               ranges::views::generate([&min, &max, &mt]() { return SimulationFactory::get_random_position_in_box(min, max, mt); }),
               ids)
           | ranges::to_vector;
}
