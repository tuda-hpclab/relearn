#include "AxonPositionIO.h"

#include "cpp-utility/ranges/views/IO.hpp"

#include <range/v3/view/getlines.hpp>
#include <spdlog/spdlog.h>

std::vector<std::vector<RelearnTypes::position_type>> AxonPositionIO::read_axon_positions(const std::filesystem::path& file_path) {
    auto file = std::ifstream(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "AxonPositionIO::read_axon_positions: Opening the file was not successful");

    constexpr static auto guessed_number_neurons = std::size_t{ 64000 };
    constexpr static auto guessed_number_positions = std::size_t{ 10 };

    auto positions = std::vector<std::vector<position_type>>{};
    positions.reserve(guessed_number_neurons);

    auto last_id = NeuronID::value_type{ 1 };

    for (const auto& line : ranges::getlines(file)) {
        // Skip line with comments
        if (line.empty()) {
            continue;
        }

        if ('#' == line[0]) {
            continue;
        }

        auto id = NeuronID::value_type{};
        auto pos_x = position_type::value_type{};
        auto pos_y = position_type::value_type{};
        auto pos_z = position_type::value_type{};

        auto sstream = std::stringstream(line);

        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z);
        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        RelearnException::check(id >= last_id, "AxonPositionIO::read_axon_positions: Neuron ids must be increasing. Found: {} but expected: {} (or larger)", id, last_id);
        last_id = std::max(id, last_id);
        
        id--;

        if (positions.size() <= id) {
            if (positions.capacity() <= id) {
                positions.reserve((id * 2) + 1);
            }

            const auto old_size = positions.size();
            const auto new_size = id + 1;
            positions.resize(new_size);

            for (auto i = old_size; i < new_size; ++i) {
                positions[i] = std::vector<position_type>{};
                positions[i].reserve(guessed_number_positions);
            }
        }

        positions[id].emplace_back(pos_x, pos_y, pos_z);
    }

    return positions;
}
