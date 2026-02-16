#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/comm_patterns/Indices.h"
#include "mpi-wrapper/comm_patterns/Types.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <cstdint>
#include <vector>

namespace mpiPP {

namespace comm_patterns {

/**
 * @brief Structure holding the answers to the questions send to other nodes
 *
 * @tparam A_parameter value type of the answers
 */
template <typename A_parameter, typename identifier_type>
class NodeToNodeAnswer {
public:
    static_assert(std::is_integral_v<identifier_type>);
    static_assert(std::is_unsigned_v<identifier_type>);

    NodeToNodeAnswer(std::vector<A_parameter> answers, Indices<identifier_type> index_rank)
        : answers_for_my_rank(std::move(answers))
        , questioner_node_to_answers_index_range(std::move(index_rank)) {

        questioner_node_to_answers_index_range.normalize();
    }

    /**
     * Method to get the answeres to states questions in the node_to_node_question method
     * Will only return information if called on an object of NodeToNodeAnswer
     *      returned by node_to_node_question
     * The answers to the list of questions created in generateQuestions by node N will
     * be returned by using local_id=N in the same order as the questions where stated.
     *
     * Hint: Other than the order ensurance and the fact that local_id was the questioner node
     *       there is no correspondence to the specific question the answer was created by.
     *       If that is not enough, transfer a question identifier in the Q_parameter and A_parameter
     *       of node_to_node_question
     */
    [[nodiscard]] std::vector<A_parameter> get_answers_of_questioner_node(const identifier_type local_id) const {
        utility::Exception::check(local_id < questioner_node_to_answers_index_range.get_number_local_values(), "T0");

        if (answers_for_my_rank.empty()) {
            return {};
        }

        const auto index_ranges_for_node = questioner_node_to_answers_index_range.get_indices(local_id);

        auto answers = std::vector<A_parameter>{};
        for (const auto& [begin, end] : index_ranges_for_node) {
            if (begin == end) {
                continue;
            }

            utility::Exception::check(begin < answers_for_my_rank.size(), "T1");
            utility::Exception::check(end <= answers_for_my_rank.size(), "T2");

            answers.insert(answers.end(), answers_for_my_rank.begin() + utility::save_cast<std::int64_t>(begin),
                           answers_for_my_rank.begin() + utility::save_cast<std::int64_t>(end));
        }

        return answers;
    }

private:
    // List of answers coresponding to nodes_to_ask_question
    // The vector is logically divided into subranges of answers from other ranks.
    // These subranges is further divided into subranges of answers for each questioner of this rank.
    std::vector<A_parameter> answers_for_my_rank;

    // stores index in nodes_to_ask_question for each rank
    Indices<identifier_type> questioner_node_to_answers_index_range;
};

} // namespace comm_patterns

} // namespace mpiPP
