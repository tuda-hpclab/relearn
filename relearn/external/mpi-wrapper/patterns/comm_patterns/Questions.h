#pragma once

/*
 * This file is part of the MPI-Wrapper software developed at Technical University Darmstadt.
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "mpi-wrapper/collectives/MPIAllToAll.h"
#include "mpi-wrapper/collectives/MPIAllToAllV.h"
#include "mpi-wrapper/core/MPIInfo.h"
#include "mpi-wrapper/core/MPIRank.h"
#include "mpi-wrapper/core/MPIRankRange.h"
#include "mpi-wrapper/core/MPITypes.h"
#include "mpi-wrapper/patterns/comm_patterns/NodeToNodeAnswer.h"
#include "mpi-wrapper/patterns/comm_patterns/NodeToNodeQuestion.h"
#include "mpi-wrapper/patterns/comm_patterns/Types.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>
#include <cpp-utility/data/displacement.hpp>

#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace mpiPP {

namespace comm_patterns {

namespace detail {

/**
 * @brief Get the questions for each local node from generate_questions
 *
 * @tparam QParameter type of the question message
 * @tparam AParameter type of the answer message
 * @param number_local_values The number of addressable local node ids
 * @param local_nodes_asking_questions The local node ids for which questions are generated
 * @param generate_questions Function invoked once for each id in local_nodes_asking_questions
 * @return NodeToNodeQuestion<QParameter, AParameter> collected questions from generate_questions
 */
template <MPICompatible QParameter, typename AParameter, typename NodesRange, typename IdentifierType>
[[nodiscard]] NodeToNodeQuestion<QParameter, AParameter, IdentifierType>
get_questions(const IdentifierType number_local_values,
              const NodesRange& local_nodes_asking_questions,
              const std::convertible_to<GenerateQuestionsFunction<QParameter, IdentifierType>> auto& generate_questions) {

    auto questions = NodeToNodeQuestion<QParameter, AParameter, IdentifierType>(MPIInfo::get_number_ranks_cast(), number_local_values);
    questions.add_questions_to_send(local_nodes_asking_questions, generate_questions);
    return questions;
}

/**
 * @brief Distribute questions to each rank and return the questions from other node to local node
 *
 * @tparam QParameter type of the question message
 * @tparam AParameter type of the answer message
 * @param questioner_structure questions
 * @param number_questions_to_receive_per_rank range of number of questions that globally is send to each rank
 * @return NodeToNodeQuestion<QParameter, AParameter> questions addressed to the local node
 */
template <MPICompatible QParameter, typename AParameter, typename IdentifierType>
[[nodiscard]] NodeToNodeQuestion<QParameter, AParameter, IdentifierType>
distribute_questions(const NodeToNodeQuestion<QParameter, AParameter, IdentifierType>& questioner_structure,
                     const std::span<const int> number_questions_to_receive_per_rank,
                     const IdentifierType number_local_nodes) {

    const auto num_questions_to_be_received_from_rank_displ = utility::calculate_displacements<int>(number_questions_to_receive_per_rank);

    // Send side: how many questions the local rank sends to each rank and where each rank's block starts
    const auto number_questions_to_ask_per_rank = questioner_structure.get_number_questions_to_ask_per_rank();
    const auto num_questions_to_ask_per_rank_displ = utility::calculate_displacements<int>(number_questions_to_ask_per_rank);

    const auto num_questions_to_send = std::reduce(number_questions_to_ask_per_rank.begin(), number_questions_to_ask_per_rank.end(), 0);
    const auto num_questions_to_send_cast = utility::safe_cast<std::size_t>(num_questions_to_send);

    // Flatten the per-rank question data into contiguous send buffers, in rank order
    auto nodes_to_ask_question_send = std::vector<IdentifierType>{};
    nodes_to_ask_question_send.reserve(num_questions_to_send_cast);
    auto question_parameters_send = std::vector<QParameter>{};
    question_parameters_send.reserve(num_questions_to_send_cast);

    const auto number_ranks = MPIInfo::get_number_ranks();
    for (const auto rank : MPIRankRange::range(number_ranks)) {
        const auto nodes_to_ask_question_for_rank = questioner_structure.get_nodes_to_ask_question_for_rank(rank);
        const auto question_parameters_for_rank = questioner_structure.get_question_parameters_for_rank(rank);

        utility::Exception::check(nodes_to_ask_question_for_rank.size() == question_parameters_for_rank.size(),
                                  "comm_patterns::detail::distribute_questions: The number of addressee nodes {} does not match the number of question parameters {} for rank {}",
                                  nodes_to_ask_question_for_rank.size(), question_parameters_for_rank.size(), rank.get_rank());

        nodes_to_ask_question_send.insert(nodes_to_ask_question_send.end(), nodes_to_ask_question_for_rank.begin(), nodes_to_ask_question_for_rank.end());
        question_parameters_send.insert(question_parameters_send.end(), question_parameters_for_rank.begin(), question_parameters_for_rank.end());
    }

    // Exchange nodes and parameters with all ranks in two collectives. The vector-returning overload
    // sizes the receive buffers automatically; the received blocks land in source-rank order at
    // num_questions_to_be_received_from_rank_displ, exactly as set_questions_received expects.
    auto my_rank_total_nodes_to_ask_question = MPICollectives::all_to_all_v<IdentifierType>(
        nodes_to_ask_question_send.data(), number_questions_to_ask_per_rank.data(), num_questions_to_ask_per_rank_displ.data(),
        number_questions_to_receive_per_rank.data(), num_questions_to_be_received_from_rank_displ.data());

    auto my_rank_total_question_parameters = MPICollectives::all_to_all_v<QParameter>(
        question_parameters_send.data(), number_questions_to_ask_per_rank.data(), num_questions_to_ask_per_rank_displ.data(),
        number_questions_to_receive_per_rank.data(), num_questions_to_be_received_from_rank_displ.data());

    // Set questions to be answered by local node
    const auto number_ranks_cast = MPIInfo::get_number_ranks_cast();
    auto questions_to_answer = NodeToNodeQuestion<QParameter, AParameter, IdentifierType>(number_ranks_cast, number_local_nodes);
    questions_to_answer.set_questions_received(my_rank_total_nodes_to_ask_question, my_rank_total_question_parameters, number_questions_to_receive_per_rank,
                                               num_questions_to_be_received_from_rank_displ);
    return questions_to_answer;
}

/**
 * @brief Exchange answers and set the answers received for the questions of the local node
 *
 * @tparam QParameter type of the question message
 * @tparam AParameter type of the answer message
 * @param questions_asked The questions that were asked by the local node
 * @param local_answers The answers generated by the local node for the received questions
 */
template <MPICompatible QParameter, MPICompatible AParameter, typename IdentifierType>
[[nodiscard]] std::vector<AParameter>
exchange_answers(const NodeToNodeQuestion<QParameter, AParameter, IdentifierType>& questions_asked,
                 const std::vector<std::vector<AParameter>>& local_answers) {
    const auto number_ranks = MPIInfo::get_number_ranks_cast();
    utility::Exception::check(local_answers.size() == number_ranks,
                              "comm_patterns::detail::exchange_answers: The number of answer lists {} does not match the number of ranks {}",
                              local_answers.size(), number_ranks);

    // Send side: one contiguous buffer holding the answers for each rank, in rank order
    auto send_counts = std::vector<int>{};
    send_counts.reserve(local_answers.size());
    auto total_send = std::size_t{ 0 };
    for (const auto& answers_for_rank : local_answers) {
        send_counts.emplace_back(utility::safe_cast<int>(answers_for_rank.size()));
        total_send += answers_for_rank.size();
    }
    const auto send_displacements = utility::calculate_displacements<int>(send_counts);

    auto flattened_answers = std::vector<AParameter>{};
    flattened_answers.reserve(total_send);
    for (const auto& answers_for_rank : local_answers) {
        flattened_answers.insert(flattened_answers.end(), answers_for_rank.begin(), answers_for_rank.end());
    }

    // Recv side: one block per rank the local node asked, in the layout NodeToNodeAnswer's indices expect
    const auto number_answers_per_rank = questions_asked.get_number_questions_to_ask_per_rank();
    const auto number_answers_per_rank_displ = utility::calculate_displacements(number_answers_per_rank);

    return MPICollectives::all_to_all_v<AParameter>(flattened_answers.data(), send_counts.data(), send_displacements.data(),
                                                    number_answers_per_rank.data(), number_answers_per_rank_displ.data());
}

} // namespace detail

/**
 * @brief A function that allows the communication from one node to another
 *  regardless of the rank the node is on.
 *
 * Template parameters that can not be represented by a MPI Type are not allowed and result in undefined
 * behaviour
 *
 * Time complexity:
 *                      = O(number_nodes_per_rank * Time(generate_questions)) +
 *                        O(number_of_generated_questions * Time(generate_answers)) +
 *                        O(number_of_generated_questions)
 *
 * MPI message complexity:
 *                      Number of MPI data transfers :O(1) collectives (each involving all ranks)
 *                      Size of MPI data transfers :O(number_of_messages * (MEMORY(QParameter)+MEMORY(AParameter)))
 *
 * Possible causes of failure:
 *                      The function is collective and must be called by every MPI rank, including ranks
 *                          whose local_nodes_asking_questions range is empty.
 *                      The function is supposed to receive valid questions by the generate_questions method.
 *                          Targeting ranks and nodes that do not exist will result in undefined behaviour.
 *
 * @tparam QParameter type of the question message
 * @tparam AParameter type of the answer message
 * @param local_nodes_asking_questions The nodes that locally ask questions
 * @param generate_questions Function to compute the "questions" for each node
 *                      A "question" is an entry in the return vector of this function defined by
 *                          std::tuple<rank_target, node_id_target, QParameter>. The first and second entry in the tuple
 *                          determine the rank and local id of the node the question is sent to.
 *                          The third entry corresponds to a question-message that each question can have.
 *                          The type of the question-message is QParameter
 *                      Function is called once for each id in local_nodes_asking_questions.
 *                          The local id of the questioner node is "local_id"
 *                      The function can generate an arbitrary number of questions (including zero) per call
 * @param generate_answers Function to compute the "answers" for each question
 *                      Function is guaranteed to be called for every "question" that was generated by generate_questions
 *                          at the corresponding rank it was sent to. The local id of the answering node is "local_id"
 *                          The answering node is the one specified with the question in the second tuple entry "node_id_target"
 *                      "Hint: The questioner is not visible for the answerer if this information is not sent in the message"
 *                      An "answer" is an instance of AParameter. Each "question" will generate a generate_answers call and
 *                          therefore an "answer". AParameter is the return type of generate_answers
 *                      The "answer" will be transmitted to the questioner rank and node
 * @return NodeToNodeAnswer<AParameter> contains the answers to the stated question, that can be extracted using
 *                          get_answers_of_questioner_node(local_id) of NodeToNodeAnswer
 *                          Look at the NodeToNodeAnswer description for details
 */
template <MPICompatible QParameter, MPICompatible AParameter, typename NodesRange, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
[[nodiscard]] NodeToNodeAnswer<AParameter, IdentifierType>
node_to_node_question(const IdentifierType number_local_values,
                      const NodesRange& local_nodes_asking_questions,
                      const std::convertible_to<GenerateQuestionsFunction<QParameter, IdentifierType>> auto& generate_questions,
                      const std::convertible_to<GenerateAnswersFunction<AParameter, QParameter, IdentifierType>> auto& generate_answers) {

    const auto questions_asked = detail::get_questions<QParameter, AParameter, NodesRange, IdentifierType>(number_local_values, local_nodes_asking_questions, generate_questions);

    const auto number_questions_to_receive_per_rank = MPICollectives::all_to_all(questions_asked.get_number_questions_to_ask_per_rank());

    const auto questions_to_answer = detail::distribute_questions(questions_asked, number_questions_to_receive_per_rank, number_local_values);

    const auto local_answers = questions_to_answer.compute_answers(generate_answers);

    auto exchanged_answers = detail::exchange_answers(questions_asked, local_answers);
    auto indices = std::move(questions_asked).get_indices();

    return NodeToNodeAnswer<AParameter, IdentifierType>(std::move(exchanged_answers), std::move(indices));
}

/**
 * @brief Same as the first node_to_node_question overload, but every local id in
 *  [0, number_local_values) asks questions (the iota range over all local ids).
 *
 * See the first node_to_node_question overload for the full semantics of generate_questions/generate_answers,
 * the complexities, and the failure modes.
 *
 * @tparam QParameter type of the question message
 * @tparam AParameter type of the answer message
 * @return NodeToNodeAnswer<AParameter> the answers to the stated questions
 */
template <MPICompatible QParameter, MPICompatible AParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
[[nodiscard]] NodeToNodeAnswer<AParameter, IdentifierType>
node_to_node_question(const IdentifierType number_local_values,
                      const std::convertible_to<GenerateQuestionsFunction<QParameter, IdentifierType>> auto& generate_questions,
                      const std::convertible_to<GenerateAnswersFunction<AParameter, QParameter, IdentifierType>> auto& generate_answers) {

    const auto ids = std::views::iota(IdentifierType(0), number_local_values);
    const auto answer = node_to_node_question<QParameter, AParameter, decltype(ids), IdentifierType>(number_local_values, ids, generate_questions, generate_answers);
    return answer;
}

/**
 * @brief Same as the first node_to_node_question overload, but with void answers: generate_answers returns
 *  nothing, no answers are transferred back, and the function returns nothing.
 *
 * See the first node_to_node_question overload for the full semantics of generate_questions/generate_answers,
 * the complexities, and the failure modes.
 *
 * @tparam QParameter type of the question message
 * @param local_nodes_asking_questions The nodes that locally ask questions
 * @param generate_answers Function invoked for each received question; its return value is ignored
 */
template <MPICompatible QParameter, typename NodesRange, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
void node_to_node_question(const IdentifierType number_local_values,
                           const NodesRange& local_nodes_asking_questions,
                           const std::convertible_to<GenerateQuestionsFunction<QParameter, IdentifierType>> auto& generate_questions,
                           const std::convertible_to<VoidAnswersFunction<QParameter, IdentifierType>> auto& generate_answers) {

    const auto questions_asked = detail::get_questions<QParameter, void, NodesRange, IdentifierType>(number_local_values, local_nodes_asking_questions, generate_questions);

    const auto number_questions_to_receive_per_rank = MPICollectives::all_to_all(questions_asked.get_number_questions_to_ask_per_rank());

    const auto questions_to_answer = detail::distribute_questions(questions_asked, number_questions_to_receive_per_rank, number_local_values);

    questions_to_answer.compute_answers(generate_answers);
}

/**
 * @brief Same as the first node_to_node_question overload, but every local id in
 *  [0, number_local_values) asks questions (the iota range over all local ids) and answers are void:
 *  generate_answers returns nothing, no answers are transferred back, and the function returns nothing.
 *
 * See the first node_to_node_question overload for the full semantics of generate_questions/generate_answers,
 * the complexities, and the failure modes.
 *
 * @tparam QParameter type of the question message
 * @param generate_answers Function invoked for each received question; its return value is ignored
 */
template <MPICompatible QParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
void node_to_node_question(const IdentifierType number_local_values,
                           const std::convertible_to<GenerateQuestionsFunction<QParameter, IdentifierType>> auto& generate_questions,
                           const std::convertible_to<VoidAnswersFunction<QParameter, IdentifierType>> auto& generate_answers) {

    const auto ids = std::views::iota(IdentifierType(0), number_local_values);
    node_to_node_question<QParameter, decltype(ids), IdentifierType>(number_local_values, ids, generate_questions, generate_answers);
}
} // namespace comm_patterns

} // namespace mpiPP
