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

#include "mpi-wrapper/MPICollectives.h"
#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"
#include "mpi-wrapper/MPITypes.h"
#include "mpi-wrapper/collectives/MPIAllGatherV.h"
#include "mpi-wrapper/collectives/MPIAllToAll.h"
#include "mpi-wrapper/comm_patterns/NodeToNodeAnswer.h"
#include "mpi-wrapper/comm_patterns/NodeToNodeQuestion.h"
#include "mpi-wrapper/comm_patterns/Types.h"

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"
#include "cpp-utility/data/displacement.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

namespace mpiPP {

namespace comm_patterns {

namespace detail {

/**
 * @brief Get the questions for each local node from generateQuestions
 *
 * @tparam Q_parameter type of the question message
 * @tparam A_parameter type of the answer message
 * @param local_nodes_asking_questions The nodes that locally ask questions
 * @param generateQuestions function to compute the questions for each node
 * @return NodeToNodeQuestion<Q_parameter, A_parameter> collected questions from generateQuestions
 */
template <MPICompatible Q_parameter, typename A_parameter, typename NodesRange, typename identifier_type>
[[nodiscard]] static NodeToNodeQuestion<Q_parameter, A_parameter, identifier_type>
get_questions(const identifier_type number_local_values,
              const NodesRange& local_nodes_asking_questions,
              const std::convertible_to<GenerateQuestionsFunction<Q_parameter, identifier_type>> auto& generateQuestions) {

    auto questions = NodeToNodeQuestion<Q_parameter, A_parameter, identifier_type>(MPIInfo::get_number_ranks_cast(), number_local_values);
    questions.add_questions_to_send(local_nodes_asking_questions, generateQuestions);
    return questions;
}

/**
 * @brief Distribute questions to each rank and return the questions from other node to local node
 *
 * @tparam Q_parameter type of the question message
 * @tparam A_parameter type of the answer message
 * @param questioner_structure questions
 * @param number_questions_to_receive_per_rank range of number of questions that globally is send to each rank
 * @return NodeToNodeQuestion<Q_parameter, A_parameter> questions addressed to the local node
 */
template <MPICompatible Q_parameter, typename A_parameter, typename identifier_type>
[[nodiscard]] static NodeToNodeQuestion<Q_parameter, A_parameter, identifier_type>
distribute_questions(const NodeToNodeQuestion<Q_parameter, A_parameter, identifier_type>& questioner_structure,
                     const std::span<const int> number_questions_to_receive_per_rank,
                     const identifier_type number_local_nodes) {

    const auto num_questions_to_be_received_from_rank_displ = utility::calculate_displacements<int>(number_questions_to_receive_per_rank);

    const auto num_questions_to_be_received = std::reduce(number_questions_to_receive_per_rank.begin(), number_questions_to_receive_per_rank.end(), 0);
    const auto num_questions_to_be_received_cast = utility::save_cast<std::size_t>(num_questions_to_be_received);

    auto my_rank_total_nodes_to_ask_question = std::vector<identifier_type>(num_questions_to_be_received_cast);
    auto my_rank_total_question_parameters = std::vector<Q_parameter>(num_questions_to_be_received_cast);

    if (num_questions_to_be_received_cast == 0) {
        // This is here to avoid nullptr later on
        my_rank_total_nodes_to_ask_question.reserve(1);
        my_rank_total_question_parameters.reserve(1);
    }

    // received_question_parameters

    const auto number_ranks = MPIInfo::get_number_ranks();
    for (const auto rank : MPIRank::range(number_ranks)) {
        const auto nodes_to_ask_question_for_rank = questioner_structure.get_nodes_to_ask_question_for_rank(rank);
        const auto question_parameters_for_rank = questioner_structure.get_question_parameters_for_rank(rank);

        utility::Exception::check(nodes_to_ask_question_for_rank.size() == question_parameters_for_rank.size(), "");

        MPICollectives::gatherv(nodes_to_ask_question_for_rank, my_rank_total_nodes_to_ask_question, number_questions_to_receive_per_rank,
                                num_questions_to_be_received_from_rank_displ, rank);

        MPICollectives::gatherv(question_parameters_for_rank, my_rank_total_question_parameters, number_questions_to_receive_per_rank,
                                num_questions_to_be_received_from_rank_displ, rank);
    }

    // Set questions to be answered by local node
    const auto number_ranks_cast = MPIInfo::get_number_ranks_cast();
    auto questions_to_answer = NodeToNodeQuestion<Q_parameter, A_parameter, identifier_type>(number_ranks_cast, number_local_nodes);
    questions_to_answer.set_questions_received(my_rank_total_nodes_to_ask_question, my_rank_total_question_parameters, number_questions_to_receive_per_rank,
                                               num_questions_to_be_received_from_rank_displ);
    return questions_to_answer;
}

/**
 * @brief Exchange answers and set the answers received for the questions of the local node
 *
 * @tparam Q_parameter type of the question message
 * @tparam A_parameter type of the answer message
 * @param questions_asked
 * @param local_answers
 */
template <MPICompatible Q_parameter, MPICompatible A_parameter, typename identifier_type>
[[nodiscard]] static std::vector<A_parameter>
exchange_answers(const NodeToNodeQuestion<Q_parameter, A_parameter, identifier_type>& questions_asked,
                 const std::vector<std::vector<A_parameter>>& local_answers) {
    const auto number_answers_per_rank = questions_asked.get_number_questions_to_ask_per_rank();
    const auto number_answers_per_rank_displ = utility::calculate_displacements(number_answers_per_rank);

    const auto my_rank_total_receive_size = std::accumulate(number_answers_per_rank.begin(), number_answers_per_rank.end(), 0);

    auto all_received_answers = std::vector<A_parameter>(utility::save_cast<std::size_t>(my_rank_total_receive_size));
    if (my_rank_total_receive_size == 0) {
        // This is to prevent nullptrs later on
        all_received_answers.reserve(1);
    }

    const auto number_ranks = MPIInfo::get_number_ranks();
    for (const auto rank : MPIRank::range(number_ranks)) {
        utility::Exception::check(rank.get_rank_cast() < local_answers.size(), "qo4h");
        const auto& answers_for_rank = local_answers[rank.get_rank_cast()];
        MPICollectives::gatherv(answers_for_rank, all_received_answers, number_answers_per_rank, number_answers_per_rank_displ, rank);
    }

    return all_received_answers;
}

} // namespace detail

/**
 * @brief A function that allows the communication from one node to another
 *  regardeless of the rank the node is on.
 *
 * Template parameters that can not be represented by a MPI Type are not allowed and result in undefined
 * behaviour
 *
 * Time complexity:
 *                      = O(number_nodes_per_rank * Time(generateQuestions)) +
 *                        O(number_of_generated_questions * Time(generateAnswers)) +
 *                        O(number_of_generated_questions)
 *
 * MPI messsage complexity:
 *                      Number of MPI data transfers :O(number_of_ranks)
 *                      Size of MPI data transfers :O(number_of_messages * (MEMORY(Q_parameter)+MEMORY(A_parameter))
 *
 * Possible causes of failure:
 *                      The function is supposed to be called by every rank that carries nodes of the graph.
 *                          Missing this requirement will result in undefined behaviour.
 *                      The function is supposed the receive valid questions by the generateQuestions method.
 *                          Targeting ranks and nodes that do not exist will result in undefined behaviour.
 *
 * @tparam Q_parameter type of the question message
 * @tparam A_parameter type of the answer message
 * @param local_nodes_asking_questions The nodes that locally ask questions
 * @param generateQuestions Function to compute the "questions" for each node
 *                      A "question" is an entry in the return vector of this function defined by
 *                          std::tuple<rank_target, node_id_target, Q_parameter>. The first and second entry in the tuple
 *                          determine the rank and local id of the node the question is send to.
 *                          The third entry corresponds to a question-message that each question can have.
 *                          The type of the question-message is Q_parameter
 *                      Function is guaranteed to be called for every node in the graph.
 *                          The local id of the questioneer node is  "local_id"
 *                      The function can generate an arbitrary number of questions (including zero) per call
 * @param generateAnswers Function to compute the "answers" for each question
 *                      Function is guaranteed to be called for every "question" that was generated by generateQuestions
 *                          at the corresponding rank it was send to. The local id of the answering node is "local_id"
 *                          The answering node is the one specified with the question in the second tuple entry "node_id_target"
 *                      "Hint: The questioneer is not visible for the answerer if this information is not send in the message"
 *                      An "answer" is an instance of A_parameter. Each "question" will generate a generateAnswers call and
 *                          therefore an "answer". A_parameter is the return type of generateAnswers
 *                      The "answer" will be transmitted to the questioneer rank and node
 * @return NodeToNodeAnswer<A_parameter> contains the answers to the stated question, thats can be extracted using
 *                          get_answers_of_questioner_node(local_id) of NodeToNodeAnswer
 *                          Look at the NodeToNodeAnswer description for details
 */
template <MPICompatible Q_parameter, MPICompatible A_parameter, typename NodesRange, typename identifier_type>
    requires std::unsigned_integral<identifier_type>
[[nodiscard]] static NodeToNodeAnswer<A_parameter, identifier_type>
node_to_node_question(const identifier_type number_local_values,
                      const NodesRange& local_nodes_asking_questions,
                      const std::convertible_to<GenerateQuestionsFunction<Q_parameter, identifier_type>> auto& generateQuestions,
                      const std::convertible_to<GenerateAnswersFunction<A_parameter, Q_parameter, identifier_type>> auto& generateAnswers) {

    const auto questions_asked = detail::get_questions<Q_parameter, A_parameter, NodesRange, identifier_type>(number_local_values, local_nodes_asking_questions, generateQuestions);

    const auto number_questions_to_receive_per_rank = MPICollectives::all_to_all(questions_asked.get_number_questions_to_ask_per_rank());

    const auto questions_to_answer = detail::distribute_questions(questions_asked, number_questions_to_receive_per_rank, number_local_values);

    const auto local_answers = questions_to_answer.compute_answers(generateAnswers);

    auto exchanged_answers = detail::exchange_answers(questions_asked, local_answers);
    auto indices = std::move(questions_asked).get_indices();

    return NodeToNodeAnswer<A_parameter, identifier_type>(std::move(exchanged_answers), std::move(indices));
}

/**
 * @brief A function that allows the communication from one node to another
 *  regardeless of the rank the node is on.
 *
 * Template parameters that can not be represented by a MPI Type are not allowed and result in undefined
 * behaviour
 *
 * Time complexity:
 *                      = O(number_nodes_per_rank * Time(generateQuestions)) +
 *                        O(number_of_generated_questions * Time(generateAnswers)) +
 *                        O(number_of_generated_questions)
 *
 * MPI messsage complexity:
 *                      Number of MPI data transfers :O(number_of_ranks)
 *                      Size of MPI data transfers :O(number_of_messages * (MEMORY(Q_parameter)+MEMORY(A_parameter))
 *
 * Possible causes of failure:
 *                      The function is supposed to be called by every rank that carries nodes of the graph.
 *                          Missing this requirement will result in undefined behaviour.
 *                      The function is supposed the receive valid questions by the generateQuestions method.
 *                          Targeting ranks and nodes that do not exist will result in undefined behaviour.
 *
 * @tparam Q_parameter type of the question message
 * @tparam A_parameter type of the answer message
 * @param generateQuestions Function to compute the "questions" for each node
 *                      A "question" is an entry in the return vector of this function defined by
 *                          std::tuple<rank_target, node_id_target, Q_parameter>. The first and second entry in the tuple
 *                          determine the rank and local id of the node the question is send to.
 *                          The third entry corresponds to a question-message that each question can have.
 *                          The type of the question-message is Q_parameter
 *                      Function is guaranteed to be called for every node in the graph.
 *                          The local id of the questioneer node is  "local_id"
 *                      The function can generate an arbitrary number of questions (including zero) per call
 * @param generateAnswers Function to compute the "answers" for each question
 *                      Function is guaranteed to be called for every "question" that was generated by generateQuestions
 *                          at the corresponding rank it was send to. The local id of the answering node is "local_id"
 *                          The answering node is the one specified with the question in the second tuple entry "node_id_target"
 *                      "Hint: The questioneer is not visible for the answerer if this information is not send in the message"
 *                      An "answer" is an instance of A_parameter. Each "question" will generate a generateAnswers call and
 *                          therefore an "answer". A_parameter is the return type of generateAnswers
 *                      The "answer" will be transmitted to the questioneer rank and node
 * @return NodeToNodeAnswer<A_parameter> contains the answers to the stated question, thats can be extracted using
 *                          get_answers_of_questioner_node(local_id) of NodeToNodeAnswer
 *                          Look at the NodeToNodeAnswer description for details
 */
template <MPICompatible Q_parameter, MPICompatible A_parameter, typename identifier_type>
    requires std::unsigned_integral<identifier_type>
[[nodiscard]] static NodeToNodeAnswer<A_parameter, identifier_type>
node_to_node_question(const identifier_type number_local_values,
                      const std::convertible_to<GenerateQuestionsFunction<Q_parameter, identifier_type>> auto& generateQuestions,
                      const std::convertible_to<GenerateAnswersFunction<A_parameter, Q_parameter, identifier_type>> auto& generateAnswers) {

    const auto ids = std::views::iota(identifier_type(0), number_local_values);
    const auto answer = node_to_node_question<Q_parameter, A_parameter, decltype(ids), identifier_type>(number_local_values, ids, generateQuestions, generateAnswers);
    return answer;
}

/**
 * @brief A function that allows the communication from one node to another
 *  regardeless of the rank the node is on.
 *
 * Template parameters that can not be represented by a MPI Type are not allowed and result in undefined
 * behaviour
 *
 * Time complexity:
 *                      = O(number_nodes_per_rank * Time(generateQuestions)) +
 *                        O(number_of_generated_questions * Time(generateAnswers)) +
 *                        O(number_of_generated_questions)
 *
 * MPI messsage complexity:
 *                      Number of MPI data transfers :O(number_of_ranks)
 *                      Size of MPI data transfers :O(number_of_messages * (MEMORY(Q_parameter)+MEMORY(A_parameter))
 *
 * Possible causes of failure:
 *                      The function is supposed to be called by every rank that carries nodes of the graph.
 *                          Missing this requirement will result in undefined behaviour.
 *                      The function is supposed the receive valid questions by the generateQuestions method.
 *                          Targeting ranks and nodes that do not exist will result in undefined behaviour.
 *
 * @tparam Q_parameter type of the question message
 * @param local_nodes_asking_questions The nodes that locally ask questions
 * @param generateQuestions Function to compute the "questions" for each node
 *                      A "question" is an entry in the return vector of this function defined by
 *                          std::tuple<rank_target, node_id_target, Q_parameter>. The first and second entry in the tuple
 *                          determine the rank and local id of the node the question is send to.
 *                          The third entry corresponds to a question-message that each question can have.
 *                          The type of the question-message is Q_parameter
 *                      Function is guaranteed to be called for every node in the graph.
 *                          The local id of the questioneer node is  "local_id"
 *                      The function can generate an arbitrary number of questions (including zero) per call
 * @param generateAnswers Function to compute the "answers" for each question
 *                      Function is guaranteed to be called for every "question" that was generated by generateQuestions
 *                          at the corresponding rank it was send to. The local id of the answering node is "local_id"
 *                          The answering node is the one specified with the question in the second tuple entry "node_id_target"
 *                      "Hint: The questioneer is not visible for the answerer if this information is not send in the message"
 *                      An "answer" is an instance of A_parameter. Each "question" will generate a generateAnswers call and
 *                          therefore an "answer". A_parameter is the return type of generateAnswers
 *                      The "answer" will be transmitted to the questioneer rank and node
 */
template <MPICompatible Q_parameter, typename NodesRange, typename identifier_type>
    requires std::unsigned_integral<identifier_type>
static void node_to_node_question(const identifier_type number_local_values,
                                  const NodesRange& local_nodes_asking_questions,
                                  const std::convertible_to<GenerateQuestionsFunction<Q_parameter, identifier_type>> auto& generateQuestions,
                                  const std::convertible_to<VoidAnswersFunction<Q_parameter, identifier_type>> auto& generateAnswers) {

    const auto questions_asked = detail::get_questions<Q_parameter, void, NodesRange, identifier_type>(number_local_values, local_nodes_asking_questions, generateQuestions);

    const auto number_questions_to_receive_per_rank = MPICollectives::all_to_all(questions_asked.get_number_questions_to_ask_per_rank());

    const auto questions_to_answer = detail::distribute_questions(questions_asked, number_questions_to_receive_per_rank, number_local_values);

    questions_to_answer.compute_answers(generateAnswers);
}

/**
 * @brief A function that allows the communication from one node to another
 *  regardeless of the rank the node is on.
 *
 * Template parameters that can not be represented by a MPI Type are not allowed and result in undefined
 * behaviour
 *
 * Time complexity:
 *                      = O(number_nodes_per_rank * Time(generateQuestions)) +
 *                        O(number_of_generated_questions * Time(generateAnswers)) +
 *                        O(number_of_generated_questions)
 *
 * MPI messsage complexity:
 *                      Number of MPI data transfers :O(number_of_ranks)
 *                      Size of MPI data transfers :O(number_of_messages * (MEMORY(Q_parameter)+MEMORY(A_parameter))
 *
 * Possible causes of failure:
 *                      The function is supposed to be called by every rank that carries nodes of the graph.
 *                          Missing this requirement will result in undefined behaviour.
 *                      The function is supposed the receive valid questions by the generateQuestions method.
 *                          Targeting ranks and nodes that do not exist will result in undefined behaviour.
 *
 * @tparam Q_parameter type of the question message
 * @param generateQuestions Function to compute the "questions" for each node
 *                      A "question" is an entry in the return vector of this function defined by
 *                          std::tuple<rank_target, node_id_target, Q_parameter>. The first and second entry in the tuple
 *                          determine the rank and local id of the node the question is send to.
 *                          The third entry corresponds to a question-message that each question can have.
 *                          The type of the question-message is Q_parameter
 *                      Function is guaranteed to be called for every node in the graph.
 *                          The local id of the questioneer node is  "local_id"
 *                      The function can generate an arbitrary number of questions (including zero) per call
 * @param generateAnswers Function to compute the "answers" for each question
 *                      Function is guaranteed to be called for every "question" that was generated by generateQuestions
 *                          at the corresponding rank it was send to. The local id of the answering node is "local_id"
 *                          The answering node is the one specified with the question in the second tuple entry "node_id_target"
 *                      "Hint: The questioneer is not visible for the answerer if this information is not send in the message"
 *                      An "answer" is an instance of A_parameter. Each "question" will generate a generateAnswers call and
 *                          therefore an "answer". A_parameter is the return type of generateAnswers
 *                      The "answer" will be transmitted to the questioneer rank and node
 */
template <MPICompatible Q_parameter, typename identifier_type>
    requires std::unsigned_integral<identifier_type>
static void node_to_node_question(const identifier_type number_local_values,
                                  const std::convertible_to<GenerateQuestionsFunction<Q_parameter, identifier_type>> auto& generateQuestions,
                                  const std::convertible_to<VoidAnswersFunction<Q_parameter, identifier_type>> auto& generateAnswers) {

    const auto ids = std::views::iota(identifier_type(0), number_local_values);
    node_to_node_question<Q_parameter, decltype(ids), identifier_type>(number_local_values, ids, generateQuestions, generateAnswers);
}
}; // namespace comm_patterns

} // namespace mpiPP
