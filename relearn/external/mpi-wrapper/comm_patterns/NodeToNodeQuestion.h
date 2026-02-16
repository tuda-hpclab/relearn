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

#include "mpi-wrapper/MPIRank.h"
#include "mpi-wrapper/MPITypes.h"
#include "mpi-wrapper/comm_patterns/Indices.h"
#include "mpi-wrapper/comm_patterns/Types.h"

#include "cpp-utility/Exception.hpp"

#include <concepts>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace mpiPP {

namespace comm_patterns {

enum class StatusType : std::uint8_t {
    Empty,
    PrepareQuestionsToSend,
    PrepareQuestionsRecv,
    ClosedQuestionsPreparation
};

template <MPICompatible Q_parameter, typename identifier_type>
    requires std::unsigned_integral<identifier_type>
class NodeToNodeQuestionBase {
public:
    static_assert(std::is_integral_v<identifier_type>);
    static_assert(std::is_unsigned_v<identifier_type>);

    /**
     * @brief Initializes the base
     * @param number_ranks The number of MPI ranks
     * @param number_local_nodes The number of local ids
     */
    NodeToNodeQuestionBase(const std::size_t num_ranks, const identifier_type number_local_nodes)
        : number_ranks(num_ranks)
        , nodes_to_ask_question(num_ranks)
        , nodes_that_ask_the_question(num_ranks)
        , indices(num_ranks, number_local_nodes)
        , question_parameters(num_ranks)
        , addressee_ranks_to_nbrOfQuestions(num_ranks, 0) {

        for (auto i = std::size_t{ 0 }; i < num_ranks; i++) {
            // This is here to avoid nullptr later on
            nodes_to_ask_question[i].reserve(1);
            nodes_that_ask_the_question[i].reserve(1);
            question_parameters[i].reserve(1);
        }
    }

    virtual ~NodeToNodeQuestionBase() = default;

    /**
     * @brief Returns the ids (on the target rank) that must be asked a question for a specific rank
     * @param rank The rank
     * @exception Throws an Exception if rank is larger than the number of ranks in the constructor
     * @return The ids on the rank that must be asked a question
     */
    [[nodiscard]] std::span<const identifier_type> get_nodes_to_ask_question_for_rank(const MPIRank rank) const {
        const auto index = rank.get_rank_cast();
        const auto size = nodes_to_ask_question.size();

        utility::Exception::check(index < size,
                                  "NodeToNodeQuestionBase::get_nodes_to_ask_question_for_rank: {} is larger than or equal to {}", index, size);

        return nodes_to_ask_question[index];
    }

    /**
     * @brief Returns the ids (on the current rank) that ask a question for a specific rank
     * @param rank The rank
     * @exception Throws an Exception if rank is larger than the number of ranks in the constructor
     * @return The ids on the current rank that ask a question
     */
    [[nodiscard]] std::span<const identifier_type> get_nodes_that_ask_the_question_for_rank(const MPIRank rank) const {
        const auto index = rank.get_rank_cast();
        const auto size = nodes_that_ask_the_question.size();

        utility::Exception::check(index < nodes_that_ask_the_question.size(),
                                  "NodeToNodeQuestionBase::get_nodes_that_ask_the_question_for_rank: {} is larger than or equal to {}", index, size);

        return nodes_that_ask_the_question[index];
    }

    /**
     * @brief Returns the index range for the specified questioner. Only returns sensible values after add_questions_to_send
     * @param questioner The local id of the questioner
     * @return The index range
     */
    [[nodiscard]] std::span<const IndexRange> get_questioner_node_to_answers_index_range(const identifier_type questioner) const {
        return indices.get_indices(questioner);
    }

    [[nodiscard]] const Indices<identifier_type>& get_indices() const& noexcept {
        return indices;
    }

    [[nodiscard]] Indices<identifier_type>&& get_indices() && noexcept {
        return std::move(indices);
    }

    /**
     * @brief Returns the questions for a specific rank
     * @param rank The rank
     * @exception Throws an Exception if rank is larger than the number of ranks in the constructor
     * @return The questions
     */
    [[nodiscard]] std::span<const Q_parameter> get_question_parameters_for_rank(const MPIRank rank) const {
        const auto index = rank.get_rank_cast();
        utility::Exception::check(index < question_parameters.size(), "NodeToNodeQuestionBase::get_question_parameters_for_rank");

        return question_parameters[index];
    }

    /**
     * @brief Returns the number of questions that are asked for a specific rank
     * @exception Throws an Exception if the structure is not in the correct state (must be called after finalize_adding_questions_to_send)
     * @return The number of questions that are asked for a specific rank
     */
    [[nodiscard]] std::span<const int> get_number_questions_to_ask_per_rank() const {
        utility::Exception::check(structure_status == StatusType::ClosedQuestionsPreparation, "NodeToNodeQuestionBase::get_number_questions_to_ask_per_rank");

        return addressee_ranks_to_nbrOfQuestions;
    }

    /**
     * @brief Calls for each value in range the generateQuestions function and adds the questions to the list of questions to send
     * @param range The local ids for which questions should be generated
     * @param generateQuestions The function that generates the questions
     * @exception Throws an Exception if the structure is not in the correct state (must be called only directly after the constructor)
     */
    void add_questions_to_send(const auto& range,
                               const std::convertible_to<GenerateQuestionsFunction<Q_parameter, identifier_type>> auto& generateQuestions) {
        utility::Exception::check(structure_status == StatusType::Empty, "NodeToNodeQuestionBase::add_questions_to_send");

        for (auto local_id : range) {
            add_questions_from_one_node_to_send(generateQuestions(local_id), local_id);
        }

        structure_status = StatusType::PrepareQuestionsToSend;
        finalize_adding_questions_to_send();
    }

    // Functions for use on answerer rank side
    void set_questions_received(const std::span<const identifier_type> total_nodes_to_ask_question,
                                const std::span<const Q_parameter> total_question_parameters,
                                const std::span<const int> rank_size,
                                const std::span<const int> rank_displ) {

        // Distribute questions parameters and addressees to lists in the NodeToNodeQuestion
        utility::Exception::check(rank_size.size() == rank_displ.size(), "NodeToNodeQuestionBase::set_questions_received: E1");

        for (auto rank = std::size_t{ 0 }; rank < rank_size.size(); ++rank) {
            if (const auto nbr_of_questions = rank_size[rank]; nbr_of_questions > 0) {
                utility::Exception::check(utility::save_cast<std::size_t>(rank_displ[rank]) < total_nodes_to_ask_question.size() && rank_displ[rank] >= 0,
                                          "NodeToNodeQuestionBase::set_questions_received: E1");

                // Distribute addressee nodes in the NodeToNodeQuestion
                utility::Exception::check(rank_displ[rank] >= 0 && utility::save_cast<std::size_t>(rank_displ[rank]) < total_nodes_to_ask_question.size(),
                                          "NodeToNodeQuestionBase::set_questions_received: E1");

                const auto questions_begin = total_nodes_to_ask_question.begin() + rank_displ[rank];
                nodes_to_ask_question[rank] = { questions_begin, questions_begin + nbr_of_questions };

                // Distribute question parameters in the NodeToNodeQuestion
                utility::Exception::check(utility::save_cast<std::size_t>(rank_displ[rank]) < total_question_parameters.size() && rank_displ[rank] >= 0,
                                          "NodeToNodeQuestionBase::set_questions_received: E1");

                const auto parameters_begin = total_question_parameters.begin() + rank_displ[rank];
                question_parameters[rank] = { parameters_begin, parameters_begin + nbr_of_questions };

                utility::Exception::check(rank < question_parameters.size(),
                                          "NodeToNodeQuestionBase::set_questions_received: E1");
                utility::Exception::check(utility::save_cast<std::size_t>(nbr_of_questions) <= question_parameters[rank].size(),
                                          "NodeToNodeQuestionBase::set_questions_received: E1");

                addressee_ranks_to_nbrOfQuestions.push_back(nbr_of_questions);
            }
        }
    }

protected:
    StatusType structure_status = StatusType::Empty;

    std::size_t number_ranks;

    // list of list of nodes to ask questions
    std::vector<std::vector<identifier_type>> nodes_to_ask_question;

    // list of list of nodes that ask the questions coresponding to nodes_to_ask_question
    std::vector<std::vector<identifier_type>> nodes_that_ask_the_question;

    // stores the index range in nodes_to_ask_question for each rank
    Indices<identifier_type> indices;

    // list of list of parameters coresponding to nodes_to_ask_question
    std::vector<std::vector<Q_parameter>> question_parameters{};

    // list of all ranks to number of questions
    std::vector<int> addressee_ranks_to_nbrOfQuestions;

private:
    void add_questions_from_one_node_to_send(std::vector<TargetedQuestion<Q_parameter, identifier_type>>&& list_of_addressees_and_parameter,
                                             const identifier_type questioner) {
        // Distribute questions to list of questions for each rank
        auto index_range_per_target_rank = std::vector<std::pair<std::optional<Index>, Index>>(number_ranks);

        for (auto&& [target_rank, target_local_node, Q_parameter_struct] : list_of_addressees_and_parameter) {
            const auto target_rank_cast = target_rank.get_rank_cast();
            utility::Exception::check(target_rank_cast < number_ranks, "");
            utility::Exception::check(target_rank_cast < nodes_to_ask_question.size(), "");
            utility::Exception::check(target_rank_cast < nodes_that_ask_the_question.size(), "");
            utility::Exception::check(target_rank_cast < question_parameters.size(), "");

            utility::Exception::check(nodes_that_ask_the_question[target_rank_cast].size() == nodes_to_ask_question[target_rank_cast].size(), "");
            utility::Exception::check(nodes_that_ask_the_question[target_rank_cast].size() == question_parameters[target_rank_cast].size(), "");

            const auto index = nodes_to_ask_question[target_rank_cast].size();
            if (!index_range_per_target_rank[target_rank_cast].first.has_value()) {
                index_range_per_target_rank[target_rank_cast].first = index;
            }
            // outside of else clause for correct ranges with one element
            index_range_per_target_rank[target_rank_cast].second = index + 1;

            nodes_to_ask_question[target_rank_cast].push_back(target_local_node);
            nodes_that_ask_the_question[target_rank_cast].push_back(questioner);
            question_parameters[target_rank_cast].emplace_back(std::move(Q_parameter_struct));
        }

        auto index_ranges = std::vector<IndexRange>();
        index_ranges.reserve(number_ranks);
        for (const auto& [begin, end] : index_range_per_target_rank) {
            index_ranges.emplace_back(begin.value_or(0U), end);
        }

        indices.set_indices(questioner, std::move(index_ranges));
    }

    void finalize_adding_questions_to_send() {
        utility::Exception::check(structure_status == StatusType::Empty || structure_status == StatusType::PrepareQuestionsToSend, "");
        utility::Exception::check(nodes_to_ask_question.size() == nodes_that_ask_the_question.size(), "");
        utility::Exception::check(nodes_that_ask_the_question.size() == question_parameters.size(), "");

        for (auto rank = std::size_t{ 0 }; rank < this->number_ranks; ++rank) {
            addressee_ranks_to_nbrOfQuestions[rank] = utility::save_cast<int>(nodes_to_ask_question[rank].size());
        }

        structure_status = StatusType::ClosedQuestionsPreparation;
    }
};

template <MPICompatible Q_parameter, MPICompatibleAnswer A_parameter, typename identifier_type>
    requires std::unsigned_integral<identifier_type>
class NodeToNodeQuestion : public NodeToNodeQuestionBase<Q_parameter, identifier_type> {
public:
    /**
     * @brief Initializes the base
     * @param number_ranks The number of MPI ranks
     * @param number_local_nodes The number of local ids
     */
    NodeToNodeQuestion(const std::size_t num_ranks, identifier_type number_local_nodes)
        : NodeToNodeQuestionBase<Q_parameter, identifier_type>(num_ranks, number_local_nodes) { }

    /**
     * @brief Computes the answers for the questions, i.e., calls the generateAnswers function for each question.
     *      This function should be called on the rank that generates the answers.
     * @param generateAnswers The function that generates the answers
     * @return The answers for the questions
     */
    [[nodiscard]] std::vector<std::vector<A_parameter>>
    compute_answers(const std::convertible_to<GenerateAnswersFunction<A_parameter, Q_parameter, identifier_type>> auto& generateAnswers) const {
        const auto num_ranks = utility::save_cast<int>(this->number_ranks);
        auto answers_to_questions = std::vector<std::vector<A_parameter>>(this->number_ranks);

        for (const auto rank : MPIRank::range(num_ranks)) {
            const auto rank_questions = this->get_question_parameters_for_rank(rank);
            const auto rank_nodes = this->get_nodes_to_ask_question_for_rank(rank);

            const auto number_questions = rank_questions.size();
            const auto number_nodes = rank_nodes.size();
            utility::Exception::check(number_nodes == number_questions, "NodeToNodeQuestion::compute_answers: The targetted nodes and the questions are of different sizes");

            auto& answers = answers_to_questions[rank.get_rank_cast()];
            answers.resize(number_questions);
            answers.reserve(1);

            for (auto j = std::size_t{ 0 }; j < number_questions; ++j) {
                const auto local_id = rank_nodes[j];
                const auto& question = rank_questions[j];
                answers[j] = generateAnswers(local_id, question);
            }
        }

        return answers_to_questions;
    }
};

template <MPICompatible Q_parameter, typename identifier_type>
    requires std::unsigned_integral<identifier_type>
class NodeToNodeQuestion<Q_parameter, void, identifier_type> : public NodeToNodeQuestionBase<Q_parameter, identifier_type> {
public:
    /**
     * @brief Initializes the base
     * @param number_ranks The number of MPI ranks
     * @param number_local_nodes The number of local ids
     */
    NodeToNodeQuestion(const std::size_t num_ranks, const identifier_type number_local_nodes)
        : NodeToNodeQuestionBase<Q_parameter, identifier_type>(num_ranks, number_local_nodes) { }

    /**
     * @brief Computes the answers for the questions, i.e., calls the generateAnswers function for each question.
     *      In this instance, there will not be an answer transfered.
     *      This function must be called on the rank that generates the answers.
     * @param generateAnswers The function that generates the answers
     */
    void compute_answers(const std::convertible_to<VoidAnswersFunction<Q_parameter, identifier_type>> auto& generateAnswers) const {
        const auto num_ranks = utility::save_cast<int>(this->number_ranks);
        for (const auto rank : MPIRank::range(num_ranks)) {
            const auto rank_questions = this->get_question_parameters_for_rank(rank);
            const auto rank_nodes = this->get_nodes_to_ask_question_for_rank(rank);

            const auto number_questions = rank_questions.size();
            const auto number_nodes = rank_nodes.size();
            utility::Exception::check(number_nodes == number_questions, "NodeToNodeQuestion::compute_answers: The targetted nodes and the questions are of different sizes");

            for (auto j = std::size_t{ 0 }; j < number_questions; ++j) {
                const auto local_id = rank_nodes[j];
                const auto& question = rank_questions[j];
                generateAnswers(local_id, question);
            }
        }
    }
};

}; // namespace comm_patterns

} // namespace mpiPP
