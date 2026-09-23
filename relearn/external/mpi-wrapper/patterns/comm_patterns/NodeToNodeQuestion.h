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

#include "mpi-wrapper/core/MPIRank.h"
#include "mpi-wrapper/core/MPIRankRange.h"
#include "mpi-wrapper/core/MPITypes.h"
#include "mpi-wrapper/patterns/comm_patterns/Indices.h"
#include "mpi-wrapper/patterns/comm_patterns/Types.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>

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
    PrepareQuestionsRecv,
    ClosedQuestionsPreparation
};

template <MPICompatible QParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
class NodeToNodeQuestionBase {
public:
    /**
     * @brief Initializes the base
     * @param num_ranks The number of MPI ranks
     * @param number_local_nodes The number of local ids
     */
    NodeToNodeQuestionBase(const std::size_t num_ranks, const IdentifierType number_local_nodes)
        : number_ranks(num_ranks)
        , nodes_to_ask_question(num_ranks)
        , indices(num_ranks, number_local_nodes)
        , question_parameters(num_ranks)
        , addressee_ranks_to_number_of_questions(num_ranks, 0) { }

    virtual ~NodeToNodeQuestionBase() = default;

    /**
     * @brief Returns the ids (on the target rank) that must be asked a question for a specific rank
     * @param rank The rank
     * @exception Throws an Exception if rank is larger than the number of ranks in the constructor
     * @return The ids on the rank that must be asked a question
     */
    [[nodiscard]] std::span<const IdentifierType> get_nodes_to_ask_question_for_rank(const MPIRank rank) const {
        const auto index = rank.get_rank_cast();
        const auto size = nodes_to_ask_question.size();

        utility::Exception::check(index < size,
                                  "NodeToNodeQuestionBase::get_nodes_to_ask_question_for_rank: {} is larger than or equal to {}", index, size);

        return nodes_to_ask_question[index];
    }

    /**
     * @brief Returns the questioner-to-question ranges while retaining ownership
     * @return The per-questioner index ranges
     */
    [[nodiscard]] const Indices<IdentifierType>& get_indices() const& noexcept {
        return indices;
    }

    /**
     * @brief Moves the questioner-to-question ranges out of a temporary question structure
     * @return The per-questioner index ranges
     */
    [[nodiscard]] Indices<IdentifierType>&& get_indices() && noexcept {
        return std::move(indices);
    }

    /**
     * @brief Returns the questions for a specific rank
     * @param rank The rank
     * @exception Throws an Exception if rank is larger than the number of ranks in the constructor
     * @return The questions
     */
    [[nodiscard]] std::span<const QParameter> get_question_parameters_for_rank(const MPIRank rank) const {
        const auto index = rank.get_rank_cast();
        const auto size = question_parameters.size();

        utility::Exception::check(index < size,
                                  "NodeToNodeQuestionBase::get_question_parameters_for_rank: {} is larger than or equal to {}", index, size);

        return question_parameters[index];
    }

    /**
     * @brief Returns the number of questions that are asked for a specific rank
     * @exception Throws an Exception if the structure is not in the correct state (must be called after finalize_adding_questions_to_send)
     * @return The number of questions that are asked for a specific rank
     */
    [[nodiscard]] std::span<const int> get_number_questions_to_ask_per_rank() const {
        utility::Exception::check(structure_status == StatusType::ClosedQuestionsPreparation,
                                  "NodeToNodeQuestionBase::get_number_questions_to_ask_per_rank: Must not be called before add_questions_to_send");

        return addressee_ranks_to_number_of_questions;
    }

    /**
     * @brief Calls for each value in range the generate_questions function and adds the questions to the list of questions to send
     * @param range The local ids for which questions should be generated
     * @param generate_questions The function that generates the questions
     * @exception Throws an Exception if the structure is not in the correct state (must be called only directly after the constructor)
     */
    void add_questions_to_send(const auto& range,
                               const std::convertible_to<GenerateQuestionsFunction<QParameter, IdentifierType>> auto& generate_questions) {
        utility::Exception::check(structure_status == StatusType::Empty, "NodeToNodeQuestionBase::add_questions_to_send: The structure already contains questions");

        for (auto local_id : range) {
            add_questions_from_one_node_to_send(generate_questions(local_id), local_id);
        }

        finalize_adding_questions_to_send();
    }

    /**
     * @brief Distributes the questions received from all ranks into the per-rank lists.
     *      This function should be called on the rank that generates the answers, on a freshly constructed structure.
     * @param total_nodes_to_ask_question All received addressee nodes, ordered by source rank
     * @param total_question_parameters All received question parameters, corresponding to total_nodes_to_ask_question
     * @param rank_size The number of questions received from each rank
     * @param rank_displ The displacement in the total buffers where the questions of each rank start
     * @exception Throws an Exception if the structure already contains questions, if the sizes of the arguments
     *      do not match, or if any (displacement, size) pair references elements outside the total buffers
     */
    void set_questions_received(const std::span<const IdentifierType> total_nodes_to_ask_question,
                                const std::span<const QParameter> total_question_parameters,
                                const std::span<const int> rank_size,
                                const std::span<const int> rank_displ) {
        utility::Exception::check(structure_status == StatusType::Empty,
                                  "NodeToNodeQuestionBase::set_questions_received: The structure already contains questions");
        utility::Exception::check(rank_size.size() == number_ranks,
                                  "NodeToNodeQuestionBase::set_questions_received: The number of sizes {} does not match the number of ranks {}",
                                  rank_size.size(), number_ranks);
        utility::Exception::check(rank_size.size() == rank_displ.size(),
                                  "NodeToNodeQuestionBase::set_questions_received: The number of sizes {} does not match the number of displacements {}",
                                  rank_size.size(), rank_displ.size());
        utility::Exception::check(total_nodes_to_ask_question.size() == total_question_parameters.size(),
                                  "NodeToNodeQuestionBase::set_questions_received: The number of addressee nodes {} does not match the number of question parameters {}",
                                  total_nodes_to_ask_question.size(), total_question_parameters.size());

        for (auto rank = std::size_t{ 0 }; rank < rank_size.size(); ++rank) {
            const auto number_of_questions = rank_size[rank];
            utility::Exception::check(number_of_questions >= 0,
                                      "NodeToNodeQuestionBase::set_questions_received: The number of questions {} of rank {} is negative", number_of_questions, rank);

            const auto displ = rank_displ[rank];
            utility::Exception::check(displ >= 0,
                                      "NodeToNodeQuestionBase::set_questions_received: The displacement {} of rank {} is negative", displ, rank);

            if (number_of_questions == 0) {
                continue;
            }

            utility::Exception::check(utility::safe_cast<std::size_t>(displ) + utility::safe_cast<std::size_t>(number_of_questions) <= total_nodes_to_ask_question.size(),
                                      "NodeToNodeQuestionBase::set_questions_received: Rank {} references the elements [{}, {}), but there are only {}",
                                      rank, displ, displ + number_of_questions, total_nodes_to_ask_question.size());

            const auto questions_begin = total_nodes_to_ask_question.begin() + displ;
            nodes_to_ask_question[rank] = { questions_begin, questions_begin + number_of_questions };

            const auto parameters_begin = total_question_parameters.begin() + displ;
            question_parameters[rank] = { parameters_begin, parameters_begin + number_of_questions };

            addressee_ranks_to_number_of_questions[rank] = number_of_questions;
        }

        structure_status = StatusType::PrepareQuestionsRecv;
    }

protected:
    StatusType structure_status = StatusType::Empty;

    std::size_t number_ranks;

    // list of list of nodes to ask questions
    std::vector<std::vector<IdentifierType>> nodes_to_ask_question;

    // stores the index range in nodes_to_ask_question for each rank
    Indices<IdentifierType> indices;

    // list of list of parameters corresponding to nodes_to_ask_question
    std::vector<std::vector<QParameter>> question_parameters{};

    // list of all ranks to number of questions
    std::vector<int> addressee_ranks_to_number_of_questions;

private:
    void add_questions_from_one_node_to_send(std::vector<TargetedQuestion<QParameter, IdentifierType>>&& list_of_addressees_and_parameter,
                                             const IdentifierType questioner) {
        // Distribute questions to list of questions for each rank
        auto index_range_per_target_rank = std::vector<std::pair<std::optional<Index>, Index>>(number_ranks);

        for (auto&& [target_rank, target_local_node, question_parameter] : list_of_addressees_and_parameter) {
            const auto target_rank_cast = target_rank.get_rank_cast();
            utility::Exception::check(target_rank_cast < number_ranks,
                                      "NodeToNodeQuestionBase::add_questions_from_one_node_to_send: The question of node {} targets rank {}, but there are only {} ranks",
                                      questioner, target_rank_cast, number_ranks);

            const auto index = nodes_to_ask_question[target_rank_cast].size();
            if (!index_range_per_target_rank[target_rank_cast].first.has_value()) {
                index_range_per_target_rank[target_rank_cast].first = index;
            }
            // outside of else clause for correct ranges with one element
            index_range_per_target_rank[target_rank_cast].second = index + 1;

            nodes_to_ask_question[target_rank_cast].push_back(target_local_node);
            question_parameters[target_rank_cast].emplace_back(std::move(question_parameter));
        }

        auto index_ranges = std::vector<IndexRange>();
        index_ranges.reserve(number_ranks);
        for (const auto& [begin, end] : index_range_per_target_rank) {
            index_ranges.emplace_back(begin.value_or(0U), end);
        }

        indices.set_indices(questioner, std::move(index_ranges));
    }

    void finalize_adding_questions_to_send() {
        for (auto rank = std::size_t{ 0 }; rank < this->number_ranks; ++rank) {
            addressee_ranks_to_number_of_questions[rank] = utility::safe_cast<int>(nodes_to_ask_question[rank].size());
        }

        structure_status = StatusType::ClosedQuestionsPreparation;
    }
};

template <MPICompatible QParameter, MPICompatibleAnswer AParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
class NodeToNodeQuestion : public NodeToNodeQuestionBase<QParameter, IdentifierType> {
public:
    /**
     * @brief Initializes the base
     * @param number_ranks The number of MPI ranks
     * @param number_local_nodes The number of local ids
     */
    NodeToNodeQuestion(const std::size_t num_ranks, IdentifierType number_local_nodes)
        : NodeToNodeQuestionBase<QParameter, IdentifierType>(num_ranks, number_local_nodes) { }

    /**
     * @brief Computes the answers for the questions, i.e., calls the generate_answers function for each question.
     *      This function should be called on the rank that generates the answers.
     * @param generate_answers The function that generates the answers
     * @return The answers for the questions
     */
    [[nodiscard]] std::vector<std::vector<AParameter>>
    compute_answers(const std::convertible_to<GenerateAnswersFunction<AParameter, QParameter, IdentifierType>> auto& generate_answers) const {
        const auto num_ranks = utility::safe_cast<int>(this->number_ranks);
        auto answers_to_questions = std::vector<std::vector<AParameter>>(this->number_ranks);

        for (const auto rank : MPIRankRange::range(num_ranks)) {
            const auto rank_questions = this->get_question_parameters_for_rank(rank);
            const auto rank_nodes = this->get_nodes_to_ask_question_for_rank(rank);

            const auto number_questions = rank_questions.size();
            const auto number_nodes = rank_nodes.size();
            utility::Exception::check(number_nodes == number_questions, "NodeToNodeQuestion::compute_answers: The targeted nodes and the questions are of different sizes");

            auto& answers = answers_to_questions[rank.get_rank_cast()];
            answers.resize(number_questions);

            for (auto j = std::size_t{ 0 }; j < number_questions; ++j) {
                const auto local_id = rank_nodes[j];
                const auto& question = rank_questions[j];
                answers[j] = generate_answers(local_id, question);
            }
        }

        return answers_to_questions;
    }
};

template <MPICompatible QParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
class NodeToNodeQuestion<QParameter, void, IdentifierType> : public NodeToNodeQuestionBase<QParameter, IdentifierType> {
public:
    /**
     * @brief Initializes the base
     * @param number_ranks The number of MPI ranks
     * @param number_local_nodes The number of local ids
     */
    NodeToNodeQuestion(const std::size_t num_ranks, const IdentifierType number_local_nodes)
        : NodeToNodeQuestionBase<QParameter, IdentifierType>(num_ranks, number_local_nodes) { }

    /**
     * @brief Computes the answers for the questions, i.e., calls the generate_answers function for each question.
     *      In this instance, there will not be an answer transferred.
     *      This function must be called on the rank that generates the answers.
     * @param generate_answers The function that generates the answers
     */
    void compute_answers(const std::convertible_to<VoidAnswersFunction<QParameter, IdentifierType>> auto& generate_answers) const {
        const auto num_ranks = utility::safe_cast<int>(this->number_ranks);
        for (const auto rank : MPIRankRange::range(num_ranks)) {
            const auto rank_questions = this->get_question_parameters_for_rank(rank);
            const auto rank_nodes = this->get_nodes_to_ask_question_for_rank(rank);

            const auto number_questions = rank_questions.size();
            const auto number_nodes = rank_nodes.size();
            utility::Exception::check(number_nodes == number_questions, "NodeToNodeQuestion::compute_answers: The targeted nodes and the questions are of different sizes");

            for (auto j = std::size_t{ 0 }; j < number_questions; ++j) {
                const auto local_id = rank_nodes[j];
                const auto& question = rank_questions[j];
                generate_answers(local_id, question);
            }
        }
    }
};

} // namespace comm_patterns

} // namespace mpiPP
