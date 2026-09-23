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

#include "mpi-wrapper/collectives/MPIAllToAllV.h"
#include "mpi-wrapper/core/MPITypes.h"
#include "mpi-wrapper/patterns/comm_patterns_2/Indices.h"
#include "mpi-wrapper/patterns/comm_patterns_2/QuestionExchange.h"
#include "mpi-wrapper/patterns/comm_patterns_2/Types.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>
#include <cpp-utility/data/displacement.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace mpiPP {

namespace comm_patterns_2 {

/**
 * @brief The finalized, immutable send side of an answer exchange: one answer per received question,
 *      tightly packed in the order of the IncomingQuestions they answer. Created by
 *      AnswerBuilder::finalize and consumed by exchange_answers.
 */
template <MPICompatible AParameter>
class OutgoingAnswers {
public:
    /**
     * @brief Assembles the send side from an already flattened buffer.
     *      Usually not called directly; use AnswerBuilder instead.
     * @param answers_list The answers, tightly packed in source-rank order of the answered questions
     * @param counts The number of answers destined for each rank
     * @exception Throws an Exception if any count is negative or if the counts do not sum up to the
     *      number of answers
     */
    OutgoingAnswers(std::vector<AParameter> answers_list, std::vector<int> counts)
        : answers(std::move(answers_list))
        , number_answers_per_rank(std::move(counts)) {
        auto total = std::size_t{ 0 };
        for (const auto count : number_answers_per_rank) {
            // throws for negative counts
            total += utility::safe_cast<std::size_t>(count);
        }
        utility::Exception::check(total == answers.size(),
                                  "OutgoingAnswers::OutgoingAnswers: The counts sum up to {}, but there are {} answers",
                                  total, answers.size());

        displacements = utility::calculate_displacements<int>(number_answers_per_rank);
    }

    OutgoingAnswers(const OutgoingAnswers&) = delete;
    OutgoingAnswers& operator=(const OutgoingAnswers&) = delete;
    OutgoingAnswers(OutgoingAnswers&&) noexcept = default;
    OutgoingAnswers& operator=(OutgoingAnswers&&) noexcept = default;
    ~OutgoingAnswers() = default;

    /**
     * @brief Returns the answers, tightly packed in rank order
     * @return The answers
     */
    [[nodiscard]] std::span<const AParameter> get_answers() const noexcept {
        return answers;
    }

    /**
     * @brief Returns the number of answers destined for each rank
     * @return The number of answers per rank
     */
    [[nodiscard]] std::span<const int> get_number_answers_per_rank() const noexcept {
        return number_answers_per_rank;
    }

    /**
     * @brief Returns where the answers destined for each rank start in the flat buffer
     * @return The displacements per rank
     */
    [[nodiscard]] std::span<const int> get_displacements() const noexcept {
        return displacements;
    }

private:
    std::vector<AParameter> answers;
    std::vector<int> number_answers_per_rank;
    std::vector<int> displacements;
};

/**
 * @brief The mutable answer phase of an exchange: the local rank pushes one answer per received
 *      question, in the iteration order of the IncomingQuestions; finalize checks completeness and
 *      turns the answers into an immutable OutgoingAnswers.
 */
template <MPICompatible AParameter>
class AnswerBuilder {
public:
    /**
     * @brief Initializes the builder for the given received questions
     * @param questions The questions the local rank has to answer
     */
    template <MPICompatible QParameter, typename IdentifierType>
        requires std::unsigned_integral<IdentifierType>
    explicit AnswerBuilder(const IncomingQuestions<QParameter, IdentifierType>& questions)
        : number_answers_per_rank(questions.get_number_questions_per_rank().begin(), questions.get_number_questions_per_rank().end())
        , number_expected_answers(questions.get_number_questions()) {
        answers.reserve(number_expected_answers);
    }

    /**
     * @brief Returns the total number of answers the builder expects
     * @return The number of expected answers
     */
    [[nodiscard]] std::size_t get_number_expected_answers() const noexcept {
        return number_expected_answers;
    }

    /**
     * @brief Returns the number of answers pushed so far
     * @return The number of pushed answers
     */
    [[nodiscard]] std::size_t get_number_pushed_answers() const noexcept {
        return answers.size();
    }

    /**
     * @brief Appends the answer to the next unanswered question, in the iteration order of the
     *      IncomingQuestions the builder was created from
     * @param answer The answer
     * @exception Throws an Exception if the builder was already finalized or if all questions are
     *      already answered
     */
    void push_answer(AParameter answer) {
        utility::Exception::check(!finalized, "AnswerBuilder::push_answer: The builder was already finalized");
        utility::Exception::check(answers.size() < number_expected_answers,
                                  "AnswerBuilder::push_answer: All {} questions are already answered", number_expected_answers);

        answers.emplace_back(std::move(answer));
    }

    /**
     * @brief Ends the local answer phase: checks that every question was answered and hands the
     *      answers over to the immutable send-side structure. The builder must not be used afterwards.
     * @exception Throws an Exception if the builder was already finalized or if not all questions
     *      were answered
     * @return The finalized send side, ready for exchange_answers
     */
    [[nodiscard]] OutgoingAnswers<AParameter> finalize() {
        utility::Exception::check(!finalized, "AnswerBuilder::finalize: The builder was already finalized");
        utility::Exception::check(answers.size() == number_expected_answers,
                                  "AnswerBuilder::finalize: Only {} of {} questions were answered", answers.size(), number_expected_answers);

        auto outgoing = OutgoingAnswers<AParameter>(std::move(answers), std::move(number_answers_per_rank));
        finalized = true;
        return outgoing;
    }

private:
    std::vector<AParameter> answers;
    std::vector<int> number_answers_per_rank;
    std::size_t number_expected_answers;
    bool finalized = false;
};

/**
 * @brief The immutable receive side of an answer exchange: the answers to the questions the local
 *      rank asked, retrievable per questioner node.
 */
template <typename AParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
class IncomingAnswers {
public:
    /**
     * @brief Assembles the receive side from the received answers and the questioner mapping.
     *      Usually not called directly; use exchange_answers instead.
     * @param answers_list All received answers, tightly packed in answering-rank order
     * @param indices_structure The questioner-to-question mapping, extracted from the OutgoingQuestions
     */
    IncomingAnswers(std::vector<AParameter> answers_list, Indices<IdentifierType> indices_structure)
        : answers_for_my_rank(std::move(answers_list))
        , questioner_node_to_answers_index_range(std::move(indices_structure)) {

        questioner_node_to_answers_index_range.normalize();
    }

    /**
     * @brief Returns the answers to the questions the given node asked, in the same order as the
     *      questions were added to the QuestionBuilder.
     *
     * @note Other than the order guarantee, there is no correspondence to the specific question
     *       each answer was created for. If that is needed, transfer a question identifier in
     *       the question and answer payloads.
     * @param local_id The local id of the questioner node
     * @exception Throws an Exception if local_id is too large or if the stored index ranges reference
     *      answers that do not exist
     * @return The answers for the specified questioner node
     */
    [[nodiscard]] std::vector<AParameter> get_answers_of_questioner_node(const IdentifierType local_id) const {
        utility::Exception::check(local_id < questioner_node_to_answers_index_range.get_number_local_values(),
                                  "IncomingAnswers::get_answers_of_questioner_node: The local id {} is larger than or equal to the number of local values {}",
                                  local_id, questioner_node_to_answers_index_range.get_number_local_values());

        if (answers_for_my_rank.empty()) {
            return {};
        }

        const auto index_ranges_for_node = questioner_node_to_answers_index_range.get_indices(local_id);

        auto total_number_answers = std::size_t{ 0 };
        for (const auto& [begin, end] : index_ranges_for_node) {
            if (begin < end) {
                total_number_answers += end - begin;
            }
        }

        auto answers = std::vector<AParameter>{};
        answers.reserve(total_number_answers);
        for (const auto& [begin, end] : index_ranges_for_node) {
            if (begin == end) {
                continue;
            }

            utility::Exception::check(begin < answers_for_my_rank.size(),
                                      "IncomingAnswers::get_answers_of_questioner_node: The range [{}, {}) of local id {} starts past the {} stored answers",
                                      begin, end, local_id, answers_for_my_rank.size());
            utility::Exception::check(end <= answers_for_my_rank.size(),
                                      "IncomingAnswers::get_answers_of_questioner_node: The range [{}, {}) of local id {} ends past the {} stored answers",
                                      begin, end, local_id, answers_for_my_rank.size());

            answers.insert(answers.end(), answers_for_my_rank.begin() + utility::safe_cast<std::int64_t>(begin),
                           answers_for_my_rank.begin() + utility::safe_cast<std::int64_t>(end));
        }

        return answers;
    }

private:
    // List of answers to the questions the local rank asked.
    // The vector is logically divided into subranges of answers from other ranks.
    // These subranges are further divided into subranges of answers for each questioner of this rank.
    std::vector<AParameter> answers_for_my_rank;

    // stores the index ranges into answers_for_my_rank for each questioner node
    Indices<IdentifierType> questioner_node_to_answers_index_range;
};

/**
 * @brief Exchanges the answers between all ranks. This is a collective operation: every rank must
 *      call this function, even if it has no answers to send. Consumes both send-side structures:
 *      the answers are transferred, and the questioner mapping of the questions moves into the result.
 * @tparam AParameter The type of the answer payload
 * @tparam QParameter The type of the question payload
 * @tparam IdentifierType The type of the local node ids
 * @param answers The finalized local answers to the received questions
 * @param questions The finalized local questions whose answers are received
 * @exception Throws an Exception if the two structures disagree on the number of ranks or mpi returns
 *      an error code
 * @return The answers to the questions the local rank asked
 */
template <MPICompatible AParameter, MPICompatible QParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
[[nodiscard]] IncomingAnswers<AParameter, IdentifierType> exchange_answers(OutgoingAnswers<AParameter> answers,
                                                                           OutgoingQuestions<QParameter, IdentifierType> questions) {
    // The answers come back exactly where the questions were sent from: the send counts and
    // displacements of the questions describe the receive layout of the answers.
    const auto receive_counts = questions.get_number_questions_per_rank();
    const auto receive_displacements = questions.get_displacements();

    utility::Exception::check(answers.get_number_answers_per_rank().size() == receive_counts.size(),
                              "comm_patterns_2::exchange_answers: The answers know {} ranks, but the questions know {}",
                              answers.get_number_answers_per_rank().size(), receive_counts.size());

    auto received_answers = MPICollectives::all_to_all_v<AParameter>(answers.get_answers().data(), answers.get_number_answers_per_rank().data(),
                                                                     answers.get_displacements().data(), receive_counts.data(), receive_displacements.data());

    return IncomingAnswers<AParameter, IdentifierType>(std::move(received_answers), questions.extract_indices());
}

} // namespace comm_patterns_2

} // namespace mpiPP
