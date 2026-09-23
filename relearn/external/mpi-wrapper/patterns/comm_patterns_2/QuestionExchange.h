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
#include "mpi-wrapper/core/MPITypes.h"
#include "mpi-wrapper/patterns/comm_patterns_2/Indices.h"
#include "mpi-wrapper/patterns/comm_patterns_2/Types.h"

#include <cpp-utility/Cast.hpp>
#include <cpp-utility/Exception.hpp>
#include <cpp-utility/data/displacement.hpp>

#include <concepts>
#include <cstddef>
#include <iterator>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace mpiPP {

namespace comm_patterns_2 {

/**
 * @brief The finalized, immutable send side of a question exchange: for each rank the addressed nodes
 *      and the question parameters, tightly packed in rank order, plus the mapping from each questioner
 *      node to its questions. Created by QuestionBuilder::finalize, read by exchange_questions, and
 *      finally consumed by exchange_answers.
 */
template <MPICompatible QParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
class OutgoingQuestions {
public:
    /**
     * @brief Assembles the send side from already flattened buffers.
     *      Usually not called directly; use QuestionBuilder instead.
     * @param target_nodes_list The addressed nodes of all questions, tightly packed in rank order
     * @param question_parameters_list The question parameters, corresponding to target_nodes_list
     * @param counts The number of questions destined for each rank
     * @param indices_structure The mapping from questioner node to its questions, as built by QuestionBuilder
     * @exception Throws an Exception if the buffer sizes do not match, if the number of counts does not
     *      match the number of ranks, if any count is negative, or if the counts do not sum up to the
     *      number of questions
     */
    OutgoingQuestions(std::vector<IdentifierType> target_nodes_list, std::vector<QParameter> question_parameters_list,
                      std::vector<int> counts, Indices<IdentifierType> indices_structure)
        : target_nodes(std::move(target_nodes_list))
        , question_parameters(std::move(question_parameters_list))
        , number_questions_per_rank(std::move(counts))
        , indices(std::move(indices_structure)) {
        utility::Exception::check(target_nodes.size() == question_parameters.size(),
                                  "OutgoingQuestions::OutgoingQuestions: The number of target nodes {} does not match the number of question parameters {}",
                                  target_nodes.size(), question_parameters.size());
        utility::Exception::check(number_questions_per_rank.size() == indices.get_number_ranks(),
                                  "OutgoingQuestions::OutgoingQuestions: The number of counts {} does not match the number of ranks {}",
                                  number_questions_per_rank.size(), indices.get_number_ranks());

        auto total = std::size_t{ 0 };
        for (const auto count : number_questions_per_rank) {
            // throws for negative counts
            total += utility::safe_cast<std::size_t>(count);
        }
        utility::Exception::check(total == target_nodes.size(),
                                  "OutgoingQuestions::OutgoingQuestions: The counts sum up to {}, but there are {} questions",
                                  total, target_nodes.size());

        displacements = utility::calculate_displacements<int>(number_questions_per_rank);
    }

    OutgoingQuestions(const OutgoingQuestions&) = delete;
    OutgoingQuestions& operator=(const OutgoingQuestions&) = delete;
    OutgoingQuestions(OutgoingQuestions&&) noexcept = default;
    OutgoingQuestions& operator=(OutgoingQuestions&&) noexcept = default;
    ~OutgoingQuestions() = default;

    /**
     * @brief Returns the addressed nodes of all questions, tightly packed in rank order
     * @return The addressed nodes
     */
    [[nodiscard]] std::span<const IdentifierType> get_target_nodes() const noexcept {
        return target_nodes;
    }

    /**
     * @brief Returns the question parameters, corresponding to get_target_nodes
     * @return The question parameters
     */
    [[nodiscard]] std::span<const QParameter> get_question_parameters() const noexcept {
        return question_parameters;
    }

    /**
     * @brief Returns the number of questions destined for each rank
     * @return The number of questions per rank
     */
    [[nodiscard]] std::span<const int> get_number_questions_per_rank() const noexcept {
        return number_questions_per_rank;
    }

    /**
     * @brief Returns where the questions destined for each rank start in the flat buffers
     * @return The displacements per rank
     */
    [[nodiscard]] std::span<const int> get_displacements() const noexcept {
        return displacements;
    }

    /**
     * @brief Moves the questioner-to-question mapping out of the structure. Called by exchange_answers
     *      to hand the mapping over to the received answers; must be called at most once.
     * @exception Throws an Exception if the indices were already extracted
     * @return The mapping from questioner node to its questions
     */
    [[nodiscard]] Indices<IdentifierType> extract_indices() {
        utility::Exception::check(!indices_extracted, "OutgoingQuestions::extract_indices: The indices were already extracted");
        indices_extracted = true;
        return std::move(indices);
    }

private:
    std::vector<IdentifierType> target_nodes;
    std::vector<QParameter> question_parameters;
    std::vector<int> number_questions_per_rank;
    std::vector<int> displacements;
    Indices<IdentifierType> indices;
    bool indices_extracted = false;
};

/**
 * @brief The mutable build phase of a question exchange: each questioner node adds its questions,
 *      finalize turns the collected questions into an immutable OutgoingQuestions.
 */
template <MPICompatible QParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
class QuestionBuilder {
public:
    /**
     * @brief Initializes an empty builder
     * @param number_local_nodes The number of local ids that may ask questions
     */
    explicit QuestionBuilder(const IdentifierType number_local_nodes)
        : number_ranks(MPIInfo::get_number_ranks_cast())
        , nodes_per_rank(number_ranks)
        , parameters_per_rank(number_ranks)
        , indices(number_ranks, number_local_nodes) { }

    /**
     * @brief Adds all questions of one questioner node. All questions of a node must be added in one
     *      call; adding a non-empty list twice for the same node throws. Adding an empty list is a no-op.
     * @param questioner The local id of the asking node
     * @param questions The questions of the node, each addressing a (rank, node) pair with a payload
     * @exception Throws an Exception if the builder was already finalized, if questioner is not smaller
     *      than the number of local nodes, if any question targets a rank that does not exist, or if the
     *      node already added questions
     */
    void add_questions(const IdentifierType questioner, std::vector<TargetedQuestion<QParameter, IdentifierType>> questions) {
        utility::Exception::check(!finalized, "QuestionBuilder::add_questions: The builder was already finalized");
        utility::Exception::check(questioner < indices.get_number_local_values(),
                                  "QuestionBuilder::add_questions: The questioner id {} is not smaller than the number of local nodes {}",
                                  questioner, indices.get_number_local_values());

        if (questions.empty()) {
            return;
        }

        auto index_range_per_target_rank = std::vector<std::pair<std::optional<Index>, Index>>(number_ranks);

        for (auto&& [target_rank, target_node, parameter] : questions) {
            const auto target_rank_cast = target_rank.get_rank_cast();
            utility::Exception::check(target_rank_cast < number_ranks,
                                      "QuestionBuilder::add_questions: The question of node {} targets rank {}, but there are only {} ranks",
                                      questioner, target_rank_cast, number_ranks);

            const auto index = nodes_per_rank[target_rank_cast].size();
            if (!index_range_per_target_rank[target_rank_cast].first.has_value()) {
                index_range_per_target_rank[target_rank_cast].first = index;
            }
            // outside of else clause for correct ranges with one element
            index_range_per_target_rank[target_rank_cast].second = index + 1;

            nodes_per_rank[target_rank_cast].push_back(target_node);
            parameters_per_rank[target_rank_cast].emplace_back(std::move(parameter));
        }

        auto index_ranges = std::vector<IndexRange>{};
        index_ranges.reserve(number_ranks);
        for (const auto& [begin, end] : index_range_per_target_rank) {
            index_ranges.emplace_back(begin.value_or(0U), end);
        }

        indices.set_indices(questioner, std::move(index_ranges));
    }

    /**
     * @brief Ends the local build phase: flattens the collected questions into the immutable send-side
     *      structure. The builder must not be used afterwards.
     * @exception Throws an Exception if the builder was already finalized
     * @return The finalized send side, ready for exchange_questions
     */
    [[nodiscard]] OutgoingQuestions<QParameter, IdentifierType> finalize() {
        utility::Exception::check(!finalized, "QuestionBuilder::finalize: The builder was already finalized");
        finalized = true;

        auto counts = std::vector<int>{};
        counts.reserve(number_ranks);
        auto total = std::size_t{ 0 };
        for (const auto& nodes : nodes_per_rank) {
            counts.emplace_back(utility::safe_cast<int>(nodes.size()));
            total += nodes.size();
        }

        auto flat_nodes = std::vector<IdentifierType>{};
        flat_nodes.reserve(total);
        auto flat_parameters = std::vector<QParameter>{};
        flat_parameters.reserve(total);
        for (auto rank = std::size_t{ 0 }; rank < number_ranks; rank++) {
            flat_nodes.insert(flat_nodes.end(), nodes_per_rank[rank].begin(), nodes_per_rank[rank].end());
            flat_parameters.insert(flat_parameters.end(), parameters_per_rank[rank].begin(), parameters_per_rank[rank].end());
        }

        return OutgoingQuestions<QParameter, IdentifierType>(std::move(flat_nodes), std::move(flat_parameters), std::move(counts), std::move(indices));
    }

private:
    std::size_t number_ranks;
    std::vector<std::vector<IdentifierType>> nodes_per_rank;
    std::vector<std::vector<QParameter>> parameters_per_rank;
    Indices<IdentifierType> indices;
    bool finalized = false;
};

/**
 * @brief The immutable receive side of a question exchange: the questions all other ranks asked the
 *      local rank, tightly packed in source-rank order. Iterating yields one QuestionView per question,
 *      ordered by source rank and, within a rank, by the order in which the questioner added them.
 */
template <MPICompatible QParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
class IncomingQuestions {
public:
    /**
     * @brief One received question: the rank it came from, the local node it addresses, and its payload
     */
    struct QuestionView {
        MPIRank source_rank;
        IdentifierType target_node;
        const QParameter& parameter;
    };

    class ConstIterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = QuestionView;
        using difference_type = std::ptrdiff_t;

        ConstIterator() = default;

        [[nodiscard]] QuestionView operator*() const {
            return QuestionView{ MPIRank{ utility::safe_cast<int>(rank_index) },
                                 questions->target_nodes[question_index],
                                 questions->question_parameters[question_index] };
        }

        ConstIterator& operator++() {
            ++question_index;
            advance_past_empty_ranks();
            return *this;
        }

        ConstIterator operator++(int) {
            auto copy = *this;
            ++(*this);
            return copy;
        }

        [[nodiscard]] friend bool operator==(const ConstIterator& first, const ConstIterator& second) noexcept {
            return first.question_index == second.question_index;
        }

    private:
        friend class IncomingQuestions;

        ConstIterator(const IncomingQuestions* incoming_questions, const std::size_t start_index)
            : questions(incoming_questions)
            , question_index(start_index) {
            advance_past_empty_ranks();
        }

        void advance_past_empty_ranks() noexcept {
            while (rank_index < questions->number_questions_per_rank.size() && question_index >= end_of_rank_block(rank_index)) {
                ++rank_index;
            }
        }

        [[nodiscard]] std::size_t end_of_rank_block(const std::size_t rank) const noexcept {
            return static_cast<std::size_t>(questions->displacements[rank]) + static_cast<std::size_t>(questions->number_questions_per_rank[rank]);
        }

        const IncomingQuestions* questions = nullptr;
        std::size_t question_index = 0;
        std::size_t rank_index = 0;
    };

    /**
     * @brief Assembles the receive side from the received buffers. Usually not called directly;
     *      use exchange_questions instead.
     * @param target_nodes_list All received addressed nodes, tightly packed in source-rank order
     * @param question_parameters_list The received question parameters, corresponding to target_nodes_list
     * @param counts The number of questions received from each rank
     * @param displacements_list Where the questions of each rank start in the flat buffers
     * @exception Throws an Exception if the buffer sizes do not match, if the number of counts does not
     *      match the number of displacements, or if any (displacement, count) pair references elements
     *      outside the buffers
     */
    IncomingQuestions(std::vector<IdentifierType> target_nodes_list, std::vector<QParameter> question_parameters_list,
                      std::vector<int> counts, std::vector<int> displacements_list)
        : target_nodes(std::move(target_nodes_list))
        , question_parameters(std::move(question_parameters_list))
        , number_questions_per_rank(std::move(counts))
        , displacements(std::move(displacements_list)) {
        utility::Exception::check(target_nodes.size() == question_parameters.size(),
                                  "IncomingQuestions::IncomingQuestions: The number of target nodes {} does not match the number of question parameters {}",
                                  target_nodes.size(), question_parameters.size());
        utility::Exception::check(number_questions_per_rank.size() == displacements.size(),
                                  "IncomingQuestions::IncomingQuestions: The number of counts {} does not match the number of displacements {}",
                                  number_questions_per_rank.size(), displacements.size());

        for (auto rank = std::size_t{ 0 }; rank < number_questions_per_rank.size(); rank++) {
            const auto count = number_questions_per_rank[rank];
            utility::Exception::check(count >= 0, "IncomingQuestions::IncomingQuestions: The count {} of rank {} is negative", count, rank);
            const auto displacement = displacements[rank];
            utility::Exception::check(displacement >= 0, "IncomingQuestions::IncomingQuestions: The displacement {} of rank {} is negative", displacement, rank);

            if (count == 0) {
                continue;
            }

            utility::Exception::check(utility::safe_cast<std::size_t>(displacement) + utility::safe_cast<std::size_t>(count) <= target_nodes.size(),
                                      "IncomingQuestions::IncomingQuestions: Rank {} references the elements [{}, {}), but there are only {}",
                                      rank, displacement, displacement + count, target_nodes.size());
        }
    }

    /**
     * @brief Returns the total number of received questions
     * @return The number of questions
     */
    [[nodiscard]] std::size_t get_number_questions() const noexcept {
        return target_nodes.size();
    }

    /**
     * @brief Returns the number of questions received from each rank
     * @return The number of questions per source rank
     */
    [[nodiscard]] std::span<const int> get_number_questions_per_rank() const noexcept {
        return number_questions_per_rank;
    }

    /**
     * @brief Returns the local nodes addressed by the questions of a specific source rank
     * @param rank The source rank
     * @exception Throws an Exception if rank is too large
     * @return The addressed nodes
     */
    [[nodiscard]] std::span<const IdentifierType> get_target_nodes_for_rank(const MPIRank rank) const {
        const auto index = rank.get_rank_cast();
        utility::Exception::check(index < number_questions_per_rank.size(),
                                  "IncomingQuestions::get_target_nodes_for_rank: {} is larger than or equal to {}", index, number_questions_per_rank.size());

        return std::span<const IdentifierType>{ target_nodes }.subspan(utility::safe_cast<std::size_t>(displacements[index]),
                                                                       utility::safe_cast<std::size_t>(number_questions_per_rank[index]));
    }

    /**
     * @brief Returns the question parameters of a specific source rank
     * @param rank The source rank
     * @exception Throws an Exception if rank is too large
     * @return The question parameters
     */
    [[nodiscard]] std::span<const QParameter> get_question_parameters_for_rank(const MPIRank rank) const {
        const auto index = rank.get_rank_cast();
        utility::Exception::check(index < number_questions_per_rank.size(),
                                  "IncomingQuestions::get_question_parameters_for_rank: {} is larger than or equal to {}", index, number_questions_per_rank.size());

        return std::span<const QParameter>{ question_parameters }.subspan(utility::safe_cast<std::size_t>(displacements[index]),
                                                                          utility::safe_cast<std::size_t>(number_questions_per_rank[index]));
    }

    [[nodiscard]] ConstIterator begin() const noexcept {
        return ConstIterator{ this, 0 };
    }

    [[nodiscard]] ConstIterator end() const noexcept {
        return ConstIterator{ this, target_nodes.size() };
    }

private:
    std::vector<IdentifierType> target_nodes;
    std::vector<QParameter> question_parameters;
    std::vector<int> number_questions_per_rank;
    std::vector<int> displacements;
};

/**
 * @brief Exchanges the questions between all ranks. This is a collective operation: every rank must
 *      call this function, even if it has no questions to ask.
 * @tparam QParameter The type of the question payload
 * @tparam IdentifierType The type of the local node ids
 * @param questions The finalized local questions
 * @exception Throws an Exception if mpi returns an error code
 * @return The questions the other ranks asked the local rank
 */
template <MPICompatible QParameter, typename IdentifierType>
    requires std::unsigned_integral<IdentifierType>
[[nodiscard]] IncomingQuestions<QParameter, IdentifierType> exchange_questions(const OutgoingQuestions<QParameter, IdentifierType>& questions) {
    const auto send_counts = questions.get_number_questions_per_rank();
    const auto send_displacements = questions.get_displacements();

    auto receive_counts = MPICollectives::all_to_all(send_counts);
    auto receive_displacements = utility::calculate_displacements<int>(receive_counts);

    auto target_nodes = MPICollectives::all_to_all_v<IdentifierType>(questions.get_target_nodes().data(), send_counts.data(), send_displacements.data(),
                                                                     receive_counts.data(), receive_displacements.data());
    auto question_parameters = MPICollectives::all_to_all_v<QParameter>(questions.get_question_parameters().data(), send_counts.data(), send_displacements.data(),
                                                                        receive_counts.data(), receive_displacements.data());

    return IncomingQuestions<QParameter, IdentifierType>(std::move(target_nodes), std::move(question_parameters),
                                                         std::move(receive_counts), std::move(receive_displacements));
}

} // namespace comm_patterns_2

} // namespace mpiPP
