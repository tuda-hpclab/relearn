#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "util/NeuronID.h"

#include <cpp-utility/data-structure/TaggedIDRange.hpp>
#include <cpp-utility/ranges/Functional.hpp>

#include <range/v3/view/transform.hpp>

/**
 * @brief The range factories for NeuronID, provided as static methods.
 *
 * They live in this separate header so that NeuronID.h itself stays free of range-v3: only translation units
 * that actually iterate over neuron ids pay for those includes. This mirrors utility::TaggedIDRange, to which
 * the bounds checking and the construction of the ids are delegated.
 *
 * The ids are constructed via the value constructor of NeuronID and are therefore initialized and not virtual.
 *
 * Example: NeuronIDRange::range(3)         the ids with the values 0, 1, 2
 *      NeuronIDRange::range(2, 5)          the ids with the values 2, 3, 4
 *      NeuronIDRange::range_id(3)          the plain values 0, 1, 2
 */
class NeuronIDRange {
public:
    using id_type = NeuronID;
    using value_type = NeuronID::value_type;

    static constexpr value_type min_value = NeuronID::limits::min;
    static constexpr value_type max_value = NeuronID::limits::max;

    /**
     * @brief Create a range of NeuronIDs within the range [begin, end)
     *
     * @param begin begin of the range
     * @param end end of the range
     * @exception RelearnException if begin > end or if end > max_value + 1
     * @return range of NeuronIDs
     */
    [[nodiscard]] static auto range(const value_type begin, const value_type end) {
        return utility::TaggedIDRange<NeuronIDData>::range(begin, end) | ranges::views::transform(utility::construct<NeuronID>);
    }

    /**
     * @brief Create a range of NeuronIDs within the range [begin, end)
     *
     * @param begin begin of the range
     * @param end end of the range
     * @exception RelearnException if one of the ids is not initialized, if one of them is virtual, or if begin > end
     * @return range of NeuronIDs
     */
    [[nodiscard]] static auto range(const NeuronID begin, const NeuronID end) {
        return range(begin.get_neuron_id(), end.get_neuron_id());
    }

    /**
     * @brief Create a range of local NeuronIDs within the range [0, size)
     *
     * @param size size of the range
     * @exception RelearnException if size > max_value + 1
     * @return range of NeuronIDs
     */
    [[nodiscard]] static auto range(const value_type size) {
        return range(min_value, size);
    }

    /**
     * @brief Create a range of local NeuronIDs within the range [0, size)
     *
     * @param size size of the range
     * @exception RelearnException if size is not initialized or if it is virtual
     * @return range of NeuronIDs
     */
    [[nodiscard]] static auto range(const NeuronID size) {
        return range(min_value, size.get_neuron_id());
    }

    /**
     * @brief Create a range of NeuronIDs within the range [begin, end) but as ids of type value_type
     *
     * @param begin begin of the range
     * @param end end of the range
     * @exception RelearnException if begin > end or if end > max_value + 1
     * @return range of NeuronIDs of type value_type
     */
    [[nodiscard]] static auto range_id(const value_type begin, const value_type end) {
        return utility::TaggedIDRange<NeuronIDData>::range_values(begin, end);
    }

    /**
     * @brief Create a range of local NeuronIDs within the range [0, size) but as ids of type value_type
     *
     * @param size size of the range
     * @exception RelearnException if size > max_value + 1
     * @return range of NeuronIDs of type value_type
     */
    [[nodiscard]] static auto range_id(const value_type size) {
        return range_id(min_value, size);
    }
};
