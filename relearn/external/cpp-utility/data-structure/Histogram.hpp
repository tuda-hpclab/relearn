#pragma once

/*
 * This file is part of the CPP-Utility software developed at Technical University Darmstadt
 *
 * Copyright (c) 2024-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "cpp-utility/Cast.hpp"
#include "cpp-utility/Exception.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ostream>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace utility {

/**
 * @brief A histogram with fixed-width bins that grows on demand.
 *      The histogram is represented by the lower borders of its bins and by the counts of the data points
 *      per bin. Data points are added one at a time; bin i covers the interval [i * bin_width, (i + 1) * bin_width).
 * @tparam DataType The data type of the data points, must be arithmetic
 */
template <typename DataType>
    requires std::is_arithmetic_v<DataType>
class FixedWidthHistogram {
public:
    /**
     * @brief Constructs a new empty histogram with the given bin width
     * @param bin_width The bin width, must be positive
     * @exception Throws an Exception if bin_width is not positive
     */
    explicit FixedWidthHistogram(const DataType bin_width)
        : bin_width_(bin_width) {
        Exception::check(bin_width > 0, "FixedWidthHistogram: bin_width must be positive, was {}", bin_width);
    }

    /**
     * @brief Adds a data point to the histogram and enlarges the histogram if necessary
     * @param data_point The data point to add, must be non-negative
     * @exception Throws an Exception if data_point is negative
     */
    void add_data_point(const DataType data_point) {
        Exception::check(data_point >= 0, "FixedWidthHistogram::add_data_point: data_point must be non-negative, was {}", data_point);

        const auto bin = static_cast<std::size_t>(data_point / bin_width_);
        ensure_large_enough(bin);
        ++counts[bin];
    }

    /**
     * @brief Returns the bin width of this histogram
     * @return The bin width
     */
    [[nodiscard]] DataType get_bin_width() const noexcept {
        return bin_width_;
    }

    /**
     * @brief Returns a span of the lower bin borders of the histogram.
     *      The size is the same as the counts, so the last upper border is implicit
     * @return The borders
     */
    [[nodiscard]] std::span<const DataType> get_borders() const noexcept {
        return borders;
    }

    /**
     * @brief Returns a span of the counts of the histogram.
     *      get_counts()[i] is the number of data points in the interval [get_borders()[i], get_borders()[i + 1])
     * @return The counts
     */
    [[nodiscard]] std::span<const std::size_t> get_counts() const noexcept {
        return counts;
    }

    /**
     * @brief Extracts the borders vector
     * @return The borders
     */
    [[nodiscard]] std::vector<DataType>&& get_borders_vector() && noexcept {
        return std::move(borders);
    }

    /**
     * @brief Extracts the counts vector
     * @return The counts
     */
    [[nodiscard]] std::vector<std::size_t>&& get_counts_vector() && noexcept {
        return std::move(counts);
    }

private:
    void ensure_large_enough(const std::size_t index) {
        const auto old_size = borders.size();
        if (index < old_size) {
            return;
        }

        borders.resize(index + 1);
        counts.resize(index + 1);

        for (auto i = old_size; i < index + 1; ++i) {
            borders[i] = static_cast<DataType>(i) * bin_width_;
        }
    }

    std::vector<DataType> borders{};
    std::vector<std::size_t> counts{};

    DataType bin_width_;
};

/**
 * @brief A histogram with a fixed number of equally sized bins over the half-open interval [minimum, maximum).
 *      The histogram is represented by the lower borders of its bins and by the counts of the data points
 *      per bin. Data points are added one at a time.
 * @tparam DataType The data type of the data points, must be arithmetic
 */
template <typename DataType>
    requires std::is_arithmetic_v<DataType>
class FixedSizeHistogram {
public:
    /**
     * @brief Constructs an empty histogram with a fixed number of bins
     * @param minimum The (inclusive) minimum value of the histogram
     * @param maximum The (exclusive) upper bound of the histogram, must be larger than minimum
     * @param number_bins The number of bins in the histogram, must be positive
     * @exception Throws an Exception if maximum is not larger than minimum, if number_bins is 0, or if DataType
     *      is an integer type and maximum - minimum is not divisible by number_bins
     */
    FixedSizeHistogram(const DataType minimum, const DataType maximum, const std::size_t number_bins)
        : minimum_(minimum)
        , maximum_(maximum) {
        Exception::check(minimum < maximum, "FixedSizeHistogram: maximum ({}) must be greater than minimum ({})", maximum, minimum);
        Exception::check(number_bins > 0, "FixedSizeHistogram: number_bins must be positive");

        const auto distance = maximum - minimum;
        bin_width_ = distance / static_cast<DataType>(number_bins);

        if constexpr (std::is_integral_v<DataType>) {
            const auto remainder = safe_cast<std::size_t>(distance) % number_bins;
            Exception::check(remainder == 0, "FixedSizeHistogram: maximum - minimum ({}) must be divisible by number_bins ({})", distance, number_bins);
        }

        counts.resize(number_bins, std::size_t{ 0 });
        borders.reserve(number_bins);
        for (auto i = std::size_t{ 0 }; i < number_bins; ++i) {
            borders.emplace_back(minimum + static_cast<DataType>(i) * bin_width_);
        }
    }

    /**
     * @brief Adds a data point into the histogram
     * @param data_point The data point, must be in [minimum, maximum)
     * @exception Throws an Exception if data_point is not in [minimum, maximum)
     */
    void add_data_point(const DataType data_point) {
        Exception::check(data_point >= minimum_, "FixedSizeHistogram::add_data_point: data_point ({}) must be in the interval [{}, {})", data_point, minimum_, maximum_);
        Exception::check(data_point < maximum_, "FixedSizeHistogram::add_data_point: data_point ({}) must be in the interval [{}, {})", data_point, minimum_, maximum_);

        auto index = static_cast<std::size_t>((data_point - minimum_) / bin_width_);
        // Floating-point rounding can push a data point just below maximum into a non-existent bin.
        if (index >= counts.size()) {
            index = counts.size() - 1;
        }

        ++counts[index];
    }

    /**
     * @brief Returns the bin width of this histogram
     * @return The bin width
     */
    [[nodiscard]] DataType get_bin_width() const noexcept {
        return bin_width_;
    }

    /**
     * @brief Returns a span of the lower bin borders of the histogram.
     *      The size is the same as the counts, so the last upper border is implicit
     * @return The borders
     */
    [[nodiscard]] std::span<const DataType> get_borders() const noexcept {
        return borders;
    }

    /**
     * @brief Returns a span of the counts of the histogram.
     *      get_counts()[i] is the number of data points in the interval [get_borders()[i], get_borders()[i + 1])
     * @return The counts
     */
    [[nodiscard]] std::span<const std::size_t> get_counts() const noexcept {
        return counts;
    }

    /**
     * @brief Extracts the borders vector
     * @return The borders
     */
    [[nodiscard]] std::vector<DataType>&& get_borders_vector() && noexcept {
        return std::move(borders);
    }

    /**
     * @brief Extracts the counts vector
     * @return The counts
     */
    [[nodiscard]] std::vector<std::size_t>&& get_counts_vector() && noexcept {
        return std::move(counts);
    }

private:
    std::vector<DataType> borders{};
    std::vector<std::size_t> counts{};

    DataType bin_width_{};

    DataType minimum_{};
    DataType maximum_{};
};

/**
 * @brief A general histogram represented by the lower borders of its bins and the counts per bin.
 *      It can be constructed from a FixedWidthHistogram or a FixedSizeHistogram and supports summing the
 *      counts of several histograms that share the same binning.
 *
 * Summing replaces what used to be a distributed reduction: instead of combining the partial histograms of
 * several processes, several histograms that live in the same process (e.g. one per thread or data chunk) are
 * combined by adding their counts component-wise. Two histograms can be summed if their bin borders coincide
 * where they overlap; a shorter histogram is treated as if its missing upper bins had a count of zero.
 *
 * @tparam DataType The data type of the data points, must be arithmetic
 */
template <typename DataType>
    requires std::is_arithmetic_v<DataType>
class Histogram {
public:
    /**
     * @brief Constructs an empty histogram without any bins. Acts as the neutral element of summing.
     */
    Histogram() = default;

    /**
     * @brief Constructs a histogram from a FixedWidthHistogram
     * @param histogram The FixedWidthHistogram to move from
     */
    Histogram(FixedWidthHistogram<DataType>&& histogram)
        : borders(std::move(histogram).get_borders_vector())
        , counts(std::move(histogram).get_counts_vector()) {
    }

    /**
     * @brief Constructs a histogram from a FixedSizeHistogram
     * @param histogram The FixedSizeHistogram to move from
     */
    Histogram(FixedSizeHistogram<DataType>&& histogram)
        : borders(std::move(histogram).get_borders_vector())
        , counts(std::move(histogram).get_counts_vector()) {
    }

    /**
     * @brief Returns a span of the lower bin borders of the histogram.
     *      The size is the same as the counts, so the last upper border is implicit
     * @return The borders
     */
    [[nodiscard]] std::span<const DataType> get_borders() const noexcept {
        return borders;
    }

    /**
     * @brief Returns a span of the counts of the histogram.
     *      get_counts()[i] is the number of data points in the interval [get_borders()[i], get_borders()[i + 1])
     * @return The counts
     */
    [[nodiscard]] std::span<const std::size_t> get_counts() const noexcept {
        return counts;
    }

    /**
     * @brief Returns the number of bins of the histogram
     * @return The number of bins
     */
    [[nodiscard]] std::size_t num_bins() const noexcept {
        return counts.size();
    }

    /**
     * @brief Adds the counts of another histogram to this one, bin by bin.
     *      If the other histogram has more bins, this histogram grows to match and adopts the additional borders.
     *      If it has fewer bins, its missing upper bins contribute a count of zero.
     * @param other The histogram whose counts are added
     * @exception Throws an Exception if the two histograms have different borders where their bins overlap
     * @return A reference to this histogram
     */
    Histogram& operator+=(const Histogram& other) {
        const auto this_size = counts.size();
        const auto other_size = other.counts.size();
        const auto common_size = std::min(this_size, other_size);

        for (auto i = std::size_t{ 0 }; i < common_size; ++i) {
            Exception::check(borders[i] == other.borders[i],
                "Histogram::operator+=: The histograms have different borders at bin {} ({} vs. {}) and cannot be summed", i, borders[i], other.borders[i]);
        }

        if (other_size > this_size) {
            borders.resize(other_size);
            counts.resize(other_size, std::size_t{ 0 });
            for (auto i = this_size; i < other_size; ++i) {
                borders[i] = other.borders[i];
            }
        }

        for (auto i = std::size_t{ 0 }; i < other_size; ++i) {
            counts[i] += other.counts[i];
        }

        return *this;
    }

    /**
     * @brief Returns the bin-by-bin sum of two histograms without modifying either operand
     * @param lhs The first histogram (taken by value and reused as the result)
     * @param rhs The second histogram
     * @exception Throws an Exception if the two histograms have different borders where their bins overlap
     * @return The summed histogram
     */
    [[nodiscard]] friend Histogram operator+(Histogram lhs, const Histogram& rhs) {
        lhs += rhs;
        return lhs;
    }

    /**
     * @brief Sums an arbitrary number of histograms bin by bin. An empty range yields an empty histogram.
     * @tparam Range An input range whose elements are Histogram
     * @param histograms The histograms to sum
     * @exception Throws an Exception if any two of the histograms have different borders where their bins overlap
     * @return The summed histogram
     */
    template <std::ranges::input_range Range>
        requires std::same_as<std::ranges::range_value_t<Range>, Histogram>
    [[nodiscard]] static Histogram sum(Range&& histograms) {
        auto result = Histogram{};
        for (const auto& histogram : histograms) {
            result += histogram;
        }
        return result;
    }

    /**
     * @brief Prints the histogram to the ostream, one line per bin in the format "i. bin: start-end: count"
     * @param stream The stream to which the object should be printed
     * @param histogram The histogram that should be printed
     * @return A reference to stream that allows chaining.
     *      Is not marked as [[nodiscard]] as that typically does happen when chaining <<
     */
    friend std::ostream& operator<<(std::ostream& stream, const Histogram& histogram) {
        for (auto i = std::size_t{ 0 }; i < histogram.borders.size(); ++i) {
            const auto start = histogram.borders[i];
            const auto end = i + 1 < histogram.borders.size() ? histogram.borders[i + 1] : std::numeric_limits<DataType>::max();
            const auto count = histogram.counts[i];

            stream << i << ". bin: " << start << '-' << end << ": " << count << '\n';
        }

        return stream;
    }

private:
    std::vector<DataType> borders{};
    std::vector<std::size_t> counts{};
};

} // namespace utility
