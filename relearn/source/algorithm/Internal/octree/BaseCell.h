#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "algorithm/VirtualPlasticityElement.h"
#include "neurons/enums/SynapticElementType.h"
#include "util/RelearnException.h"

#include <optional>
#include <ostream>

#ifdef WIN32
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define RELEARN_NUA [[msvc::no_unique_address]]
#endif

#ifndef WIN32
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define RELEARN_NUA [[no_unique_address]]
#endif

/**
 * This class encapsulates the logic for the additional cell attributes in the octree.
 * Different algorithms needs different synaptic elements in the cell, so this class
 * offers a way to enable or disable certain ones.
 * @tparam has_excitatory_dendrite_ Determines if the cell will have excitatory dendrites
 * @tparam has_inhibitory_dendrite_ Determines if the cell will have inhibitory dendrites
 * @tparam has_excitatory_axon_ Determines if the cell will have excitatory axons
 * @tparam has_inhibitory_axon_ Determines if the cell will have inhibitory axons
 */
template <bool has_excitatory_dendrite_, bool has_inhibitory_dendrite_, bool has_excitatory_axon_, bool has_inhibitory_axon_>
class BaseCell {
public:
    using counter_type = VirtualPlasticityElement::counter_type;
    using position_type = VirtualPlasticityElement::position_type;

    constexpr static bool has_excitatory_dendrite = has_excitatory_dendrite_;
    constexpr static bool has_inhibitory_dendrite = has_inhibitory_dendrite_;
    constexpr static bool has_excitatory_axon = has_excitatory_axon_;
    constexpr static bool has_inhibitory_axon = has_inhibitory_axon_;

    /**
     * @brief Sets the number of free excitatory dendrites in this cell
     * @param num_dendrites The number of free excitatory dendrites
     */
    constexpr void set_number_excitatory_dendrites(const counter_type num_dendrites) noexcept
        requires(has_excitatory_dendrite)
    {
        excitatory_dendrite.set_number_free_elements(num_dendrites);
    }

    /**
     * @brief Returns the number of free excitatory dendrites in this cell
     * @return The number of free excitatory dendrites
     */
    [[nodiscard]] constexpr counter_type get_number_excitatory_dendrites() const noexcept
        requires(has_excitatory_dendrite)
    {
        return excitatory_dendrite.get_number_free_elements();
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param virtual_position The new position of the excitatory dendrites
     */
    constexpr void set_excitatory_dendrites_position(const std::optional<position_type>& virtual_position) noexcept
        requires(has_excitatory_dendrite)
    {
        excitatory_dendrite.set_position(virtual_position);
    }

    /**
     * @brief Returns the position of the excitatory dendrite
     * @return The position of the excitatory dendrite
     */
    [[nodiscard]] constexpr std::optional<position_type> get_excitatory_dendrites_position() const noexcept
        requires(has_excitatory_dendrite)
    {
        return excitatory_dendrite.get_position();
    }

    /**
     * @brief Sets the number of free inhibitory dendrites in this cell
     * @param num_dendrites The number of free inhibitory dendrites
     */
    constexpr void set_number_inhibitory_dendrites(const counter_type num_dendrites) noexcept
        requires(has_inhibitory_dendrite)
    {
        inhibitory_dendrite.set_number_free_elements(num_dendrites);
    }

    /**
     * @brief Returns the number of free inhibitory dendrites in this cell
     * @return The number of free inhibitory dendrites
     */
    [[nodiscard]] constexpr counter_type get_number_inhibitory_dendrites() const noexcept
        requires(has_inhibitory_dendrite)
    {
        return inhibitory_dendrite.get_number_free_elements();
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param virtual_position The new position of the inhibitory dendrites
     */
    constexpr void set_inhibitory_dendrites_position(const std::optional<position_type>& virtual_position) noexcept
        requires(has_inhibitory_dendrite)
    {
        inhibitory_dendrite.set_position(virtual_position);
    }

    /**
     * @brief Returns the position of the inhibitory dendrite
     * @return The position of the inhibitory dendrite
     */
    [[nodiscard]] constexpr std::optional<position_type> get_inhibitory_dendrites_position() const noexcept
        requires(has_inhibitory_dendrite)
    {
        return inhibitory_dendrite.get_position();
    }

    /**
     * @brief Sets the number of free dendrites for the associated type in this cell
     * @param dendrite_type The requested dendrite type
     * @param num_dendrites The number of free dendrites
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_number_dendrites_for(const SignalType dendrite_type, const counter_type num_dendrites)
        requires(has_excitatory_dendrite || has_inhibitory_dendrite)
    {
        if (dendrite_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_dendrite) {
                excitatory_dendrite.set_number_free_elements(num_dendrites);
                return;
            }
        }

        if constexpr (has_inhibitory_dendrite) {
            inhibitory_dendrite.set_number_free_elements(num_dendrites);
            return;
        }

        RelearnException::fail("BaseCell::set_number_dendrites_for(): dendrite_type {} is not present in the cell!", dendrite_type);
    }

    /**
     * @brief Returns the number of free dendrites for the associated type in this cell
     * @param dendrite_type The requested dendrite type
     * @exception Throws a RelearnException if the requested type is not in this cell
     * @return The number of free dendrites for the associated type
     */
    [[nodiscard]] constexpr counter_type get_number_dendrites_for(const SignalType dendrite_type) const
        requires(has_excitatory_dendrite || has_inhibitory_dendrite)
    {
        if (dendrite_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_dendrite) {
                return excitatory_dendrite.get_number_free_elements();
            }
        }

        if constexpr (has_inhibitory_dendrite) {
            return inhibitory_dendrite.get_number_free_elements();
        }

        RelearnException::fail("BaseCell::get_number_dendrites_for(): dendrite_type {} is not present in the cell!", dendrite_type);
    }

    /**
     * @brief Sets the position of the free dendrites for the associated type in this cell
     * @param dendrite_type The requested dendrite type
     * @param virtual_position The position of the free dendrites
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_dendrites_position_for(const SignalType dendrite_type, const std::optional<position_type>& virtual_position)
        requires(has_excitatory_dendrite || has_inhibitory_dendrite)
    {
        if (dendrite_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_dendrite) {
                excitatory_dendrite.set_position(virtual_position);
                return;
            }
        }

        if constexpr (has_inhibitory_dendrite) {
            inhibitory_dendrite.set_position(virtual_position);
            return;
        }

        RelearnException::fail("BaseCell::set_dendrites_position_for(): dendrite_type {} is not present in the cell!", dendrite_type);
    }

    /**
     * @brief Returns the position of the free dendrites for the associated type in this cell
     * @param dendrite_type The requested dendrite type
     * @exception Throws a RelearnException if the requested type is not in this cell
     * @return The position of the free dendrites for the associated type
     */
    [[nodiscard]] constexpr std::optional<position_type> get_dendrites_position_for(const SignalType dendrite_type) const
        requires(has_excitatory_dendrite || has_inhibitory_dendrite)
    {
        if (dendrite_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_dendrite) {
                return excitatory_dendrite.get_position();
            }
        }

        if constexpr (has_inhibitory_dendrite) {
            return inhibitory_dendrite.get_position();
        }

        RelearnException::fail("BaseCell::get_dendrites_position_for(): dendrite_type {} is not present in the cell!", dendrite_type);
    }

    /**
     * @brief Sets the number of free excitatory axons in this cell
     * @param num_axons The number of free excitatory axons
     */
    constexpr void set_number_excitatory_axons(const counter_type num_axons) noexcept
        requires(has_excitatory_axon)
    {
        excitatory_axon.set_number_free_elements(num_axons);
    }

    /**
     * @brief Returns the number of free excitatory axons in this cell
     * @return The number of free excitatory axons
     */
    [[nodiscard]] constexpr counter_type get_number_excitatory_axons() const noexcept
        requires(has_excitatory_axon)
    {
        return excitatory_axon.get_number_free_elements();
    }

    /**
     * @brief Sets the position of the excitatory position, which can be empty
     * @param virtual_position The new position of the excitatory axons
     */
    constexpr void set_excitatory_axons_position(const std::optional<position_type>& virtual_position) noexcept
        requires(has_excitatory_axon)
    {
        excitatory_axon.set_position(virtual_position);
    }

    /**
     * @brief Returns the position of the excitatory axon
     * @return The position of the excitatory axon
     */
    [[nodiscard]] constexpr std::optional<position_type> get_excitatory_axons_position() const noexcept
        requires(has_excitatory_axon)
    {
        return excitatory_axon.get_position();
    }

    /**
     * @brief Sets the number of free inhibitory axons in this cell
     * @param num_axons The number of free inhibitory axons
     */
    constexpr void set_number_inhibitory_axons(const counter_type num_axons) noexcept
        requires(has_inhibitory_axon)
    {
        inhibitory_axon.set_number_free_elements(num_axons);
    }

    /**
     * @brief Returns the number of free inhibitory axons in this cell
     * @return The number of free inhibitory axons
     */
    [[nodiscard]] constexpr counter_type get_number_inhibitory_axons() const noexcept
        requires(has_inhibitory_axon)
    {
        return inhibitory_axon.get_number_free_elements();
    }

    /**
     * @brief Sets the position of the inhibitory position, which can be empty
     * @param virtual_position The new position of the inhibitory axons
     */
    constexpr void set_inhibitory_axons_position(const std::optional<position_type>& virtual_position) noexcept
        requires(has_inhibitory_axon)
    {
        inhibitory_axon.set_position(virtual_position);
    }

    /**
     * @brief Returns the position of the inhibitory axon
     * @return The position of the inhibitory axon
     */
    [[nodiscard]] constexpr std::optional<position_type> get_inhibitory_axons_position() const noexcept
        requires(has_inhibitory_axon)
    {
        return inhibitory_axon.get_position();
    }

    /**
     * @brief Sets the number of free axons for the associated type in this cell
     * @param axon_type The requested axon type
     * @param num_axons The number of free axons
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_number_axons_for(const SignalType axon_type, const counter_type num_axons)
        requires(has_excitatory_axon || has_inhibitory_axon)
    {
        if (axon_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_axon) {
                excitatory_axon.set_number_free_elements(num_axons);
                return;
            }
        }

        if constexpr (has_inhibitory_axon) {
            inhibitory_axon.set_number_free_elements(num_axons);
            return;
        }

        RelearnException::fail("BaseCell::set_number_axons_for(): axon_type {} is not present in the cell!", axon_type);
    }

    /**
     * @brief Returns the number of free axons for the associated type in this cell
     * @param axon_type The requested axon type
     * @exception Throws a RelearnException if the requested type is not in this cell
     * @return The number of free axons for the associated type
     */
    [[nodiscard]] constexpr counter_type get_number_axons_for(const SignalType axon_type) const
        requires(has_excitatory_axon || has_inhibitory_axon)
    {
        if (axon_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_axon) {
                return excitatory_axon.get_number_free_elements();
            }
        }

        if constexpr (has_inhibitory_axon) {
            return inhibitory_axon.get_number_free_elements();
        }

        RelearnException::fail("BaseCell::get_number_axons_for(): axon_type {} is not present in the cell!", axon_type);
    }

    /**
     * @brief Sets the position of the free axons for the associated type in this cell
     * @param axon_type The requested axon type
     * @param virtual_position The position of the free axons
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_axons_position_for(const SignalType axon_type, const std::optional<position_type>& virtual_position)
        requires(has_excitatory_axon || has_inhibitory_axon)
    {
        if (axon_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_axon) {
                excitatory_axon.set_position(virtual_position);
                return;
            }
        }

        if constexpr (has_inhibitory_axon) {
            inhibitory_axon.set_position(virtual_position);
            return;
        }

        RelearnException::fail("BaseCell::set_axons_position_for(): axon_type {} is not present in the cell!", axon_type);
    }

    /**
     * @brief Returns the position of the free axons for the associated type in this cell
     * @param axon_type The requested axon type
     * @exception Throws a RelearnException if the requested type is not in this cell
     * @return The position of the free axons for the associated type
     */
    [[nodiscard]] constexpr std::optional<position_type> get_axons_position_for(const SignalType axon_type) const
        requires(has_excitatory_axon || has_inhibitory_axon)
    {
        if (axon_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_axon) {
                return excitatory_axon.get_position();
            }
        }

        if constexpr (has_inhibitory_axon) {
            return inhibitory_axon.get_position();
        }

        RelearnException::fail("BaseCell::get_axons_position_for(): axon_type {} is not present in the cell!", axon_type);
    }

    /**
     * @brief Sets the number of free elements for the associated type in this cell
     * @param element_type The requested elements' type
     * @param signal_type The requested elements' signal type
     * @param num_elements The number of free elements
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_number_elements_for(const ElementType element_type, const SignalType signal_type, const counter_type num_elements)
        requires(has_excitatory_dendrite || has_inhibitory_dendrite || has_excitatory_axon || has_inhibitory_axon)
    {
        if (element_type == ElementType::Axon && signal_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_axon) {
                excitatory_axon.set_number_free_elements(num_elements);
                return;
            }
        }

        if (element_type == ElementType::Axon && signal_type == SignalType::Inhibitory) {
            if constexpr (has_inhibitory_axon) {
                inhibitory_axon.set_number_free_elements(num_elements);
                return;
            }
        }

        if (element_type == ElementType::Dendrite && signal_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_dendrite) {
                excitatory_dendrite.set_number_free_elements(num_elements);
                return;
            }
        }

        if (element_type == ElementType::Dendrite && signal_type == SignalType::Inhibitory) {
            if constexpr (has_inhibitory_dendrite) {
                inhibitory_dendrite.set_number_free_elements(num_elements);
                return;
            }
        }

        RelearnException::fail("BaseCell::set_number_elements_for(): element_type {} with signal_type {} is not present in the cell!", element_type, signal_type);
    }

    /**
     * @brief Returns the number of free elements for the associated type in this cell
     * @param element_type The requested elements' type
     * @param signal_type The requested elements' signal type
     * @exception Throws a RelearnException if the requested type is not in this cell
     * @return The number of free elements for the associated type
     */
    [[nodiscard]] constexpr counter_type get_number_elements_for(const ElementType element_type, const SignalType signal_type) const
        requires(has_excitatory_dendrite || has_inhibitory_dendrite || has_excitatory_axon || has_inhibitory_axon)
    {
        if (element_type == ElementType::Axon && signal_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_axon) {
                return excitatory_axon.get_number_free_elements();
            }
        }

        if (element_type == ElementType::Axon && signal_type == SignalType::Inhibitory) {
            if constexpr (has_inhibitory_axon) {
                return inhibitory_axon.get_number_free_elements();
            }
        }

        if (element_type == ElementType::Dendrite && signal_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_dendrite) {
                return excitatory_dendrite.get_number_free_elements();
            }
        }

        if (element_type == ElementType::Dendrite && signal_type == SignalType::Inhibitory) {
            if constexpr (has_inhibitory_dendrite) {
                return inhibitory_dendrite.get_number_free_elements();
            }
        }

        RelearnException::fail("BaseCell::get_number_elements_for(): element_type {} with signal_type {} is not present in the cell!", element_type, signal_type);
    }

    /**
     * @brief Sets the position of the free elements for the associated type in this cell
     * @param element_type The requested elements' type
     * @param signal_type The requested elements' signal type
     * @param virtual_position The position of the free elements
     * @exception Throws a RelearnException if the requested type is not in this cell
     */
    constexpr void set_position_for(const ElementType element_type, const SignalType signal_type, const std::optional<position_type>& virtual_position)
        requires(has_excitatory_dendrite || has_inhibitory_dendrite || has_excitatory_axon || has_inhibitory_axon)
    {
        if (element_type == ElementType::Axon && signal_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_axon) {
                excitatory_axon.set_position(virtual_position);
                return;
            }
        }

        if (element_type == ElementType::Axon && signal_type == SignalType::Inhibitory) {
            if constexpr (has_inhibitory_axon) {
                inhibitory_axon.set_position(virtual_position);
                return;
            }
        }

        if (element_type == ElementType::Dendrite && signal_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_dendrite) {
                excitatory_dendrite.set_position(virtual_position);
                return;
            }
        }

        if (element_type == ElementType::Dendrite && signal_type == SignalType::Inhibitory) {
            if constexpr (has_inhibitory_dendrite) {
                inhibitory_dendrite.set_position(virtual_position);
                return;
            }
        }

        RelearnException::fail("BaseCell::set_elements_position_for(): element_type {} with signal_type {} is not present in the cell!", element_type, signal_type);
    }

    /**
     * @brief Returns the position of the free elements for the associated type in this cell
     * @param element_type The requested elements' type
     * @param signal_type The requested elements' signal type
     * @exception Throws a RelearnException if the requested type is not in this cell
     * @return The position of the free elements for the associated type
     */
    [[nodiscard]] constexpr std::optional<position_type> get_position_for(const ElementType element_type, const SignalType signal_type) const
        requires(has_excitatory_dendrite || has_inhibitory_dendrite || has_excitatory_axon || has_inhibitory_axon)
    {
        if (element_type == ElementType::Axon && signal_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_axon) {
                return excitatory_axon.get_position();
            }
        }

        if (element_type == ElementType::Axon && signal_type == SignalType::Inhibitory) {
            if constexpr (has_inhibitory_axon) {
                return inhibitory_axon.get_position();
            }
        }

        if (element_type == ElementType::Dendrite && signal_type == SignalType::Excitatory) {
            if constexpr (has_excitatory_dendrite) {
                return excitatory_dendrite.get_position();
            }
        }

        if (element_type == ElementType::Dendrite && signal_type == SignalType::Inhibitory) {
            if constexpr (has_inhibitory_dendrite) {
                return inhibitory_dendrite.get_position();
            }
        }

        RelearnException::fail("BaseCell::get_elements_position_for(): element_type {} with signal_type {} is not present in the cell!", element_type, signal_type);
    }

    /**
     * @brief Sets the position of every element in the cell
     * @param virtual_position The position, can be empty
     */
    constexpr void set_neuron_position(const std::optional<position_type>& virtual_position) noexcept {
        if constexpr (has_excitatory_dendrite) {
            excitatory_dendrite.set_position(virtual_position);
        }

        if constexpr (has_inhibitory_dendrite) {
            inhibitory_dendrite.set_position(virtual_position);
        }

        if constexpr (has_excitatory_axon) {
            excitatory_axon.set_position(virtual_position);
        }

        if constexpr (has_inhibitory_axon) {
            inhibitory_axon.set_position(virtual_position);
        }
    }

    /**
     * @brief Returns the position of all elements, which can be empty
     * @exception Throws a RelearnException if not all stored positions are equal
     * @return The position of the elements. Is empty, if no element is present.
     */
    [[nodiscard]] constexpr std::optional<position_type> get_neuron_position() const {
        std::optional<position_type> current_position{};
        bool current_position_valid = false;

        if constexpr (has_excitatory_dendrite) {
            if (!current_position_valid) {
                current_position = excitatory_dendrite.get_position();
                current_position_valid = true;
            } else {
                const auto& other_position = excitatory_dendrite.get_position();
                const auto position_matches = current_position == other_position;
                RelearnException::check(position_matches, "BaseCell::get_neuron_position(): The positions don't match.");
            }
        }

        if constexpr (has_inhibitory_dendrite) {
            if (!current_position_valid) {
                current_position = inhibitory_dendrite.get_position();
                current_position_valid = true;
            } else {
                const auto& other_position = inhibitory_dendrite.get_position();
                const auto position_matches = current_position == other_position;
                RelearnException::check(position_matches, "BaseCell::get_neuron_position(): The positions don't match.");
            }
        }

        if constexpr (has_excitatory_axon) {
            if (!current_position_valid) {
                current_position = excitatory_axon.get_position();
                current_position_valid = true;
            } else {
                const auto& other_position = excitatory_axon.get_position();
                const auto position_matches = current_position == other_position;
                RelearnException::check(position_matches, "BaseCell::get_neuron_position(): The positions don't match.");
            }
        }

        if constexpr (has_inhibitory_axon) {
            if (!current_position_valid) {
                current_position = inhibitory_axon.get_position();
                current_position_valid = true;
            } else {
                const auto& other_position = inhibitory_axon.get_position();
                const auto position_matches = current_position == other_position;
                RelearnException::check(position_matches, "BaseCell::get_neuron_position(): The positions don't match.");
            }
        }

        return current_position;
    }

    /**
     * @brief Prints the base cell to the output stream
     * @param output_stream The output stream
     * @param base_cell The base cell to print
     * @return The output stream after printing the base cell
     */
    friend std::ostream& operator<<(std::ostream& output_stream, const BaseCell<has_excitatory_dendrite, has_inhibitory_dendrite, has_excitatory_axon, has_inhibitory_axon>& base_cell) {
        // NOLINTNEXTLINE
        output_stream << "  == BaseCell (" << reinterpret_cast<std::size_t>(&base_cell) << " ==\n";

        if constexpr (has_excitatory_dendrite) {
            output_stream << "\tNumber excitatory dendrites: " << base_cell.get_number_excitatory_dendrites() << '\n';
            if (base_cell.get_excitatory_dendrites_position().has_value()) {
                output_stream << "\tPosition excitatory dendrites: " << base_cell.get_excitatory_dendrites_position().value() << '\n';
            } else {
                output_stream << "\tPosition excitatory dendrites: empty\n";
            }
        }

        if constexpr (has_inhibitory_dendrite) {
            output_stream << "\tNumber inhibitory dendrites: " << base_cell.get_number_inhibitory_dendrites() << '\n';
            if (base_cell.get_inhibitory_dendrites_position().has_value()) {
                output_stream << "\tPosition inhibitory dendrites: " << base_cell.get_inhibitory_dendrites_position().value() << '\n';
            } else {
                output_stream << "\tPosition inhibitory dendrites: empty\n";
            }
        }

        if constexpr (has_excitatory_axon) {
            output_stream << "\tNumber excitatory axons: " << base_cell.get_number_excitatory_axons() << '\n';
            if (base_cell.get_excitatory_axons_position().has_value()) {
                output_stream << "\tPosition excitatory axons: " << base_cell.get_excitatory_axons_position().value() << '\n';
            } else {
                output_stream << "\tPosition excitatory axons: empty\n";
            }
        }

        if constexpr (has_inhibitory_axon) {
            output_stream << "\tNumber inhibitory axons: " << base_cell.get_number_inhibitory_axons() << '\n';
            if (base_cell.get_inhibitory_axons_position().has_value()) {
                output_stream << "\tPosition inhibitory axons: " << base_cell.get_inhibitory_axons_position().value() << '\n';
            } else {
                output_stream << "\tPosition inhibitory axons: empty\n";
            }
        }

        return output_stream;
    }

    /**
     * @brief Compare two BaseCells
     * @return True iff both are equal
     */
    [[nodiscard]] friend constexpr bool operator==(const BaseCell& first, const BaseCell& second) noexcept {
        auto flag_1 = true;
        auto flag_2 = true;
        auto flag_3 = true;
        auto flag_4 = true;

        if constexpr (has_excitatory_dendrite) {
            flag_1 = (first.excitatory_dendrite == second.excitatory_dendrite);
        }

        if constexpr (has_inhibitory_dendrite) {
            flag_2 = (first.inhibitory_dendrite == second.inhibitory_dendrite);
        }

        if constexpr (has_excitatory_axon) {
            flag_3 = (first.excitatory_axon == second.excitatory_axon);
        }

        if constexpr (has_inhibitory_axon) {
            flag_4 = (first.inhibitory_axon == second.inhibitory_axon);
        }

        return flag_1 && flag_2 && flag_3 && flag_4;
    }

private:
    struct empty_type_1 { };
    struct empty_type_2 { };
    struct empty_type_3 { };
    struct empty_type_4 { };

    RELEARN_NUA std::conditional_t<has_excitatory_dendrite, VirtualPlasticityElement, empty_type_1> excitatory_dendrite{};
    RELEARN_NUA std::conditional_t<has_inhibitory_dendrite, VirtualPlasticityElement, empty_type_2> inhibitory_dendrite{};
    RELEARN_NUA std::conditional_t<has_excitatory_axon, VirtualPlasticityElement, empty_type_3> excitatory_axon{};
    RELEARN_NUA std::conditional_t<has_inhibitory_axon, VirtualPlasticityElement, empty_type_4> inhibitory_axon{};
};

#undef RELEARN_NUA
