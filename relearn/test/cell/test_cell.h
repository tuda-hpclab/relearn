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

#include "RelearnTest.hpp"

class CellTest : public RelearnTest {
protected:
    template <typename AdditionalCellAttributes>
    void test_cell_size();

    template <typename AdditionalCellAttributes>
    void test_cell_dendrites_position();

    template <typename AdditionalCellAttributes>
    void test_cell_dendrites_position_exception();

    template <typename AdditionalCellAttributes>
    void test_cell_set_number_dendrites();

    template <typename AdditionalCellAttributes>
    void test_cell_dendrites_position_combined();

    template <typename AdditionalCellAttributes>
    void test_cell_axons_position();

    template <typename AdditionalCellAttributes>
    void test_cell_axons_position_exception();

    template <typename AdditionalCellAttributes>
    void test_cell_set_number_axons();

    template <typename AdditionalCellAttributes>
    void test_cell_axons_position_combined();

    template <typename AdditionalCellAttributes>
    void test_cell_neuron_position();

    template <typename AdditionalCellAttributes>
    void test_cell_set_neuron_id();

    template <typename AdditionalCellAttributes>
    void test_cell_octants();

    template <typename AdditionalCellAttributes>
    void test_cell_octants_exception();

    template <typename AdditionalCellAttributes>
    void test_cell_octants_size();

    template <typename AdditionalCellAttributes>
    void test_cell_print();

    template <typename VirtualPlasticityElement>
    void test_vpe_number_elements();

    template <typename VirtualPlasticityElement>
    void test_vpe_position();

    template <typename VirtualPlasticityElement>
    void test_vpe_mixed();
};
