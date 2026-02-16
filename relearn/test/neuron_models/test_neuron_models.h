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

class NeuronModel;

class NeuronModelsTest : public RelearnTest {
protected:
    static void assert_getter_equality(const NeuronModel& model);

    static void assert_getter_throws(const NeuronModel& model);
};

class AEIFModelTest : public NeuronModelsTest {
};

class FitzHughNagumoModelTest : public NeuronModelsTest {
};

class IzhikevichModelTest : public NeuronModelsTest {
};

class PoissonModelTest : public NeuronModelsTest {
};
