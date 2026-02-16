#pragma once

#include "Types.h"
#include "Types2.h"

class CalculatorUtil {
public:
    /**
     * Creates a function that takes a neuron id and returns a constant every time regardless of the neuron id
     * @param constant The constant that the function returns
     * @return Function that takes a neuron id and returns the given constant
     */
    static RelearnTypes::neuron_id_to_calcium_calculator construct_constant_calculator(const double constant) {
        return [constant](const RelearnTypes::number_neurons_type /*neuron_id*/) {
            return constant;
        };
    };
};
