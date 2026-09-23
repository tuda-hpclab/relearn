/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2022-2026, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include <mpi-wrapper/MPIWrapper.h>

#include <mpi.h>

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cerr << "Please pass arguments!\n";
        return 1;
    }

    try {
        mpiPP::MPIWrapper::init(argc, argv);
        // You are free to insert code that checks anything.
        // For example, an activity trace of a neuron with predetermined input.
        mpiPP::MPIWrapper::finalize();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "relearn_analysis: fatal error: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
