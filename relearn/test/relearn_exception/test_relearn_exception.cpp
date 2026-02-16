/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_relearn_exception.h"

#include "util/RelearnException.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include <gtest/gtest.h>

#include <iostream>

TEST_F(RelearnExceptionTest, testException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    RelearnException::hide_messages = false;

    using namespace std::string_literals;
    const auto message = "sadflhbcn\nkow97430921*:)§\" $SDMFSL "s;
    ASSERT_THROW_NO_PRINT(RelearnException::fail(message), RelearnException);

    try {
        RelearnException::fail(message);
    } catch (const RelearnException& ex) {
        const auto& reason = ex.what();
        ASSERT_EQ(message, reason);
    }

    RelearnException::hide_messages = true;
}

TEST_F(RelearnExceptionTest, testCheck) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    RelearnException::hide_messages = false;

    using namespace std::string_literals;
    const auto message = "sadflhbcn\nkow97430921*:)§\" $SDMFSL "s;

    ASSERT_NO_THROW(RelearnException::check(true, message));
    ASSERT_THROW_NO_PRINT(RelearnException::check(false, message), RelearnException);

    try {
        RelearnException::check(false, message);
    } catch (const RelearnException& ex) {
        const auto& reason = ex.what();
        ASSERT_EQ(message, reason);
    }

    RelearnException::hide_messages = true;
}

TEST_F(RelearnExceptionTest, testFormatting) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    RelearnException::hide_messages = false;

    using namespace std::string_literals;
    const auto message = "This is the first value: {} and this the second: {}\n"s;
    const auto expected_message = "This is the first value: 123456 and this the second: false\n"s;

    try {
        RelearnException::fail(message, 123456, false);
    } catch (const RelearnException& ex) {
        const auto& reason = ex.what();
        ASSERT_EQ(expected_message, reason);
    }

    RelearnException::hide_messages = true;
}

TEST_F(RelearnExceptionTest, testFormattingWrongNumberArguments) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_ANY_THROW(RelearnException::fail("{}"));
    ASSERT_ANY_THROW(RelearnException::fail("{} {}", 2));
    ASSERT_ANY_THROW(RelearnException::fail("{}", 4.2, false));
    ASSERT_ANY_THROW(RelearnException::fail("{}", "Hallo", 32));
    ASSERT_ANY_THROW(RelearnException::fail("{} {} {}"));
}
