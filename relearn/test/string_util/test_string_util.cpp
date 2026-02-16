/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_string_util.h"

#include "util/RelearnException.h"
#include "util/StringUtil.h"

#include "mpi-wrapper/MPIInfo.h"
#include "mpi-wrapper/MPIRank.h"

#include "factory/random/random_factory.h"

#include <gtest/gtest.h>

#include <iostream>
#include <tuple>

TEST_F(StringUtilTest, testFormatException) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto random_int = RandomFactory::get_random_integer<int>(1, 999, mt);
    ASSERT_THROW_NO_PRINT(std::ignore = StringUtil::format_int_with_leading_zeros(random_int, 0);, RelearnException);
}

TEST_F(StringUtilTest, testFormatFill) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto check = [](const auto num, const auto digits, const auto descr) {
        const auto formatted = StringUtil::format_int_with_leading_zeros(num, digits);
        ASSERT_EQ(descr, formatted);
    };

    check(0, 2U, "00");
    check(1, 2U, "01");
    check(2, 2U, "02");
    check(3, 2U, "03");
    check(4, 2U, "04");
    check(5, 2U, "05");
    check(6, 2U, "06");
    check(7, 2U, "07");
    check(8, 2U, "08");
    check(9, 2U, "09");

    check(0, 3U, "000");
    check(1, 3U, "001");
    check(2, 3U, "002");
    check(3, 3U, "003");
    check(4, 3U, "004");
    check(5, 3U, "005");
    check(6, 3U, "006");
    check(7, 3U, "007");
    check(8, 3U, "008");
    check(9, 3U, "009");

    check(0, 7U, "0000000");
    check(1, 7U, "0000001");
    check(2, 7U, "0000002");
    check(3, 7U, "0000003");
    check(4, 7U, "0000004");
    check(5, 7U, "0000005");
    check(6, 7U, "0000006");
    check(7, 7U, "0000007");
    check(8, 7U, "0000008");
    check(9, 7U, "0000009");

    check(5410642, 8U, "05410642");
    check(5411642, 8U, "05411642");
    check(5412642, 8U, "05412642");
    check(5413642, 8U, "05413642");
    check(5414642, 8U, "05414642");
    check(5415642, 8U, "05415642");
    check(5416642, 8U, "05416642");
    check(5417642, 8U, "05417642");
    check(5418642, 8U, "05418642");
    check(5419642, 8U, "05419642");
}

TEST_F(StringUtilTest, testFormatNoFill) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto check = [](const auto num, const auto digits, const auto descr) {
        const auto formatted = StringUtil::format_int_with_leading_zeros(num, digits);
        ASSERT_EQ(descr, formatted);
    };

    check(0, 1U, "0");
    check(1, 1U, "1");
    check(2, 1U, "2");
    check(3, 1U, "3");
    check(4, 1U, "4");
    check(5, 1U, "5");
    check(6, 1U, "6");
    check(7, 1U, "7");
    check(8, 1U, "8");
    check(9, 1U, "9");

    check(32, 2U, "32");
    check(39, 2U, "39");
    check(65, 2U, "65");
    check(53, 2U, "53");
    check(13, 2U, "13");
    check(72, 2U, "72");
    check(95, 2U, "95");
    check(89, 2U, "89");
    check(32, 2U, "32");
    check(23, 2U, "23");

    check(303, 3U, "303");
    check(709, 3U, "709");
    check(628, 3U, "628");
    check(832, 3U, "832");
    check(221, 3U, "221");
    check(705, 3U, "705");
    check(751, 3U, "751");
    check(478, 3U, "478");
    check(359, 3U, "359");
    check(877, 3U, "877");

    check(5927, 4U, "5927");
    check(7476, 4U, "7476");
    check(4041, 4U, "4041");
    check(1774, 4U, "1774");
    check(2593, 4U, "2593");
    check(5438, 4U, "5438");
    check(4319, 4U, "4319");
    check(6947, 4U, "6947");
    check(8510, 4U, "8510");
    check(9998, 4U, "9998");

    check(18019, 5U, "18019");
    check(58587, 5U, "58587");
    check(44734, 5U, "44734");
    check(36505, 5U, "36505");
    check(44333, 5U, "44333");
    check(31948, 5U, "31948");
    check(36589, 5U, "36589");
    check(19340, 5U, "19340");
    check(79639, 5U, "79639");
    check(35187, 5U, "35187");
}

TEST_F(StringUtilTest, testFormatTooLong) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    const auto check = [](const auto num, const auto digits, const auto descr) {
        const auto formatted = StringUtil::format_int_with_leading_zeros(num, digits);
        ASSERT_EQ(descr, formatted);
    };

    check(32, 1U, "32");
    check(39, 1U, "39");
    check(65, 1U, "65");
    check(53, 1U, "53");
    check(13, 1U, "13");
    check(72, 1U, "72");
    check(95, 1U, "95");
    check(89, 1U, "89");
    check(32, 1U, "32");
    check(23, 1U, "23");

    check(303, 2U, "303");
    check(709, 2U, "709");
    check(628, 2U, "628");
    check(832, 2U, "832");
    check(221, 2U, "221");
    check(705, 2U, "705");
    check(751, 2U, "751");
    check(478, 2U, "478");
    check(359, 2U, "359");
    check(877, 2U, "877");

    check(5927, 3U, "5927");
    check(7476, 3U, "7476");
    check(4041, 3U, "4041");
    check(1774, 3U, "1774");
    check(2593, 3U, "2593");
    check(5438, 3U, "5438");
    check(4319, 3U, "4319");
    check(6947, 3U, "6947");
    check(8510, 3U, "8510");
    check(9998, 3U, "9998");

    check(18019, 4U, "18019");
    check(58587, 4U, "58587");
    check(44734, 4U, "44734");
    check(36505, 4U, "36505");
    check(44333, 4U, "44333");
    check(31948, 4U, "31948");
    check(36589, 4U, "36589");
    check(19340, 4U, "19340");
    check(79639, 4U, "79639");
    check(35187, 4U, "35187");
}

TEST_F(StringUtilTest, testStringSplitEmpty) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using namespace std::string_literals;
    const auto original = ""s;
    const auto& split = StringUtil::split_string(original, 'k');

    ASSERT_TRUE(split.empty());
}

TEST_F(StringUtilTest, testStringSplitNoDelim) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using namespace std::string_literals;
    const auto original = "gukztqns2moa,y.l#sdp+cawö34i3zc\"!$%)=&&%\"\"!°"s;

    const auto& split = StringUtil::split_string(original, 'x');

    ASSERT_EQ(split.size(), 1);
    ASSERT_EQ(split[0], "gukztqns2moa,y.l#sdp+cawö34i3zc\"!$%)=&&%\"\"!°");
}

TEST_F(StringUtilTest, testStringSplitOneDelim) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using namespace std::string_literals;
    const auto original = "0"s;

    const auto& split = StringUtil::split_string(original, '0');

    ASSERT_EQ(split.size(), 1);
    ASSERT_EQ(split[0], "");
}

TEST_F(StringUtilTest, testStringSplit) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    using namespace std::string_literals;
    const auto original = ";hfke;kg83;;058372;058372;4sfsaf ;iousahfu-+30o3q;021u3zhrns;"s;

    const auto& split = StringUtil::split_string(original, ';');

    ASSERT_EQ(split.size(), 9);
    ASSERT_EQ(split[0], "");
    ASSERT_EQ(split[1], "hfke");
    ASSERT_EQ(split[2], "kg83");
    ASSERT_EQ(split[3], "");
    ASSERT_EQ(split[4], "058372");
    ASSERT_EQ(split[5], "058372");
    ASSERT_EQ(split[6], "4sfsaf ");
    ASSERT_EQ(split[7], "iousahfu-+30o3q");
    ASSERT_EQ(split[8], "021u3zhrns");
}

TEST_F(StringUtilTest, testIsNumberEmpty) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_TRUE(StringUtil::is_number(""));
}

TEST_F(StringUtilTest, testIsNumberDigits) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_TRUE(StringUtil::is_number("984615"));
    ASSERT_TRUE(StringUtil::is_number("0"));
    ASSERT_TRUE(StringUtil::is_number("84512158875612"));
    ASSERT_TRUE(StringUtil::is_number("1111111111111"));
    ASSERT_TRUE(StringUtil::is_number("2288664422"));
}

TEST_F(StringUtilTest, testIsNumberNegative) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_FALSE(StringUtil::is_number("-984615"));
    ASSERT_FALSE(StringUtil::is_number("-0"));
    ASSERT_FALSE(StringUtil::is_number("-84512158875612"));
    ASSERT_FALSE(StringUtil::is_number("-1111111111111"));
    ASSERT_FALSE(StringUtil::is_number("-2288664422"));
}

TEST_F(StringUtilTest, testIsNumberGibberish) {
    if (mpiPP::MPIInfo::get_number_ranks() != 1) {
        if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
            std::cerr << "Test only works with 1 MPI ranks.\n";
        }

        return;
    }

    ASSERT_FALSE(StringUtil::is_number("58412121f"));
    ASSERT_FALSE(StringUtil::is_number("iaszfdoq3"));
    ASSERT_FALSE(StringUtil::is_number("56s4f225484"));
    ASSERT_FALSE(StringUtil::is_number("DEADBEEF"));
    ASSERT_FALSE(StringUtil::is_number("121484 2"));
}
