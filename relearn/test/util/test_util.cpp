/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_util.h"

#include "util/NeuronID.h"
#include "util/SetUtil.h"
#include "util/VectorUtil.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>
#include <vector>

TEST_F(UtilTest, consecutiveVectorTest) {
    const auto is_consecutive_int = [](const auto i1, const auto i2) { return i1 + 1 == i2 || i1 == i2; };
    ASSERT_EQ(VectorUtil::splitConsecutiveOrEqual<int>({}, is_consecutive_int).size(), 0);

    ASSERT_EQ(std::vector<std::vector<int>>{ { 5 } }, VectorUtil::splitConsecutiveOrEqual<int>({ 5 }, is_consecutive_int));

    ASSERT_EQ((std::vector<std::vector<int>>{ { 1, 2, 3 } }), VectorUtil::splitConsecutiveOrEqual<int>({ 1, 2, 3 }, is_consecutive_int));

    ASSERT_EQ((std::vector<std::vector<int>>{ { 3, 4, 5 }, { 7, 8 }, { 12, 13 }, { 15 } }), VectorUtil::splitConsecutiveOrEqual<int>({ 12, 3, 7, 4, 5, 8, 13, 15 }, is_consecutive_int));

    ASSERT_EQ((std::vector<std::vector<int>>{ { 7, 7, 7 } }), VectorUtil::splitConsecutiveOrEqual<int>({ 7, 7, 7 }, is_consecutive_int));
}

TEST_F(UtilTest, testContainersHaveCommonElement) {
    ASSERT_TRUE(SetUtil::containers_have_common_element(std::unordered_set<int>{ 1, 2, 3, 4, 5 }, std::vector<int>{ 5, 6, 7, 8 }));
    ASSERT_TRUE(SetUtil::containers_have_common_element(std::unordered_set<int>{ 1 }, std::vector<int>{ 1 }));
    ASSERT_TRUE(SetUtil::containers_have_common_element(std::unordered_set<std::string>{ "same", "diff" }, std::vector<std::string>{ "same", "different" }));
    ASSERT_TRUE(SetUtil::containers_have_common_element(std::unordered_set<NeuronID>{ NeuronID{ 0 }, NeuronID{ 1 } }, std::vector<NeuronID>{ NeuronID{ 1 }, NeuronID{ 2 } }));
    ASSERT_FALSE(SetUtil::containers_have_common_element(std::unordered_set<int>{ 1, 2, 3, 4, 5 }, std::vector<int>{ 6, 7, 8 }));
    ASSERT_FALSE(SetUtil::containers_have_common_element(std::unordered_set<int>{}, std::vector<int>{ 5, 6, 7, 8 }));
    ASSERT_FALSE(SetUtil::containers_have_common_element(std::unordered_set<int>{ 1, 2, 3, 4, 5 }, std::vector<int>{}));
    ASSERT_FALSE(SetUtil::containers_have_common_element(std::unordered_set<int>{}, std::vector<int>{}));
    ASSERT_FALSE(SetUtil::containers_have_common_element(std::unordered_set<std::string>{ "diff", "yza" }, std::vector<std::string>{ "different", "xyz" }));
    ASSERT_FALSE(SetUtil::containers_have_common_element(std::unordered_set<NeuronID>{ NeuronID{ 0 }, NeuronID{ false, 1 } }, std::vector<NeuronID>{ NeuronID{ true, 1 }, NeuronID{ 2 } }));
}