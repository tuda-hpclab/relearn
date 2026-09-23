// /*
//  * This file is part of the RELeARN software developed at Technical University Darmstadt
//  *
//  * Copyright (c) 2020, Technical University of Darmstadt, Germany
//  *
//  * This software may be modified and distributed under the terms of a BSD-style license.
//  * See the LICENSE file in the base directory for details.
//  *
//  */
//
// #include "test_timers.h"
//
// #include "sim/Essentials.h"
// #include "util/Timers.h"
//
// #include <mpi-wrapper/core/MPIInfo.h>
// #include <mpi-wrapper/core/MPIRank.h>
//
// #include "adapter/timers/TimersAdapter.h"
//
// #include "factory/random/random_factory.h"
//
// #include <gtest/gtest.h>
//
// #include <functional>
// #include <iostream>
// #include <thread>
// #include <tuple>
// #include <unordered_set>
// #include <utility>
// #include <vector>
//
// TEST_F(TimersTest, testReset) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     Timers::reset_all();
//
//     const auto root = Timers::to_timer_tree();
//     ASSERT_FALSE(root.has_value());
// }
//
// TEST_F(TimersTest, testStartStop) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     using namespace std::chrono_literals;
//
//     const auto region = TimersAdapter::get_random_timer_region(mt);
//     Timers::reset_all();
//
//     Timers::start(region);
//     std::this_thread::sleep_for(10000ns);
//     Timers::stop(region);
//
//     const auto root = Timers::to_timer_tree();
//     ASSERT_TRUE(root.has_value());
//     ASSERT_EQ(region, root->timer_region);
//     const auto elapsed = root->time_ns;
//     ASSERT_EQ(elapsed, 10000);
// }
//
// TEST_F(TimersTest, testMultipleStartStop) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     using namespace std::chrono_literals;
//
//     const auto region = TimersAdapter::get_random_timer_region(mt);
//     Timers::reset_all();
//
//     const auto outer_iterations = RandomFactory::get_random_integer<unsigned int>(2, 10, mt);
//
//     for (auto outer = 0U; outer < outer_iterations; outer++) {
//         const auto start_iterations = RandomFactory::get_random_integer<unsigned int>(2, 10, mt);
//         for (auto start = 0U; start < start_iterations; start++) {
//             Timers::start(region);
//         }
//
//         const auto end_iterations = RandomFactory::get_random_integer<unsigned int>(2, 10, mt);
//         for (auto end = 0U; end < end_iterations; end++) {
//             Timers::stop(region);
//         }
//     }
//
//     const auto elapsed = Timers::get_elapsed(region);
//     ASSERT_EQ(elapsed.count(), 0);
// }
//
// TEST_F(TimersTest, testResetZero) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     using namespace std::chrono_literals;
//
//     const auto region = TimersAdapter::get_random_timer_region(mt);
//     Timers::reset_all();
//
//     Timers::start(region);
//     std::this_thread::sleep_for(10000ns);
//     Timers::stop(region);
//     Timers::add_start_stop_diff_to_elapsed(region);
//
//     Timers::reset_all();
//
//     const auto elapsed = Timers::get_elapsed(region);
//     ASSERT_EQ(elapsed.count(), 0);
// }
//
// TEST_F(TimersTest, testResetZero2) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     using namespace std::chrono_literals;
//
//     const auto region = TimersAdapter::get_random_timer_region(mt);
//     Timers::reset_all();
//
//     Timers::start(region);
//     std::this_thread::sleep_for(10000ns);
//     Timers::stop_and_add(region);
//
//     Timers::reset_all();
//
//     const auto elapsed = Timers::get_elapsed(region);
//     ASSERT_EQ(elapsed.count(), 0);
// }
//
// TEST_F(TimersTest, testAdd) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     using namespace std::chrono_literals;
//
//     const auto region = TimersAdapter::get_random_timer_region(mt);
//     Timers::reset_all();
//
//     Timers::start(region);
//     std::this_thread::sleep_for(10000ns);
//     Timers::stop(region);
//     Timers::add_start_stop_diff_to_elapsed(region);
//
//     const auto elapsed = Timers::get_elapsed(region);
//     ASSERT_GE(elapsed.count(), 10000);
//
//     std::this_thread::sleep_for(10000ns);
//     const auto elapsed_again = Timers::get_elapsed(region);
//     ASSERT_EQ(elapsed_again, elapsed);
// }
//
// TEST_F(TimersTest, testAdd2) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     using namespace std::chrono_literals;
//
//     const auto region = TimersAdapter::get_random_timer_region(mt);
//     Timers::reset_all();
//
//     Timers::start(region);
//     std::this_thread::sleep_for(10000ns);
//     Timers::stop_and_add(region);
//
//     const auto elapsed = Timers::get_elapsed(region);
//     ASSERT_GE(elapsed.count(), 10000);
//
//     std::this_thread::sleep_for(10000ns);
//     const auto elapsed_again = Timers::get_elapsed(region);
//     ASSERT_EQ(elapsed_again, elapsed);
// }
//
// TEST_F(TimersTest, testNonInterference) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     using namespace std::chrono_literals;
//
//     auto get_two_timers = [this]() {
//         auto first = TimersAdapter::get_random_timer_region(mt);
//         while (true) {
//             auto second = TimersAdapter::get_random_timer_region(mt);
//             if (second != first) {
//                 return std::pair{ first, second };
//             }
//         }
//     };
//     const auto [first_region, second_region] = get_two_timers();
//
//     Timers::reset_elapsed(first_region);
//     Timers::reset_elapsed(second_region);
//
//     Timers::start(first_region);
//     std::this_thread::sleep_for(10000ns);
//     Timers::stop_and_add(first_region);
//
//     const auto elapsed_0 = Timers::get_elapsed(first_region);
//     ASSERT_GE(elapsed_0.count(), 10000);
//
//     const auto elapsed_1 = Timers::get_elapsed(second_region);
//     ASSERT_EQ(elapsed_1.count(), 0);
//
//     Timers::reset_elapsed(second_region);
//
//     const auto elapsed_2 = Timers::get_elapsed(first_region);
//     ASSERT_EQ(elapsed_0, elapsed_2);
//
//     Timers::start(second_region);
//     std::this_thread::sleep_for(10000ns);
//     Timers::stop_and_add(second_region);
//
//     const auto elapsed_3 = Timers::get_elapsed(first_region);
//     ASSERT_EQ(elapsed_0, elapsed_3);
//
//     Timers::start(second_region);
//     std::this_thread::sleep_for(10000ns);
//     Timers::stop(second_region);
//     Timers::add_start_stop_diff_to_elapsed(second_region);
//
//     const auto elapsed_4 = Timers::get_elapsed(first_region);
//     ASSERT_EQ(elapsed_0, elapsed_4);
// }
//
// TEST_F(TimersTest, testNoThrowWallTime) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     ASSERT_NO_THROW(std::ignore = Timers::wall_clock_time());
// }
//
// TEST_F(TimersTest, testPrintNoThrow) {
//     if (mpiPP::MPIInfo::get_number_ranks() != 1) {
//         if (mpiPP::MPIInfo::get_my_rank() == mpiPP::MPIRank::root_rank()) {
//             std::cerr << "Test only works with 1 MPI ranks.\n";
//         }
//
//         return;
//     }
//
//     auto essentials = std::make_unique<Essentials>();
//
//     auto ss = std::stringstream{};
//     essentials->print(ss);
//
//     ASSERT_EQ(ss.str().size(), 0);
//
//     ASSERT_NO_THROW(Timers::print_human_readable(essentials));
//     ASSERT_NO_THROW(Timers::print_json());
//     ASSERT_NO_THROW(Timers::print_extrap(1, 1));
//     essentials->print(ss);
//
//     ASSERT_GT(ss.str().size(), 0);
// }
//
// TEST_F(TimersTest, testHierarchy) {
//     const auto root = root_timer;
//
//     std::function<std::vector<TimerRegion>(TimerHierarchy)> walk;
//
//     walk = [&walk](const auto& n) -> std::vector<TimerRegion> {
//         auto flat_childs = std::vector{ n.timer_region };
//         for (const auto& child : n.children) {
//             const auto res = walk(child);
//             std::copy(res.cbegin(), res.cend(), std::back_inserter(flat_childs));
//         }
//         return flat_childs;
//     };
//
//     const auto& flat_childs = walk(root);
//     const auto child_set = std::unordered_set<TimerRegion>{ flat_childs.begin(), flat_childs.end() };
//     ASSERT_EQ(child_set.size(), flat_childs.size());
//     ASSERT_EQ(child_set.size(), NUMBER_TIMERS);
// }
