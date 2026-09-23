/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#ifdef RELEARN_CUDA_ENABLED

#include "RelearnTest.hpp"

#include "cuda/CudaConfig.h"
#include "cuda/memory/DeviceArray.h"
#include "cuda/mpi/MPICuda.h"
#include "cuda/wrapper/StreamWrapper.h"

#include <mpi-wrapper/core/MPIInfo.h>

#include <cstdint>
#include <span>
#include <vector>

class MPICudaTest : public RelearnTest { };

// These tests exercise real MPI collectives (MPI_Alltoallv / MPI_Allgatherv / MPI_Allreduce) over
// GPU buffers. mpiPP::MPIWrapper::init() runs once in main() (see RelearnTest.cpp), so this works
// whether relearn_tests happens to run as a single process (the common local/CI case, where every
// rank is also its own only peer) or under `mpirun -n N`. To stay correct either way, these tests
// never hardcode number_ranks == 1: they read the actual rank count and verify each rank's
// self-addressed segment (what a rank sends to itself) round-trips correctly, which is a
// well-defined, meaningful check regardless of how many ranks are actually running.

TEST_F(MPICudaTest, testAllToAllRoundTripsSelfAddressedData) {
    const auto number_ranks = static_cast<std::size_t>(mpiPP::MPIInfo::get_number_ranks());
    const auto my_rank = static_cast<std::uint64_t>(mpiPP::MPIInfo::get_my_rank().get_rank());
    constexpr auto per_rank_count = 3UL;

    const auto h_sizes = std::vector<int>(number_ranks, static_cast<int>(per_rank_count));

    // Slot (target_rank, j) sent by this rank encodes (my_rank, target_rank, j) so that whatever
    // ends up back in this rank's own segment (data addressed to itself) can be checked exactly.
    auto h_send = std::vector<std::uint64_t>(number_ranks * per_rank_count);
    for (auto target_rank = 0UL; target_rank < number_ranks; ++target_rank) {
        for (auto j = 0UL; j < per_rank_count; ++j) {
            h_send[target_rank * per_rank_count + j] = (my_rank << 32) | (target_rank << 16) | j;
        }
    }

    const auto send_bytes_before = get_send_bytes();
    const auto recv_bytes_before = get_recv_bytes();

    auto d_send = DeviceArray<std::uint64_t>(std::span<const std::uint64_t>(h_send));
    auto [recv_buffer, recv_sizes, recv_displs] = cuda_all_to_all<std::uint64_t>(d_send, h_sizes, number_ranks, /*stats=*/true);

    ASSERT_GT(get_send_bytes(), send_bytes_before) << "get_send_bytes() must reflect the just-performed transfer";
    ASSERT_GT(get_recv_bytes(), recv_bytes_before) << "get_recv_bytes() must reflect the just-performed transfer";

    ASSERT_EQ(recv_sizes[my_rank], static_cast<int>(per_rank_count));

    const auto h_recv = recv_buffer.get_device_data();
    for (auto j = 0UL; j < per_rank_count; ++j) {
        const auto expected = (my_rank << 32) | (my_rank << 16) | j;
        ASSERT_EQ(h_recv[static_cast<std::size_t>(recv_displs[my_rank]) + j], expected);
    }
}

TEST_F(MPICudaTest, testAllToAllAsyncRoundTripsSelfAddressedData) {
    // cuda_all_to_all_async is only explicitly instantiated for CudaConfig::bh_index_type
    // (uint32_t) in MPICuda.cpp, so the packed encoding below uses 32, not 64, bits.
    using elem_t = CudaConfig::bh_index_type;
    const auto number_ranks = static_cast<std::size_t>(mpiPP::MPIInfo::get_number_ranks());
    const auto my_rank = static_cast<elem_t>(mpiPP::MPIInfo::get_my_rank().get_rank());
    constexpr auto per_rank_count = 2UL;

    const auto h_sizes = std::vector<int>(number_ranks, static_cast<int>(per_rank_count));

    auto h_send = std::vector<elem_t>(number_ranks * per_rank_count);
    for (auto target_rank = 0UL; target_rank < number_ranks; ++target_rank) {
        for (auto j = 0UL; j < per_rank_count; ++j) {
            h_send[target_rank * per_rank_count + j] = (my_rank << 16) | (static_cast<elem_t>(target_rank) << 8) | static_cast<elem_t>(j);
        }
    }

    // The taking-ownership overload: the send buffer is moved in, so build it fresh here.
    auto d_send = DeviceArray<elem_t>(std::span<const elem_t>(h_send));
    auto [recv_sizes, recv_displs, async_request] = cuda_all_to_all_async<elem_t>(std::move(d_send), h_sizes, number_ranks, /*stats=*/false);

    auto&& recv_buffer = async_request.wait_no_ref();

    ASSERT_EQ(recv_sizes[my_rank], static_cast<int>(per_rank_count));

    const auto h_recv = recv_buffer.get_device_data();
    for (auto j = 0UL; j < per_rank_count; ++j) {
        const auto expected = (my_rank << 16) | (my_rank << 8) | static_cast<elem_t>(j);
        ASSERT_EQ(h_recv[static_cast<std::size_t>(recv_displs[my_rank]) + j], expected);
    }
}

TEST_F(MPICudaTest, testAllToAllAsyncP2PByRefRoundTripsSelfAddressedData) {
    // cuda_all_to_all_async_p2p is only explicitly instantiated for CudaConfig::bh_index_type.
    using elem_t = CudaConfig::bh_index_type;
    const auto number_ranks = static_cast<std::size_t>(mpiPP::MPIInfo::get_number_ranks());
    const auto my_rank = static_cast<elem_t>(mpiPP::MPIInfo::get_my_rank().get_rank());
    constexpr auto per_rank_count = 2UL;

    const auto h_sizes = std::vector<int>(number_ranks, static_cast<int>(per_rank_count));

    auto h_send = std::vector<elem_t>(number_ranks * per_rank_count);
    for (auto target_rank = 0UL; target_rank < number_ranks; ++target_rank) {
        for (auto j = 0UL; j < per_rank_count; ++j) {
            h_send[target_rank * per_rank_count + j] = (my_rank << 16) | (static_cast<elem_t>(target_rank) << 8) | static_cast<elem_t>(j);
        }
    }

    // The by-reference overload reuses caller-owned send/recv buffers instead of taking
    // ownership of the send buffer -- so, unlike testAllToAllAsyncRoundTripsSelfAddressedData,
    // d_send stays valid (and is not moved) after the call, and the result must be read back
    // via async_request.wait() + d_recv (the buffer this test owns), not wait_no_ref(): the
    // AsyncRequest built by the reuse constructor never touches its own internal receive
    // buffer, so wait_no_ref() would hand back that untouched (empty) buffer instead.
    auto d_send = DeviceArray<elem_t>(std::span<const elem_t>(h_send));
    auto d_recv = DeviceArray<elem_t>(0);
    auto [recv_sizes, recv_displs, async_request] = cuda_all_to_all_async_p2p<elem_t>(d_send, d_recv, h_sizes, number_ranks, /*stats=*/false);

    async_request.wait();

    ASSERT_EQ(recv_sizes[my_rank], static_cast<int>(per_rank_count));

    const auto h_recv = d_recv.get_device_data();
    for (auto j = 0UL; j < per_rank_count; ++j) {
        const auto expected = (my_rank << 16) | (my_rank << 8) | static_cast<elem_t>(j);
        ASSERT_EQ(h_recv[static_cast<std::size_t>(recv_displs[my_rank]) + j], expected);
    }
}

TEST_F(MPICudaTest, testAllGatherCollectsEachRanksContribution) {
    // cuda_allgather is only explicitly instantiated for SimpleVec3d and unsigned int.
    using elem_t = unsigned int;
    const auto number_ranks = static_cast<std::size_t>(mpiPP::MPIInfo::get_number_ranks());
    const auto my_rank = static_cast<elem_t>(mpiPP::MPIInfo::get_my_rank().get_rank());

    const auto h_send = std::vector<elem_t>{ my_rank, my_rank * 100 };
    auto d_send = DeviceArray<elem_t>(std::span<const elem_t>(h_send));

    const auto stream = StreamWrapper::default_stream();
    auto [recv_buffer, recv_sizes, recv_displs] = cuda_allgather<elem_t>(d_send, number_ranks, stream);

    ASSERT_EQ(recv_buffer.size(), number_ranks * h_send.size());
    for (const auto s : recv_sizes) {
        ASSERT_EQ(s, static_cast<int>(h_send.size()));
    }

    const auto h_recv = recv_buffer.get_device_data();
    // This rank's own contribution must appear intact at its own displacement.
    ASSERT_EQ(h_recv[static_cast<std::size_t>(recv_displs[my_rank])], my_rank);
    ASSERT_EQ(h_recv[static_cast<std::size_t>(recv_displs[my_rank]) + 1], my_rank * 100);
}

// check_cuda_awareness() is deliberately not tested here: it unconditionally passes raw device
// pointers straight to MPI_Allreduce, which is only valid with a CUDA-aware MPI build. Production
// code only calls it behind `if (Config::cuda_aware_mpi_available)` (see relearn.cpp) precisely
// because it segfaults otherwise -- and Config::cuda_aware_mpi_available is never set to true in
// this test binary (it's populated from a CLI flag in relearn.cpp's main(), not RelearnTest.cpp's).
// Calling it unconditionally here reliably segfaults on a non-CUDA-aware MPI install (confirmed:
// MPI_Allreduce -> mca_coll_self_allreduce_intra crashes with "Invalid permissions" trying to read
// the device pointer as host memory), which is very likely the common case for CI/local dev
// environments, so it is not safe to exercise unconditionally.

#endif
