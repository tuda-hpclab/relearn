/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "CudaBaseBridgeFunctions.h"

#ifndef RELEARN_CUDA_ENABLED

#include "algorithm/BarnesHutInternalCUDA/BarnesHutCUDA_CU.h"
#include "algorithm/NaiveInternalCUDA/NaiveCUDA_CU.h"
#include "cuda/CudaConfig.h"
#include "cuda/calcium/Calcium.h"
#include "cuda/input/ActivityInput.h"
#include "cuda/input/SynapticEquallyWeightedActivityInput.h"
#include "cuda/spikes/ExchangeAlgorithm.h"
#include "cuda/spikes/SpikePreparation.h"
#include "cuda/synaptic_elements/SynapticElements.h"
#include "cuda/synaptic_elements/SynapticElementsHandle.h"
#include "cuda/util/NeuronsExtraInfoHandle.h"
#include "cuda/util/Util.h"
#include "cuda/wrapper/StreamWrapper.h"
#include "network_graph/NetworkHandle.h"
#include "neuron_model/NeuronModels.h"
#include "neurons/enums/SynapticElementType.h"
#include "neurons/enums/UpdateStatus.h"
#include "neurons/helper/SynapseCreationResponse.h"
#include "random/RandomNumberKeys.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

// ---- StreamWrapper (null stub for non-CUDA builds) ----

StreamWrapper::StreamWrapper()
    = default;
StreamWrapper::StreamWrapper(void*) { }
StreamWrapper::~StreamWrapper() = default;
void* StreamWrapper::native_handle() const noexcept { return nullptr; }

// ---- memory / device bridges ----

void cudaMemset_bridge(void*, int, std::size_t) { CUDA_NOT_SUPPORTED }

void cudaMemsetAsync_bridge(void*, int, std::size_t, const StreamWrapper&) { CUDA_NOT_SUPPORTED }

void cudaSetDevice_bridge(int) { CUDA_NOT_SUPPORTED }

void cudaMalloc_bridge(void**, std::size_t) { CUDA_NOT_SUPPORTED }

void cudaMallocAsync_bridge(void**, std::size_t, const StreamWrapper&) { CUDA_NOT_SUPPORTED }

void cudaMallocHost_bridge(void**, std::size_t) { CUDA_NOT_SUPPORTED }

void cudaProfilerStart_bridge() { CUDA_NOT_SUPPORTED }

void cudaProfilerStop_bridge() { CUDA_NOT_SUPPORTED }

void cudaMemcpy_to_device_bridge(void*, const void*, std::size_t) { CUDA_NOT_SUPPORTED }

void cudaMemcpyAsync_to_device_bridge(void*, const void*, std::size_t) { CUDA_NOT_SUPPORTED }

void cudaMemcpyAsync_to_device_bridge(void*, const void*, std::size_t, const StreamWrapper&) { CUDA_NOT_SUPPORTED }

void cudaMemcpy_on_device_bridge(void*, const void*, std::size_t) { CUDA_NOT_SUPPORTED }

void cudaMemcpyAsync_on_device_bridge(void*, const void*, std::size_t) { CUDA_NOT_SUPPORTED }

void cudaMemcpyAsync_on_device_bridge(void*, const void*, std::size_t, const StreamWrapper&) { CUDA_NOT_SUPPORTED }

void cudaMemcpy_to_host_bridge(void*, const void*, std::size_t) { CUDA_NOT_SUPPORTED }

void cudaMemcpyAsync_to_host_bridge(void*, const void*, std::size_t) { CUDA_NOT_SUPPORTED }

void cudaMemcpyAsync_to_host_bridge(void*, const void*, std::size_t, const StreamWrapper&) { CUDA_NOT_SUPPORTED }

void cudaStreamSynchronize_bride(const StreamWrapper&) { CUDA_NOT_SUPPORTED }

void cudaFree_bridge(void*) { CUDA_NOT_SUPPORTED }

void cudaFreeHost_bridge(void*) { CUDA_NOT_SUPPORTED }

void cudaFreeAsync_bridge(void*, const StreamWrapper&) { CUDA_NOT_SUPPORTED }

void cudaDeviceSynchronize_bridge() { CUDA_NOT_SUPPORTED }

void cudaResetLastError_bridge() { CUDA_NOT_SUPPORTED }

int cudaGetDeviceCount_bridge(){ CUDA_NOT_SUPPORTED }

std::size_t getCurandStateSize(){ CUDA_NOT_SUPPORTED }

// ---- MPICuda (non-template functions) ----

std::uint64_t get_send_bytes(){ CUDA_NOT_SUPPORTED }

std::uint64_t get_recv_bytes(){ CUDA_NOT_SUPPORTED }

std::vector<std::vector<int>> get_received_bytes_vector(){ CUDA_NOT_SUPPORTED }

std::vector<std::vector<int>> get_send_bytes_vector() { CUDA_NOT_SUPPORTED }

void check_cuda_awareness(CudaConfig::mpi_rank_type, CudaConfig::mpi_rank_type) { CUDA_NOT_SUPPORTED }

// ---- neuron models ----

void update_activity_izhikevich_entry(NeuronExtraInfoHandle, unsigned,
                                      const models::izhikevich::Parameters<RelearnTypes::activity_type>&,
                                      const CudaConfig::input_type*, IzhikevichDeviceState,
                                      FiredRecorderHandle) { CUDA_NOT_SUPPORTED }

void update_activity_aeif_entry(NeuronExtraInfoHandle, unsigned,
                                const models::aeif::Parameters<RelearnTypes::activity_type>&,
                                CudaConfig::input_type*, AeifDeviceState,
                                FiredRecorderHandle) { CUDA_NOT_SUPPORTED }

void update_activity_fitzhughnagumo_entry(NeuronExtraInfoHandle, unsigned,
                                          const models::fitzhughnagumo::Parameters<RelearnTypes::activity_type>&,
                                          const CudaConfig::input_type*, FitzHughNagumoDeviceState,
                                          FiredRecorderHandle) { CUDA_NOT_SUPPORTED }

void update_activity_poisson_entry(NeuronExtraInfoHandle, unsigned,
                                   const models::poisson::Parameters<RelearnTypes::activity_type>&,
                                   const CudaConfig::input_type*, PoissonDeviceState,
                                   FiredRecorderHandle, std::uint32_t) { CUDA_NOT_SUPPORTED }

// ---- calcium ----

void update_current_calcium_entry(NeuronsExtraInfoGPUHandleConst,
                                  const FiredStatus*, CalciumHandle,
                                  unsigned int, CudaConfig::calcium_type, CudaConfig::calcium_type) { CUDA_NOT_SUPPORTED }

void update_target_calcium_absolute_decay_entry(NeuronsExtraInfoGPUHandleConst,
                                                CalciumHandle,
                                                CudaConfig::calcium_type) { CUDA_NOT_SUPPORTED }

void update_target_calcium_relative_decay_entry(NeuronsExtraInfoGPUHandleConst,
                                                CalciumHandle,
                                                CudaConfig::calcium_type) { CUDA_NOT_SUPPORTED }

// ---- synaptic elements ----

void commit_synaptic_elements_entry(SynapticElementsBaseCudaHandle,
                                    CudaConfig::synaptic_count_type*,
                                    NeuronsExtraInfoGPUHandleConst,
                                    const std::shared_ptr<StreamWrapper>&) { CUDA_NOT_SUPPORTED }

void update_number_elements_kernel_entry(CalciumHandleConst,
                                         SynapticElementsBaseCudaHandle,
                                         NeuronsExtraInfoGPUHandleConst,
                                         CudaConfig::synaptic_grown_type,
                                         const std::shared_ptr<StreamWrapper>&) { CUDA_NOT_SUPPORTED }

// ---- random ----

void curand_setup_entry(const CudaConfig::number_neurons_type, void*, const std::uint64_t) { CUDA_NOT_SUPPORTED }

void init_random_configs(RandomNumbers::RandomNumbersConfig*) { CUDA_NOT_SUPPORTED }

// ---- activity input ----

void update_combined_activity_input_range_entry(CudaConfig::number_neurons_type,
                                                CudaConfig::number_neurons_type,
                                                std::size_t,
                                                CudaConfig::input_type*,
                                                CudaConfig::input_type**,
                                                const std::shared_ptr<StreamWrapper>&) { CUDA_NOT_SUPPORTED }

void update_constant_activity_input_range_entry(CudaConfig::number_neurons_type,
                                                CudaConfig::number_neurons_type,
                                                const UpdateStatus*,
                                                CudaConfig::input_type*,
                                                CudaConfig::input_type,
                                                const std::shared_ptr<StreamWrapper>&) { CUDA_NOT_SUPPORTED }

void update_normal_activity_input_range_entry(CudaConfig::number_neurons_type,
                                              CudaConfig::number_neurons_type,
                                              const UpdateStatus*,
                                              CudaConfig::input_type*,
                                              CudaConfig::input_type, CudaConfig::input_type,
                                              std::uint32_t,
                                              const std::shared_ptr<StreamWrapper>&){ CUDA_NOT_SUPPORTED }

std::optional<EventWrapper> launch(CudaConfig::number_neurons_type,
                                   CudaConfig::number_neurons_type,
                                   CudaConfig::input_type*,
                                   const LaunchConfig&,
                                   const LaunchHandles&,
                                   const std::shared_ptr<StreamWrapper>&,
                                   CudaConfig::mpi_rank_type, bool,
                                   bool, float) { CUDA_NOT_SUPPORTED }

void release_synaptic_activity_set(){ CUDA_NOT_SUPPORTED }

// ---- synapse creation / deletion ----

std::size_t process_requests_entry_aware(SynapseCreationRequestHandle,
                                         SynapseCreationResponse*,
                                         SynapticElementsBaseCudaHandle,
                                         SynapticElementsBaseCudaHandle,
                                         NeuronsExtraInfoGPUHandleConst,
                                         NetworkHandle, std::uint64_t, std::uint64_t) { CUDA_NOT_SUPPORTED }

void process_responses_entry_aware(const int*,
                                   SynapseCreationResponseHandle,
                                   SynapseCreationResponse*,
                                   SynapticElementsBaseCudaHandle,
                                   NeuronsExtraInfoGPUHandleConst,
                                   NetworkHandle,
                                   CudaConfig::mpi_rank_type*) { CUDA_NOT_SUPPORTED }

void process_calculation_requests_entry_aware(std::uint64_t, std::uint64_t,
                                              BHCalculationRequestHandle,
                                              NeuronsExtraInfoGPUHandleConst,
                                              NeuronPopulationDeviceHandle,
                                              LinearizedTreeDeviceHandle,
                                              CudaConfig::gaussian_type,
                                              RemoteNodeRankHandle,
                                              CudaConfig::gaussian_type) { CUDA_NOT_SUPPORTED }

void find_synapses_to_delete_random_entry(DeletionRequestHandle,
                                          NeuronsExtraInfoGPUHandleConst,
                                          NetworkHandle,
                                          ElementType, SignalType,
                                          std::uint32_t, std::uint32_t) { CUDA_NOT_SUPPORTED }

void commit_deletions_entry(NeuronsExtraInfoGPUHandleConst,
                            SynapticElementsBaseCudaHandle,
                            SynapticElementsBaseCudaHandle,
                            SynapticElementsBaseCudaHandle,
                            NetworkHandle,
                            DeletionCommitHandle) { CUDA_NOT_SUPPORTED }

void check_only_outgoing_view(NetworkHandle, CudaConfig::number_neurons_type) { CUDA_NOT_SUPPORTED }

void remove_deleted_memory_pool(EdgeHandle, const CudaConfig::number_neurons_type*, CudaConfig::number_neurons_type){ CUDA_NOT_SUPPORTED }

std::uint32_t compute_sum_and_partition(const CudaConfig::synaptic_count_type*,
                                        CudaConfig::synaptic_count_type*,
                                        std::size_t){ CUDA_NOT_SUPPORTED }

// ---- spike preparation ----

PreparedSpikes
    prepare_spikes(FiredStatusHandle, NetworkHandle, NeuronsExtraInfoGPUHandleConst,
                   bool, const std::shared_ptr<StreamWrapper>&){ CUDA_NOT_SUPPORTED }

// ---- utility ----

std::size_t get_gpu_max_memory_used() { CUDA_NOT_SUPPORTED }

void set_memory_entry(std::uint32_t*, std::size_t, std::uint32_t) { CUDA_NOT_SUPPORTED }

// ---- NaiveCUDA_CU ----

namespace NaiveCUDA_CU {

void find_target_neurons(const DeviceArray<SimpleVec3d>&, std::uint64_t,
                         NaiveTargetSelectionTask,
                         std::vector<std::uint64_t>&,
                         double) { CUDA_NOT_SUPPORTED }

} // namespace NaiveCUDA_CU

// ---- BarnesHutCUDA_CU ----

namespace BarnesHutCUDA_CU {

LinearizedTreeInitResult
init_neurons(std::span<const CudaConfig::bh_index_type>,
             std::span<const CudaConfig::bh_index_type>,
             std::span<const CudaConfig::gaussian_type>,
             std::span<const CudaConfig::number_neurons_type>,
             std::span<const CudaConfig::bh_index_type>,
             std::span<const NodeType>){ CUDA_NOT_SUPPORTED }

NeuronPopulationInitResult
    init_neuron_details(std::span<const SimpleVec3d>,
                        std::span<const CudaConfig::synaptic_count_type>,
                        std::span<const CudaConfig::synaptic_count_type>,
                        std::span<const CudaConfig::mpi_rank_type>) { CUDA_NOT_SUPPORTED }

void update_vacant_elements(std::span<const CudaConfig::synaptic_count_type>,
                            CudaConfig::synaptic_count_type*,
                            std::span<const CudaConfig::synaptic_count_type>,
                            CudaConfig::synaptic_count_type*) { CUDA_NOT_SUPPORTED }

void get_updated_octree(NeuronPopulationDeviceHandle,
                        std::size_t,
                        std::vector<SimpleVec3d>&,
                        std::vector<CudaConfig::synaptic_count_type>&) { CUDA_NOT_SUPPORTED }

void calculate_updated_octree_host(NeuronPopulationDeviceHandle,
                                   LinearizedTreeDeviceHandle,
                                   std::span<const CudaConfig::bh_index_type>,
                                   const std::shared_ptr<StreamWrapper>&){ CUDA_NOT_SUPPORTED }

std::vector<bool> test_acceptance_criterion_host(CudaConfig::number_neurons_type,
                                                 const SimpleVec3d&,
                                                 const SimpleVec3d&,
                                                 CudaConfig::synaptic_count_type,
                                                 CudaConfig::gaussian_type,
                                                 CudaConfig::gaussian_type,
                                                 bool){ CUDA_NOT_SUPPORTED }

TargetNeuronSearchResultBothSignalTypes
    find_target_neurons(
        std::uint64_t, std::uint64_t,
        NeuronPopulationDeviceHandle,
        const CudaConfig::bh_index_type*, std::size_t,
        NeuronPopulationDeviceHandle,
        const CudaConfig::bh_index_type*, std::size_t,
        CudaConfig::number_neurons_type,
        LinearizedTreeDeviceHandle,
        CudaConfig::gaussian_type,
        const CudaConfig::mpi_rank_type* const,
        CudaConfig::mpi_rank_type,
        CudaConfig::gaussian_type,
        CudaConfig::mpi_rank_type,
        CudaConfig::number_neurons_type,
        const std::shared_ptr<StreamWrapper>&,
        const std::shared_ptr<StreamWrapper>&) { CUDA_NOT_SUPPORTED }

void update_leaf_nodes_entry(LinearizedTreeDeviceHandle,
                             const SignalType*,
                             SynapticElementsBaseCudaHandleConst,
                             SynapticElementsBaseCudaHandleConst,
                             SynapticElementsBaseCudaHandleConst,
                             TreeVacancyOutputHandle,
                             const std::shared_ptr<StreamWrapper>&) { CUDA_NOT_SUPPORTED }

void update_remote_nodes_host(CudaConfig::bh_index_type, int,
                              std::span<const CudaConfig::bh_index_type>,
                              NeuronPopulationDeviceHandle,
                              CudaConfig::number_neurons_type*,
                              CudaConfig::mpi_rank_type,
                              const std::shared_ptr<StreamWrapper>&) { CUDA_NOT_SUPPORTED }

} // namespace BarnesHutCUDA_CU

#endif
