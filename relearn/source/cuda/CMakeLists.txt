add_library(relearn_gpu STATIC
        algorithm/BarnesHutInternalCUDA/BarnesHutCUDA_CU.cu
        algorithm/NaiveInternalCUDA/NaiveCUDA_CU.cu
        CudaBaseBridgeFunctions.cu
        wrapper/EventWrapper.cu
        wrapper/StreamWrapper.cu
        network_graph/GPUEdges.cu
        random/RandomNumber.cu
        input/ActivityInput.cu
        calcium/Calcium.cu
        memory/SharedBlockPool.cu
        memory/DeviceVecVec.cu
        spikes/ExchangeAlgorithm.cu
        network_graph/BloomFilter.cu
        network_graph/NetworkGraph.cu
        network_graph/Views.cu
        network_graph/NetworkGraphGPU.cu
        neuron_model/NeuronModels.cu
        spikes/SpikePreparation.cu
        synaptic_elements/SynapticElements.cu
        input/SynapticEquallyWeightedActivityInput.cu
        util/Util.cu
        mpi/MPICuda.cpp
)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_target_properties(relearn_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CMAKE_CUDA_STANDARD 20
        CXX_STANDARD 20
        CMAKE_CUDA_STANDARD_REQUIRED ON
        CMAKE_CXX_STANDARD_REQUIRED ON
)

target_sources(
        relearn_gpu
        PUBLIC FILE_SET
        HEADERS
        BASE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}"
        FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/memory/LazySyncedArray.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/CudaBaseBridgeFunctions.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/CudaConfig.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/CudaConversion.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/CudaTypes.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/CudaTypes.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/common.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory/DeviceArray.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/calcium/Calcium.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/calcium/CalciumHandle.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/input/Handle.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Kernel.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/network_graph/BlockCountAtomics.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/util/BinarySearch.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/util/PrefixSum.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/util/UpperBound.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/mpi/MPICuda.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternalCUDA/BarnesHutCUDA_CU.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternalCUDA/BarnesHutCUDA_CU.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternalCUDA/BarnesHutCUDA.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternalCUDA/BarnesHutCUDACell.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternalCUDA/LinearizedTree.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/ForwardGPUAlgorithm.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/NaiveInternalCUDA/NaiveCUDA_CU.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/NaiveInternalCUDA/NaiveCUDA.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/NaiveInternalCUDA/NaiveCUDACell.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/random/RandomNumber.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/random/RandomNumberHost.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/random/RandomNumberKeys.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory/SharedBlockPool.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory/DeviceVecVec.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/network_graph/BloomFilter.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/network_graph/Views.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/neuron_model/NeuronModels.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/random/RandomBridge.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/spikes/ExchangeAlgorithm.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/spikes/SpikePreparation.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/synaptic_elements/SynapticElements.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/synaptic_elements/SynapticElementsHandle.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/synaptic_elements/SynapticElements.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/util/LinearizedTreeDeviceHandle.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/util/NeuronsExtraInfoHandle.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/util/SmallNeuronIdType.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/util/Util.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/util/Util.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/wrapper/EventWrapper.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/wrapper/StreamWrapper.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/wrapper/StreamWrapper.cuh"

        "${CMAKE_CURRENT_SOURCE_DIR}/input/ActivityInput.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/input/SynapticEquallyWeightedActivityInput.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/network_graph/NetworkGraph.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/network_graph/BloomFilter.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory/SharedBlockPool.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory/DeviceVecVec.cuh"

)

target_compile_features(relearn_gpu PRIVATE cxx_std_20 cuda_std_20)

target_compile_options(relearn_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda -rdc=true>
)
# --ptxas-options=-v

set_target_properties(relearn_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA)


set_target_properties(relearn_gpu PROPERTIES CMAKE_CUDA_STANDARD 20 CMAKE_CXX_STANDARD 20 CMAKE_CUDA_STANDARD_REQUIRED ON CMAKE_CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)

target_include_directories(relearn_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(relearn_gpu PUBLIC project_libraries_gpu MPI::MPI_CXX CUDA::cudart CUDA::cuda_driver)

# relearn_gpu's .cu/.cpp translation units call back into plain C++ symbols defined in
# relearn_lib (RelearnException, GPUEdgesBase::init, ...), while relearn_lib PUBLICly links
# relearn_gpu the other way (see ../CMakeLists.txt) -- a genuine circular dependency between the
# two static libraries. Declaring it in both directions lets CMake repeat the archives on the
# final link line as needed; without this, executables whose own translation unit doesn't
# reference enough of relearn_lib up front (e.g. relearn_analysis, unlike relearn/relearn_tests)
# get "undefined reference to vtable for RelearnException" / GPUEdgesBase::init, because ld's
# single left-to-right archive pass extracts relearn_gpu's members after relearn_lib's pass has
# already finished.
target_link_libraries(relearn_gpu PUBLIC relearn_lib)

include_directories(${CMAKE_SOURCE_DIR}/source)

verify_source_files_in_targets(TARGETS relearn_gpu relearn_lib DIR ${CMAKE_CURRENT_SOURCE_DIR})
