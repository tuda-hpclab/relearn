add_library(relearn_lib STATIC)

target_sources(
        relearn_lib
        PRIVATE
        # algorithm
        algorithm/BarnesHutInternal/BarnesHut.cpp
        algorithm/BarnesHutInternal/BarnesHutInverted.cpp
        algorithm/BarnesHutInternal/BarnesHutLocationAware.cpp
        algorithm/CombinedAlgorithmsInternal/AlgorithmConfig.cpp
        algorithm/CombinedAlgorithmsInternal/CombinedAlgorithms.cpp
        algorithm/FMMInternal/FastMultipoleMethod.cpp
        algorithm/FMMInternal/FastMultipoleMethodInverted.cpp
        algorithm/NaiveInternal/Naive.cpp
        algorithm/Connector.cpp

        # io
        io/parser/ActivityInputParser.cpp
        io/AxonPositionIO.cpp
        io/BackgroundActivityIO.cpp
        io/CalciumIO.cpp
        io/Event.cpp
        io/InteractiveNeuronIO.cpp
        io/LogFiles.cpp
        io/NeuronIO.cpp
        io/NeuronToAlgorithmIO.cpp
        io/SynapticElementsIO.cpp

        # neurons
        neurons/calcium/AbsoluteDecayCalciumCalculator.cpp
        neurons/calcium/CalciumCalculator.cpp
        neurons/calcium/RelativeDecayCalciumCalculator.cpp

        neurons/firing/FiredStatusApproximator.cpp
        neurons/firing/FiredStatusCommunicationMap.cpp
        neurons/firing/FiredStatusRecorder.cpp

        neurons/helper/GroupMonitor.cpp
        neurons/helper/CachedChoiceFunction.cpp
        neurons/helper/NeuronMonitor.cpp
        neurons/helper/SynapseDeletionFinder.cpp

        neurons/input/FlexibleActivityInput.cpp
        neurons/input/NormalActivityInput.cpp
        neurons/input/ScaleActivityInput.cpp
        neurons/input/StimulationActivityInput.cpp
        neurons/input/SynapticActivityInput.cpp
        neurons/input/SynapticEquallyWeightedActivityInput.cpp
        neurons/input/SynapticScalingActivityInput.cpp
        neurons/input/SynapticIndividuallyWeightedActivityInput.cpp
        
        neurons/models/aeif/AEIFModel.cpp
        neurons/models/fitzhughnagumo/FitzHughNagumoModel.cpp
        neurons/models/izhikevich/IzhikevichModel.cpp
        neurons/models/poisson/PoissonModel.cpp
		
        neurons/synaptic_elements/Axons.cpp
        neurons/synaptic_elements/Dendrites.cpp
        neurons/synaptic_elements/MultiPositionAxons.cpp
        neurons/synaptic_elements/SynapticElements.cpp

        neurons/models/NeuronModel.cpp

        neurons/Neurons.cpp
        neurons/NeuronsExtraInfo.cpp

        # sim
        sim/file/MultipleFilesSynapseLoader.cpp
        sim/file/MultipleSubdomainsFromFile.cpp

        sim/random/BoxBasedRandomSubdomainAssignment.cpp
        sim/random/RandomSynapseLoader.cpp
        sim/random/SubdomainFromNeuronDensity.cpp
        sim/random/SubdomainFromNeuronPerRank.cpp

        sim/NeuronToSubdomainAssignment.cpp
        sim/Simulation.cpp
        sim/SynapseLoader.cpp

        structure/Partition.cpp

        util/RelearnException.cpp
        util/StringUtil.cpp
        util/Timers.cpp
)

target_sources(
  relearn_lib
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/Config.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/Types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/Types2.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/Types3.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Algorithm.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/AlgorithmEnum.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Algorithms.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternal/BarnesHut.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternal/BarnesHutBase.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternal/BarnesHutCell.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternal/BarnesHutInverted.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/BarnesHutInternal/BarnesHutLocationAware.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Cells.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/CombinedAlgorithmsInternal/AlgorithmConfig.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/CombinedAlgorithmsInternal/CombinedAlgorithms.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/CombinedAlgorithmsInternal/RequestEnums.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Connector.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/FMMInternal/FastMultipoleMethod.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/FMMInternal/FastMultipoleMethodBase.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/FMMInternal/FastMultipoleMethodCell.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/FMMInternal/FastMultipoleMethodInverted.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Internal/ExchangingAlgorithm.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Internal/OctreeAlgorithm.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Internal/octree/BaseCell.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Internal/octree/Cell.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Internal/octree/NodeCache.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Internal/octree/Octree.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Internal/octree/OctreeNode.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Internal/octree/OctreeNodeHelper.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Kernel/Gamma.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Kernel/Gaussian.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Kernel/Kernel.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Kernel/KernelBase.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Kernel/KernelType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Kernel/Linear.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/Kernel/Weibull.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/NaiveInternal/Naive.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/NaiveInternal/NaiveBase.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/NaiveInternal/NaiveCell.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithm/VirtualPlasticityElement.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/AxonPositionIO.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/BackgroundActivityIO.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/CalciumIO.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/Event.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/FileValidator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/InteractiveNeuronIO.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/LogFiles.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/NeuronIO.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/NeuronToAlgorithmIO.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/SynapticElementsIO.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/parser/ActivityInputParser.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/parser/IntervalParser.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/parser/MonitorParser.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/parser/NeuronIdParser.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/parser/StepParser.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/parser/StimulusParser.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/LocalGroupTranslator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/NetworkGraph.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/Neurons.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/NeuronsExtraInfo.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/calcium/AbsoluteDecayCalciumCalculator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/calcium/CalciumCalculator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/calcium/RelativeDecayCalciumCalculator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/AxonsType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/BackgroundActivityCalculatorType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/CalciumCalculatorType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/FiredStatus.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/FiredStatusCommunicatorType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/GrowthrateCalculatorType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/NeuronModelType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/SynapticElementType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/SynapticInputCalculatorType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/TargetCalciumDecay.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/enums/UpdateStatus.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/firing/FiredStatusApproximator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/firing/FiredStatusCommunicationMap.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/firing/FiredStatusCommunicator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/firing/FiredStatusRecorder.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/growthrate/ConstantGrowthrateCalculator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/growthrate/GrowthrateCalculator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/GroupMonitor.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/CachedChoiceFunction.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/ChoiceFunction.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/DistantNeuronRequests.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/GlobalGroupMapper.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/NeuronMonitor.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/RankNeuronId.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/Synapse.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/SynapseCreationRequests.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/SynapseDeletionFinder.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/helper/SynapseDeletionRequests.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/ActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/CombinedActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/ConstantActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/FlexibleActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/NormalActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/ScaleActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/StimulationActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/SynapticActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/SynapticEquallyWeightedActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/SynapticIndividuallyWeightedActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/input/SynapticScalingActivityInput.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/NeuronModel.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/aeif/AEIFModel.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/aeif/Calculation.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/aeif/Parameters.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/fitzhughnagumo/Calculation.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/fitzhughnagumo/FitzHughNagumoModel.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/fitzhughnagumo/Parameters.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/izhikevich/Calculation.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/izhikevich/IzhikevichModel.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/izhikevich/Parameters.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/poisson/Calculation.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/poisson/Parameters.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/models/poisson/PoissonModel.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/synaptic_elements/Axons.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/synaptic_elements/Dendrites.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/synaptic_elements/ElementBase.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/synaptic_elements/GrowthCurves.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/synaptic_elements/MultiPositionAxons.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/neurons/synaptic_elements/SynapticElements.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/Essentials.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/LoadedNeuron.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/NeuronToSubdomainAssignment.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/Simulation.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/SynapseLoader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/file/AdditionalPositionInformation.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/file/MultipleFilesSynapseLoader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/file/MultipleSubdomainsFromFile.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/random/BoxBasedRandomSubdomainAssignment.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/random/RandomSynapseLoader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/random/SubdomainFromNeuronDensity.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sim/random/SubdomainFromNeuronPerRank.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/structure/Hilbert.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/structure/Morton.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/structure/Partition.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/structure/SpaceFillingCurve.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/structure/SpaceFillingCurveType.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/Accumulate.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/BoundingBox.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/CalculatorUtil.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/File.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/MemoryHolder.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/NeuronFilePaths.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/NeuronID.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/ProbabilityPicker.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/Random.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/RelearnAllocator.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/RelearnException.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/SetUtil.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/StatisticalMeasures.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/StringUtil.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/Timers.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/Vec3.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/VectorUtil.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util/shuffle/shuffle.h"
)

set_target_properties(relearn_lib PROPERTIES ENABLE_EXPORTS 1)
target_link_libraries(relearn_lib PUBLIC project_options project_libraries
                                         project_warnings)
set_target_properties(relearn_lib PROPERTIES CXX_STANDARD 20)
target_compile_definitions(relearn_lib PRIVATE RELEARN_LIB)
target_include_directories(relearn_lib PRIVATE "${CMAKE_SOURCE_DIR}/source")

# Required for the ActivityInputParser to be created during compile-time (ctpg
# library) Source:
# https://github.com/peter-winter/ctpg/blob/master/examples/CMakeLists.txt
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(relearn_lib PRIVATE -fconstexpr-steps=100000000)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(relearn_lib PRIVATE -fconstexpr-ops-limit=100000000)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(relearn_lib PRIVATE /EHsc /constexpr:steps100000000)
endif()

get_target_property(relearn_lib_sources relearn_lib SOURCES)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_lib_sources})
get_target_property(relearn_lib_headers relearn_lib HEADERS)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_lib_headers})

# benchmark# dont clutter the compile_commands file with libraries
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG 4160d259d961cd393fd8d67590a8c7d210207348
)
FetchContent_MakeAvailable(cli11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_target_property(cli11_includes CLI11 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(CLI11 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
        "${cli11_includes}")

# relearn executable
add_executable(relearn)
target_sources(relearn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/relearn.cpp)
target_link_libraries(relearn PUBLIC relearn_lib project_libraries)
target_link_libraries(relearn PRIVATE CLI11::CLI11)
set_target_properties(relearn PROPERTIES ENABLE_EXPORTS 1)
set_target_properties(relearn PROPERTIES CXX_STANDARD 20)

target_include_directories(relearn PRIVATE "${CMAKE_SOURCE_DIR}/source")
get_target_property(relearn_sources relearn SOURCES)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_sources})
get_target_property(relearn_headers relearn HEADERS)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${relearn_headers})

verify_source_files_in_targets(TARGETS relearn_lib relearn DIR ${CMAKE_CURRENT_SOURCE_DIR})
