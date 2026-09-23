# The floating point precision in which the simulation calculates, i.e., the type behind
# RelearnTypes::real in source/types/BasicTypes.h and behind CudaTypes::cuda_real in
# source/cuda/CudaTypes.h. FP64 is the default and the precision every result so far was produced
# with; FP32 halves what the positions, the activities, and the octree take in memory and in the
# messages between the ranks, at the price of the ~7 significant digits of a float.
#
# The value is handed to the compilers as PRECISION_FP32=true resp. =false, which source/Macros.h
# picks up as RELEARN_PRECISION_FP32, the same way CUDA_FOUND becomes RELEARN_CUDA_FOUND. A
# translation unit that is compiled without the definition, e.g., by a syntax-only check outside of
# this build, sees the undefined macro as 0 in the #if and calculates in FP64, which is the default
# here as well.
set(PRECISION "FP64" CACHE STRING "The floating point precision of the simulation, FP32 or FP64")
set_property(CACHE PRECISION PROPERTY STRINGS "FP64" "FP32")

# Accepted in any spelling, so that -DPRECISION=fp32 configures instead of aborting.
string(TOUPPER "${PRECISION}" precision_normalized)

if (precision_normalized STREQUAL "FP64")
    add_compile_definitions(PRECISION_FP32=false)
    message(STATUS "Calculating in double precision (FP64).")
elseif (precision_normalized STREQUAL "FP32")
    add_compile_definitions(PRECISION_FP32=true)
    message(STATUS "Calculating in single precision (FP32).")
else ()
    message(FATAL_ERROR "PRECISION is '${PRECISION}', which is neither FP32 nor FP64.")
endif ()
