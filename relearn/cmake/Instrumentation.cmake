option(ENABLE_SCOREP "Enable scorep" OFF)
if (ENABLE_SCOREP)
    set(SCOREP_ARGS
            ""
            CACHE STRING "Arguments for scorep")
    string(
            REPLACE ";"
            " "
            SCOREP_ARGS
            "${SCOREP_ARGS}")
    message(STATUS "scorep command: scorep ${SCOREP_ARGS}")
    message(
            NOTICE
            "scorep instrumentation enabled; targets test and benchmarks are disabled")
    message(
            NOTICE
            "Using compiler/linker launcher to run scorep triggers rebuilds, to avoid rebuilds reconfigure with:\n\tSCOREP_WRAPPER=off CC=scorep-gcc CXX=scorep-g++ cmake .. -DENABLE_SCOREP=OFF \n\tSCOREP_WRAPPER_INSTRUMENTER_FLAGS=\"--thread=omp...\" make"
    )
    file(WRITE ${CMAKE_BINARY_DIR}/scorep_launcher.sh
            "#!/bin/sh\nscorep ${SCOREP_ARGS} $@\n")
    file(
            CHMOD
            ${CMAKE_BINARY_DIR}/scorep_launcher.sh
            FILE_PERMISSIONS
            OWNER_EXECUTE
            OWNER_READ
            OWNER_WRITE)
    set(SCOREP_CMD ${CMAKE_BINARY_DIR}/scorep_launcher.sh)
    add_custom_target(scorep_launcher
            DEPENDS ${CMAKE_BINARY_DIR}/scorep_launcher.sh)
    add_dependencies(project_options scorep_launcher)
    set(CMAKE_C_COMPILER_LAUNCHER ${SCOREP_CMD})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${SCOREP_CMD})
    set(CMAKE_C_LINKER_LAUNCHER ${SCOREP_CMD})
    set(CMAKE_CXX_LINKER_LAUNCHER ${SCOREP_CMD})
endif ()
