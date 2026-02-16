option(ENABLE_USER_LINKER "Enable a specific linker if available" OFF)

include(CheckCXXCompilerFlag)

set(USER_LINKER_OPTION
        "lld"
        CACHE STRING "Linker to be used")
set(USER_LINKER_OPTION_VALUES
        "lld"
        "gold"
        "bfd"
        "mold")
set_property(CACHE USER_LINKER_OPTION PROPERTY STRINGS
        ${USER_LINKER_OPTION_VALUES})
list(
        FIND
        USER_LINKER_OPTION_VALUES
        ${USER_LINKER_OPTION}
        USER_LINKER_OPTION_INDEX)

if (${USER_LINKER_OPTION_INDEX} EQUAL -1)
    message(
            STATUS
            "Using custom linker: '${USER_LINKER_OPTION}', explicitly supported entries are ${USER_LINKER_OPTION_VALUES}"
    )
endif ()

function(configure_linker project_name)
    if (NOT ENABLE_USER_LINKER)
        return()
    endif ()

    set(LINKER_FLAG "-fuse-ld=${USER_LINKER_OPTION}")

    check_cxx_compiler_flag(${LINKER_FLAG} CXX_SUPPORTS_USER_LINKER)
    if (CXX_SUPPORTS_USER_LINKER)
        target_link_options(${project_name} INTERFACE ${LINKER_FLAG})
    endif ()
endfunction()

option(ENABLE_IPO
        "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)"
        OFF)

if (ENABLE_IPO)
    if (ENABLE_USER_LINKER)
        if (USER_LINKER_OPTION STREQUAL "lld" OR USER_LINKER_OPTION STREQUAL "mold")
            add_link_options(-flto=thin)
        endif ()
    else ()
        include(CheckIPOSupported)
        check_ipo_supported(RESULT result OUTPUT output)
        if (result)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        else ()
            message(SEND_ERROR "IPO is not supported: ${output}")
        endif ()
    endif ()
endif ()
