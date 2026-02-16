option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)
option(ENABLE_INCLUDE_WHAT_YOU_USE
        "Enable static analysis with include-what-you-use" OFF)

if (ENABLE_CPPCHECK)
    find_program(CPPCHECK cppcheck)
    if (CPPCHECK)
        if (DEFINED CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
            set(ADDITIONAL_CPPCHECK_FLAG -i ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        endif ()

        set(CMAKE_CXX_CPPCHECK
                ${CPPCHECK}
                --suppress=missingInclude
                --enable=all
                --inline-suppr
                --inconclusive
                ${ADDITIONAL_CPPCHECK_FLAG})
        set(CMAKE_CUDA_CPPCHECK
                ${CPPCHECK}
                --suppress=missingInclude
                --enable=all
                --inline-suppr
                --inconclusive
                ${ADDITIONAL_CPPCHECK_FLAG})
    else ()
        message(SEND_ERROR "cppcheck requested but executable not found")
    endif ()
endif ()

if (ENABLE_CLANG_TIDY)
    find_program(CLANGTIDY clang-tidy)
    if (CLANGTIDY)
        set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY}
                -extra-arg=-Wno-unknown-warning-option)
    else ()
        message(SEND_ERROR "clang-tidy requested but executable not found")
    endif ()
endif ()

if (ENABLE_INCLUDE_WHAT_YOU_USE)
    find_program(INCLUDE_WHAT_YOU_USE include-what-you-use)
    if (INCLUDE_WHAT_YOU_USE)
        set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${INCLUDE_WHAT_YOU_USE})
        set(CMAKE_CUDA_INCLUDE_WHAT_YOU_USE ${INCLUDE_WHAT_YOU_USE})
    else ()
        message(
                SEND_ERROR "include-what-you-use requested but executable not found")
    endif ()
endif ()
