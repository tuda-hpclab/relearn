# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(
            STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE
            RelWithDebInfo
            CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui, ccmake
    set_property(
            CACHE CMAKE_BUILD_TYPE
            PROPERTY STRINGS
            "Debug"
            "Release"
            "MinSizeRel"
            "RelWithDebInfo")
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    # Workaround: Internal compiler error gcc
    add_compile_options(-fno-tree-bit-ccp)
    add_compile_options(-fno-tree-ccp)
    add_compile_options(-fno-tree-ch)
    add_compile_options(-fno-tree-coalesce-vars)
    add_compile_options(-fno-tree-copy-prop)
    add_compile_options(-fno-tree-dce)
    add_compile_options(-fno-tree-dominator-opts)
    add_compile_options(-fno-tree-dse)
    add_compile_options(-fno-tree-forwprop)
    add_compile_options(-fno-tree-fre)
    add_compile_options(-fno-tree-phiprop)
    add_compile_options(-fno-tree-pta)
    add_compile_options(-fno-tree-scev-cprop)
    add_compile_options(-fno-tree-sink)
    add_compile_options(-fno-tree-slsr)
    add_compile_options(-fno-tree-sra)
    add_compile_options(-fno-tree-ter)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    add_compile_options(
	  "$<$<COMPILE_LANGUAGE:CXX>:/MP;/bigobj>"
	)
endif()

# Generate compile_commands.json to make it easier to work with clang based
# tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
else ()
    message(
            STATUS
            "No colored compiler diagnostic set for '${CMAKE_CXX_COMPILER_ID}' compiler."
    )
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
