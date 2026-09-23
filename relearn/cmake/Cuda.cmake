option(ENABLE_CUDA "Enable cuda" ON)
option(GENCODE "Cuda gen code" none)

if (ENABLE_CUDA)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        set(CUDA_SEPARABLE_COMPILATION ON)
        # -lineinfo conflicts with -G (device-debug, added for CMAKE_BUILD_TYPE=Debug in the top-level
        # CMakeLists.txt): nvcc silently drops -lineinfo but ptxas still sees both and aborts. Only add
        # it outside of Debug builds.
        if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
        endif ()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number")

        if (NOT GENCODE)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES "${GENCODE}")
        endif()

        enable_language(CUDA)
        set(RELEARN_CUDA_ENABLED ON)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        add_compile_definitions(RELEARN_CUDA_ENABLED=true)
		message(STATUS "CUDA successfully found.")

		if (CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
			if (CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES "^[0-9]")
				set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
				message(STATUS "Building for native CUDA architecture(s): ${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
			else()
				set(_CUDA_ARCH_FALLBACK "75")
				set(CMAKE_CUDA_ARCHITECTURES "${_CUDA_ARCH_FALLBACK}")
				message(WARNING
					"No CUDA device found for 'native' detection "
					"(detected value: '${CMAKE_CUDA_ARCHITECTURES_NATIVE}'). "
					"Falling back to architecture(s): ${_CUDA_ARCH_FALLBACK}. "
					"Set -DGENCODE=... to choose explicitly.")
			endif()
		else()
			message(STATUS "Building for CUDA architecture(s): ${CMAKE_CUDA_ARCHITECTURES}")
		endif()

		if(WIN32)
		  add_compile_options(
			  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
			)
		else()
		endif()

        find_package(CUDAToolkit)

    else()
        message(FATAL_ERROR "CUDA was enabled but not found.")
    endif()
else()
	message(STATUS "CUDA was not enabled.")
endif()
