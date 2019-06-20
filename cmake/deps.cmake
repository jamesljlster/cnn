# Find CUDA
if(${WITH_CUDA})
    set(CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDA QUIET REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Find CBLAS
find_package(CBLAS QUIET)
if(NOT ${CBLAS_FOUND})

    # Find OpenBLAS instead
    set(BLAS_VENDOR "OpenBLAS")
    find_package(BLAS QUIET)

    # Find cblas.h
    find_path(CBLAS_INCDIRS cblas.h
        "/usr/include"
        "/usr/local/include"
        "/opt/OpenBLAS/include"
        )

    # Check result
    if(NOT CBLAS_INCDIRS OR NOT ${BLAS_FOUND})
        message(FATAL "Failed to find CBLAS")
    else()
        set(CBLAS_LIBS ${BLAS_LIBRARIES} CACHE PATH "CBLAS library path")
    endif()

else()
    set(CBLAS_LIBS ${CBLAS_LIBRARIES} CACHE PATH "CBLAS library path")
    set(CBLAS_INCDIRS ${CBLAS_INCLUDE_DIRS} CACHE PATH "CBLAS header path")
endif()

# Find libxml2
find_package(LibXml2 REQUIRED)

# Include directories
include_directories(${DEPS_PATHS}
    ${CBLAS_INCDIRS}
    ${LIBXML2_INCLUDE_DIR}
    )

# Find check
#if(BUILD_TEST)
#	find_path(Check_INCLUDE_DIR check.h
#		"/usr/include"
#		"/usr/local/include"
#		)
#
#	find_library(Check_LIB check
#		"/usr/lib"
#		"/usr/local/lib"
#		)
#
#	include_directories(${Check_INCLUDE_DIR})
#endif()

# Find OpenMP
find_package(OpenMP QUIET)
if(${OpenMP_FOUND})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add subdirectories
foreach(DEPS_PATH ${DEPS_PATHS})
    add_subdirectory(${DEPS_PATH})
endforeach()
