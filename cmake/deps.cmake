# Find CUDA
if(${WITH_CUDA})
    set(CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDA QUIET REQUIRED)
endif()

# Find OpenBlas
find_path(OpenBLAS_INCLUDE_DIR cblas.h
    "/usr/include"
    "/usr/local/include"
    "/opt/OpenBLAS/include"
    )

find_library(OpenBLAS_LIB openblas
    "/usr/lib"
    "/usr/local/lib"
    "/opt/OpenBLAS/lib"
    )

if(NOT OpenBLAS_INCLUDE_DIR OR NOT OpenBLAS_LIB)
    message(FATAL "Failed to find OpenBlas")
endif()

# Find libxml2
find_package(LibXml2 REQUIRED)

# Include directories
include_directories(${DEPS_PATHS}
    ${OpenBLAS_INCLUDE_DIR}
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

# Add subdirectories
foreach(DEPS_PATH ${DEPS_PATHS})
    add_subdirectory(${DEPS_PATH})
endforeach()
