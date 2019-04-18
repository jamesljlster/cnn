# Find CUDA
if(${WITH_CUDA})
    set(CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDA QUIET REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Find CBLAS
find_package(CBLAS REQUIRED)

# Find libxml2
find_package(LibXml2 REQUIRED)

# Include directories
include_directories(${DEPS_PATHS}
    ${CBLAS_INCLUDE_DIRS}
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
