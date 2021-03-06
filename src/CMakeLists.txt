cmake_minimum_required(VERSION 2.8)

# Set variables
set(PROJECT_NAME cnn)
set(PROJECT_LANGUAGE C)
if(${WITH_CUDA})
    set(PROJECT_LANGUAGE ${PROJECT_LANGUAGE} CUDA)
endif()

set(PROJECT_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/cnn.h)
set(PROJECT_DEPS ${OpenBLAS_LIB} ${LIBXML2_LIBRARIES})

set(CNN_CONV_BIAS_TYPES "None;Filter;Layer" CACHE STRING "CNN convolution bias types" FORCE)
if(DEFINED CNN_CONV_BIAS_TYPE)
    set_property(CACHE CNN_CONV_BIAS_TYPE PROPERTY STRINGS ${CNN_CONV_BIAS_TYPES})
endif()
if(NOT CNN_CONV_BIAS_TYPE)
    set(CNN_CONV_BIAS_TYPE "Filter" CACHE STRING "CNN convolution bias type")
endif()

# Export configuration files
if(CNN_CONV_BIAS_TYPE MATCHES "Filter")
    set(CNN_CONV_BIAS_FILTER 1)
elseif(CNN_CONV_BIAS_TYPE MATCHES "Layer")
    set(CNN_CONV_BIAS_LAYER 1)
endif()

if(${WITH_CUDA})
    set(CNN_WITH_CUDA 1)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cnn_config.h.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cnn_config.h
    )

# Compile setting
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W3")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_C_FLAGS_RELEASE "-O3")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O")
endif()

if(${WITH_CUDA})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Set default build option
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF)
endif()

# Set project
project(${PROJECT_NAME} ${PROJECT_LANGUAGE})

# Add definition
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DDEBUG)
endif()

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set file list
file(GLOB PROJECT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    )

if(${WITH_CUDA})
    file(GLOB PROJECT_CUDA_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        )

    set(PROJECT_SRCS ${PROJECT_SRCS} ${PROJECT_CUDA_SRCS})
endif()

# Build library
add_library(${PROJECT_NAME} ${PROJECT_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES
    C_STANDARD 99
    OUTPUT_NAME ${PROJECT_NAME}
    PREFIX "lib"
    )

# Install
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
    )
install(FILES ${PROJECT_HEADERS}
    DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
    )
