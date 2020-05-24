cmake_minimum_required(VERSION 2.8)

# Set variables
set(PROJECT_NAME cnn_test)
set(PROJECT_LANGUAGE C)
set(PROJECT_DEPS cnn ${CBLAS_LIBS} ${LIBXML2_LIBRARIES})

if(${WITH_CUDA})
    set(PROJECT_DEPS
        ${PROJECT_DEPS}
        cudnn
        CUDA::cublas
        CUDA::cudart
        )
endif()

if(UNIX)
    set(PROJECT_DEPS ${PROJECT_DEPS} m)
elseif(MINGW)
    set(PROJECT_DEPS ${PROJECT_DEPS} ws2_32)
endif()

# Compile setting
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W3")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_C_FLAGS_RELEASE "-O3")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O")
endif()

# Set project
project(${PROJECT_NAME} ${PROJECT_LANGUAGE})

# Set file list
file(GLOB PROJECT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    )

# Build test
foreach(TEST_FILE_PATH ${PROJECT_SRCS})
    # Get each file name
    get_filename_component(TEST_FILE_NAME ${TEST_FILE_PATH} NAME_WE)

    # Build executable
    add_executable(${TEST_FILE_NAME} ${TEST_FILE_PATH})
    set_target_properties(${TEST_FILE_NAME} PROPERTIES
        C_STANDARD 99
        OUTPUT_NAME ${TEST_FILE_NAME}
        )
    target_link_libraries(${TEST_FILE_NAME} ${PROJECT_DEPS})

    # Install
    install(TARGETS ${TEST_FILE_NAME}
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
        )
endforeach()

