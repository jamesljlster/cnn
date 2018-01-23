cmake_minimum_required(VERSION 2.8)

# Set variables
set(PROJECT_NAME cnn_test)
set(PROJECT_LANGUAGE C)
set(PROJECT_DEPS cnn m)

# Compile setting
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_C_FLAGS_RELEASE "-O3")

# Variables
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Build confugure" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
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
	set_target_properties(${TEST_FILE_NAME}
		PROPERTIES OUTPUT_NAME ${TEST_FILE_NAME}
		)
	target_link_libraries(${TEST_FILE_NAME} ${PROJECT_DEPS})

	# Install
	install(TARGETS ${TEST_FILE_NAME}
		RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
		)
endforeach()

