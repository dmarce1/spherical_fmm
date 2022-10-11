cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(spherical_fmm CXX C Fortran)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
option(FMM_WITH_CUDA "Enable CUDA" ON)
if( FMM_WITH_CUDA ) 
	enable_language(CUDA)
	set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
	execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
	execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
	message(STATUS "CUDA Architecture: ${CUDA_ARCH}")
endif()


configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/src/codegen.cpp
)

execute_process(
  COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
  RESULT_VARIABLE RETURN_VALUE
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
if (NOT RETURN_VALUE EQUAL 0)
    message(FATAL_ERROR "Failed to get the dependencies")
endif()
execute_process(
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/a.out
)

FILE(GLOB GENERATED_SRC_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*)
FILE(GLOB GENERATED_HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/*)
FILE(GLOB GENERATED_DETAIL_HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/detail/*)

add_library(spherical_fmm ${GENERATED_SRC_FILES} ${GENERATED_DETAIL_HEADER_FILES} ${GENERATED_HEADER_FILES})
target_include_directories(spherical_fmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)


add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(test PUBLIC spherical_fmm)


