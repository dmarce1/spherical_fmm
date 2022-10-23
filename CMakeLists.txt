cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(spherical_fmm CXX C Fortran)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
option(FMM_WITH_CUDA "Enable CUDA" ON)
#if( FMM_WITH_CUDA ) 
	enable_language(CUDA)
	set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
	execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
	execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
	message(STATUS "CUDA Architecture: ${CUDA_ARCH}")
#endif()


configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/src/codegen.cpp
)

execute_process(
  COMMAND ${CMAKE_CXX_COMPILER} -O3 ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
  RESULT_VARIABLE RETURN_VALUE
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
if (NOT RETURN_VALUE EQUAL 0)
    message(FATAL_ERROR "Failed to get the dependencies")
endif()
execute_process(
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/a.out
)
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
FILE(GLOB GENERATED_SRC_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*.cpp)
FILE(GLOB GENERATED_CUDA_SRC_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*.cu)
FILE(GLOB GENERATED_HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/*)
FILE(GLOB GENERATED_DETAIL_HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/detail/*)

add_library(spherical_fmm ${GENERATED_SRC_FILES} ${GENERATED_DETAIL_HEADER_FILES} ${GENERATED_HEADER_FILES})
target_include_directories(spherical_fmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(spherical_fmm PUBLIC "-march=native")
add_library(cuda_spherical_fmm ${GENERATED_CUDA_SRC_FILES} ${GENERATED_DETAIL_HEADER_FILES} ${GENERATED_HEADER_FILES})
target_include_directories(cuda_spherical_fmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(cuda_spherical_fmm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                      --use_fast_math --gpu-architecture=sm_${CUDA_ARCH};
                        >)
set_property(TARGET cuda_spherical_fmm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(test_float ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(test_float PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(test_float PUBLIC spherical_fmm)
target_compile_options(test_float PUBLIC -march=native -DTEST_TYPE_FLOAT)

add_executable(test_double ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(test_double PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(test_double PUBLIC spherical_fmm)
target_compile_options(test_double PUBLIC -march=native -DTEST_TYPE_DOUBLE)

add_executable(test_vec_float ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(test_vec_float PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(test_vec_float PUBLIC spherical_fmm)
target_compile_options(test_vec_float PUBLIC -march=native -DTEST_TYPE_VEC_FLOAT)

add_executable(test_vec_double ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(test_vec_double PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(test_vec_double PUBLIC spherical_fmm)
target_compile_options(test_vec_double PUBLIC -march=native -DTEST_TYPE_VEC_DOUBLE)


