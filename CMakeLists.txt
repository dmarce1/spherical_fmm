cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(sfmm CXX C Fortran)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
option(SFMM_WITH_CUDA "Enable CUDA" ON)
set(SFMM_PMIN "3" CACHE STRING "pmin")
set(SFMM_PMAX "10" CACHE STRING "pmax")
if( SFMM_WITH_CUDA ) 
	enable_language(CUDA)
	set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
	execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
	execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
	message(STATUS "CUDA Architecture: ${CUDA_ARCH}")
endif()
message( STATUS "pmin = ${SFMM_PMIN}" )
message( STATUS "pmax = ${SFMM_PMAX}" )


configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/src/codegen.cpp
)

execute_process(
  COMMAND ${CMAKE_CXX_COMPILER} -O3  -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX}  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
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
FILE(GLOB FLOAT_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/float/*/*.cpp)
FILE(GLOB DOUBLE_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/double/*/*.cpp)
FILE(GLOB VEC_FLOAT_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/vec_float*/*/*.cpp)
FILE(GLOB VEC_DOUBLE_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/vec_double*/*/*.cpp)
FILE(GLOB CUDA_FLOAT_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/float/*/*.cu)
FILE(GLOB CUDA_DOUBLE_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/double/*/*.cu)
FILE(GLOB GENERATED_HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/*)

add_library(sfmmf ${FLOAT_FILES} ./generated_code/src/math/math_float.cpp ${GENERATED_HEADER_FILES})
target_include_directories(sfmmf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(sfmmf PUBLIC -march=native -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX} )

add_library(sfmmd ${DOUBLE_FILES} ./generated_code/src/math/math_double.cpp ${GENERATED_HEADER_FILES})
target_include_directories(sfmmd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(sfmmd PUBLIC -march=native -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX} )

add_library(sfmmvf ${VEC_FLOAT_FILES} ./generated_code/src/math/math_vec_float.cpp ${GENERATED_HEADER_FILES})
target_include_directories(sfmmvf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(sfmmvf PUBLIC -march=native -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX} )

add_library(sfmmvd ${VEC_DOUBLE_FILES} ./generated_code/src/math/math_vec_double.cpp ${GENERATED_HEADER_FILES})
target_include_directories(sfmmvd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(sfmmvd PUBLIC -march=native -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX} )

add_library(cusfmmf ${CUDA_FLOAT_FILES1} ./generated_code/src/math/math_float.cu ${GENERATED_HEADER_FILES})
target_include_directories(cusfmmf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(cusfmmf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX} --use_fast_math --gpu-architecture=sm_${CUDA_ARCH};
                        >)
set_property(TARGET cusfmmf PROPERTY CUDA_SEPARABLE_COMPILATION ON)
 
add_library(cusfmmd ${CUDA_FLOAT_FILES1} ./generated_code/src/math/math_double.cu ${GENERATED_HEADER_FILES})
target_include_directories(cusfmmd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(cusfmmd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX} --use_fast_math --gpu-architecture=sm_${CUDA_ARCH};
                        >)
set_property(TARGET cusfmmf PROPERTY CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(test_float ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(test_float PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(test_float PUBLIC sfmmf)
target_compile_options(test_float PUBLIC -march=native  -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX} -DTEST_TYPE_FLOAT)

add_executable(test_double ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(test_double PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(test_double PUBLIC sfmmd)
target_compile_options(test_double PUBLIC -march=native  -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX} -DTEST_TYPE_DOUBLE)

add_executable(test_vec_float ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(test_vec_float PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(test_vec_float PUBLIC sfmmvf)
target_compile_options(test_vec_float PUBLIC -march=native  -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX} -DTEST_TYPE_VEC_FLOAT)

add_executable(test_vec_double ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
target_include_directories(test_vec_double PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(test_vec_double PUBLIC sfmmvd)
target_compile_options(test_vec_double PUBLIC -march=native -DSFMM_PMIN=${SFMM_PMIN} -DSFMM_PMAX=${SFMM_PMAX}  -DTEST_TYPE_VEC_DOUBLE)


