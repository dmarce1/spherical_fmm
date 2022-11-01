cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(sfmm CXX C Fortran)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
option(SFMM_WITH_CUDA "Enable CUDA" ON)
option(SFMM_DOUBLE "Enable Double Precision" OFF)
option(SFMM_FLOAT "Enable Single Precsion" OFF)
option(SFMM_CUDA_DOUBLE "Enable CUDA Double Precision" OFF)
option(SFMM_CUDA_FLOAT "Enable CUDA Single Precision" OFF)
option(SFMM_VEC_DOUBLE "Enable CUDA SIMD Double Precision" ON)
option(SFMM_VEC_FLOAT "Enable CUDA SIMD Single Precision" OFF)
set(SFMM_PMIN "3" CACHE STRING "pmin")
set(SFMM_PMAX "7" CACHE STRING "pmax")
set(SFMM_VEC_DOUBLE_SIZE "4" CACHE STRING "width of double precision simd vectors")
set(SFMM_VEC_FLOAT_SIZE "8" CACHE STRING "width of single precision simd vectors")
if( SFMM_WITH_CUDA ) 
	find_package(CUDA)
	enable_language(CUDA)
	set(CUDA_PROPAGATE_HOST_FLAGS FALSE)
	execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
	execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
	message(STATUS "CUDA Architecture: ${CUDA_ARCH}")
endif()
message( STATUS "Pmin = ${SFMM_PMIN}" )
message( STATUS "Pmax = ${SFMM_PMAX}" )


configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/src/codegen.cpp
)

execute_process(
	COMMAND rm -rf  ./generated_code
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
	

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
FILE(GLOB GENERATED_HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/*)


if( SFMM_FLOAT ) 
	execute_process(
  		COMMAND ${CMAKE_CXX_COMPILER} -O3 -DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\" -DVEC_FLOAT_SIZE=${SFMM_VEC_FLOAT_SIZE} -DVEC_DOUBLE_SIZE=${SFMM_VEC_DOUBLE_SIZE} -DFLOAT -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX}  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp -o sfmm_codegen
  		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  	)
execute_process(
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sfmm_codegen
)
endif()

if( SFMM_DOUBLE ) 
	execute_process(
  		COMMAND ${CMAKE_CXX_COMPILER} -O3 -DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"  -DVEC_FLOAT_SIZE=${SFMM_VEC_FLOAT_SIZE} -DVEC_DOUBLE_SIZE=${SFMM_VEC_DOUBLE_SIZE} -DDOUBLE -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX}  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp -o sfmm_codegen
  		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)
execute_process(
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sfmm_codegen
)
endif()

if( SFMM_VEC_FLOAT ) 
	execute_process(
  		COMMAND ${CMAKE_CXX_COMPILER} -O3 -DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"  -DVEC_FLOAT_SIZE=${SFMM_VEC_FLOAT_SIZE} -DVEC_DOUBLE_SIZE=${SFMM_VEC_DOUBLE_SIZE} -DVEC_FLOAT -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX}  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp -o sfmm_codegen
  		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)
execute_process(
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sfmm_codegen
)
endif()

if( SFMM_VEC_DOUBLE ) 
	execute_process(
  		COMMAND ${CMAKE_CXX_COMPILER} -O3  -DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\" -DVEC_FLOAT_SIZE=${SFMM_VEC_FLOAT_SIZE} -DVEC_DOUBLE_SIZE=${SFMM_VEC_DOUBLE_SIZE} -DVEC_DOUBLE -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX}  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp -o sfmm_codegen
  		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
   	    RESULT_VARIABLE RETURN_VALUE
  	)
	if (NOT RETURN_VALUE EQUAL 0)
    	message(FATAL_ERROR "Failed to build codegen: ${CODEGEN_OUT}")
	endif()
	execute_process(
    	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sfmm_codegen 
    	RESULT_VARIABLE RETURN_VALUE
  	)
	if (NOT RETURN_VALUE EQUAL 0)
    	message(FATAL_ERROR "Failed to execute codegen: ${CODEGEN_OUT}")
	endif()
endif()

if( SFMM_CUDA_FLOAT ) 
	execute_process(
  		COMMAND ${CMAKE_CXX_COMPILER} -O3 -DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\" -DVEC_FLOAT_SIZE=${SFMM_VEC_FLOAT_SIZE} -DVEC_DOUBLE_SIZE=${SFMM_VEC_DOUBLE_SIZE}  -DCUDA_FLOAT -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX}  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp -o sfmm_codegen
  		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)
execute_process(
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sfmm_codegen
)
endif()
 
if( SFMM_CUDA_DOUBLE ) 
	execute_process(
  		COMMAND ${CMAKE_CXX_COMPILER} -O3  -DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\" -DVEC_FLOAT_SIZE=${SFMM_VEC_FLOAT_SIZE} -DVEC_DOUBLE_SIZE=${SFMM_VEC_DOUBLE_SIZE} -DCUDA_DOUBLE -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX}  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp -o sfmm_codegen
  		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)
execute_process(
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sfmm_codegen
)
endif()



if( SFMM_FLOAT ) 
	FILE(GLOB FLOAT_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/float/*/*.cpp)
	add_library(sfmmf ${FLOAT_FILES} ./generated_code/src/math/math_float.cpp ${GENERATED_HEADER_FILES})
	target_include_directories(sfmmf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
	target_compile_options(sfmmf PUBLIC -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} )
#	add_executable(test_float ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
#	target_include_directories(test_float PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
#	target_link_libraries(test_float PUBLIC sfmmf)
#	target_compile_options(test_float PUBLIC -march=native  -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} -DTEST_TYPE_FLOAT)
endif()

if( SFMM_DOUBLE ) 
	FILE(GLOB DOUBLE_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/double/*/*.cpp)
	add_library(sfmmd ${DOUBLE_FILES} ./generated_code/src/math/math_double.cpp ${GENERATED_HEADER_FILES})
	target_include_directories(sfmmd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
	target_compile_options(sfmmd PUBLIC -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} )
#	add_executable(test_double ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
#	target_include_directories(test_double PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
#	target_link_libraries(test_double PUBLIC sfmmd)
#	target_compile_options(test_double PUBLIC -march=native  -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} -DTEST_TYPE_DOUBLE)
endif()

if( SFMM_VEC_FLOAT ) 
	FILE(GLOB VEC_FLOAT_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/v*sf/*/*.cpp)
	add_library(sfmmv${SFMM_VEC_FLOAT_SIZE}sf ${VEC_FLOAT_FILES} ./generated_code/src/math/math_vec_float.cpp ${GENERATED_HEADER_FILES})
	target_include_directories(sfmmv${SFMM_VEC_FLOAT_SIZE}sf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
	target_compile_options(sfmmv${SFMM_VEC_FLOAT_SIZE}sf PUBLIC -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} )
#	add_executable(test_vec_float ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
#	target_include_directories(test_vec_float PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
#	target_link_libraries(test_vec_float PUBLIC sfmmv${SFMM_VEC_FLOAT_SIZE}sf)
#	target_compile_options(test_vec_float PUBLIC -march=native  -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} -DTEST_TYPE_VEC_FLOAT)
endif()

if( SFMM_VEC_DOUBLE ) 
	FILE(GLOB VEC_DOUBLE_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/v*df/*/*.cpp)
	add_library(sfmmv${SFMM_VEC_DOUBLE_SIZE}df ${VEC_DOUBLE_FILES} ./generated_code/src/math/math_vec_double.cpp ${GENERATED_HEADER_FILES})
	target_include_directories(sfmmv${SFMM_VEC_DOUBLE_SIZE}df PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
	target_compile_options(sfmmv${SFMM_VEC_DOUBLE_SIZE}df PUBLIC -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} )
#	add_executable(test_vec_double ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cpp )
#	target_include_directories(test_vec_double PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
#	target_link_libraries(test_vec_double PUBLIC sfmmv${SFMM_VEC_DOUBLE_SIZE}df)
#	target_compile_options(test_vec_double PUBLIC -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX}  -DTEST_TYPE_VEC_DOUBLE)
endif()

if( SFMM_CUDA_FLOAT ) 
	FILE(GLOB CUDA_FLOAT_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/float_cuda/*/*.cu)
	add_library(cusfmmf ${CUDA_FLOAT_FILES} ./generated_code/src/math/math_float.cu ${GENERATED_HEADER_FILES})
	target_include_directories(cusfmmf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
	target_compile_options(cusfmmf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} --use_fast_math --gpu-architecture=sm_${CUDA_ARCH};
                        >)
#	set_property(TARGET cusfmmf PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()
 
if( SFMM_CUDA_DOUBLE ) 
	FILE(GLOB CUDA_DOUBLE_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/double_cuda/*/*.cu )
	add_library(cusfmmd ${CUDA_DOUBLE_FILES} ./generated_code/src/math/math_double.cu ${GENERATED_HEADER_FILES})
	target_include_directories(cusfmmd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
	target_compile_options(cusfmmd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} --use_fast_math --gpu-architecture=sm_${CUDA_ARCH};
                        >)
#	set_property(TARGET cusfmmd PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()



add_executable(sfmm_treecode ${CMAKE_CURRENT_SOURCE_DIR}/src/tree_code.cpp )
target_include_directories(sfmm_treecode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(sfmm_treecode PUBLIC sfmmf sfmmd)
target_compile_options(sfmm_treecode PUBLIC -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} )
