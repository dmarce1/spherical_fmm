cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(sfmm CXX C CUDA )
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(SFMM_USE_DOUBLE "1" CACHE STRING "Enable Double Precision" )
set(SFMM_USE_FLOAT "1" CACHE STRING "Enable Single Precision")
set(SFMM_USE_SIMD "1" CACHE STRING "Enable SIMD")
set(SFMM_USE_CUDA "1" CACHE STRING "Enable CUDA")
set(SFMM_PMIN "7" CACHE STRING "pmin")
set(SFMM_PMAX "7" CACHE STRING "pmax")
set(SFMM_DOUBLE_SIMD_WIDTH "4" CACHE STRING "width of double precision simd vectors (except for m2m)")
set(SFMM_FLOAT_SIMD_WIDTH "8" CACHE STRING "width of single precision simd vectors (except for m2m)")
message( STATUS "Pmin = ${SFMM_PMIN}" )
message( STATUS "Pmax = ${SFMM_PMAX}" )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/src/codegen.cpp
)
message(STATUS "Compiling code generator")
execute_process(
  	COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp -march=native -DUSE_CUDA_FLAG=${SFMM_USE_CUDA} -DUSE_FLOAT_FLAG=${SFMM_USE_FLOAT} -DUSE_SIMD_FLAG=${SFMM_USE_SIMD} -DUSE_DOUBLE_FLAG=${SFMM_USE_DOUBLE} -DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\" -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} -DSIMD_FLOAT_WIDTH=${SFMM_FLOAT_SIMD_WIDTH} -DSIMD_DOUBLE_WIDTH=${SFMM_DOUBLE_SIMD_WIDTH} -o sfmm_codegen
  	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
  	RESULT_VARIABLE RESULT
)

if(NOT RESULT EQUAL 0)
	message(STATUS "error in compilng code generator")
endif()

message(STATUS "Cleaning old files")
execute_process(
	COMMAND rm -rf  ./generated_code
)

message(STATUS "Running code generator")
execute_process(
	COMMAND ./sfmm_codegen
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  	RESULT_VARIABLE RESULT
)

if(NOT RESULT EQUAL 0)
	message(STATUS "error in code generation")
endif()

message(STATUS "Code generation complete")

execute_process(COMMAND nvcc -lcuda ${PROJECT_SOURCE_DIR}/src/cuda_detect.cu -o cuda_detect)
execute_process(COMMAND ./cuda_detect OUTPUT_VARIABLE CUDA_ARCH)
message(STATUS "CUDA Architecture: ${CUDA_ARCH}")


FILE(GLOB HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/*)
FILE(GLOB SRC_FILES1 ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*.c )
FILE(GLOB SRC_FILES2 ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*.cpp )
FILE(GLOB SRC_FILES3 ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*/*.cpp )
FILE(GLOB SRC_FILES4 ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*/*.cu )
FILE(GLOB SRC_FILES5 ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/math/*.cpp)
set( SRC_FILES ${SRC_FILES4} ${SRC_FILES1} ${SRC_FILES2} ${SRC_FILES3} ${SRC_FILES5} ) 

cuda_add_library(sfmm ${SRC_FILES} ${HEADER_FILES}
	OPTIONS
                       --generate-line-info
                       --use_fast_math
                       -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} 
                       --diag-suppress=177 --diag-suppress=550 
)
set_property(TARGET sfmm PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
set_property(TARGET sfmm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(sfmm PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/sfmm.hpp")
target_compile_options(sfmm PUBLIC -ffast-math -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX})
target_include_directories(sfmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)

#add_executable(sfmm_treecode ${CMAKE_CURRENT_SOURCE_DIR}/src/tree_code.cpp )
#target_include_directories(sfmm_treecode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
#target_link_libraries(sfmm_treecode PUBLIC sfmm pthread  simd )
#target_compile_options(sfmm_treecode PUBLIC -ffast-math -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX})

INSTALL(TARGETS sfmm
        PUBLIC_HEADER DESTINATION include
)
