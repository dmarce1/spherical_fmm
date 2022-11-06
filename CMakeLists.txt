cmake_minimum_required(VERSION 3.12.4)
cmake_policy(VERSION 3.12.4)
project(sfmm CXX C)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(SFMM_USE_DOUBLE "0" CACHE STRING "Enable Double Precision" )
set(SFMM_USE_FLOAT "1" CACHE STRING "Enable Single Precision")
set(SFMM_USE_SIMD "1" CACHE STRING "Enable SIMD")
set(SFMM_PMIN "7" CACHE STRING "pmin")
set(SFMM_PMAX "8" CACHE STRING "pmax")
set(SFMM_DOUBLE_SIMD_WIDTH "4" CACHE STRING "width of float precision simd vectors (except for m2m)")
set(SFMM_FLOAT_SIMD_WIDTH "8" CACHE STRING "width of double precision simd vectors (except for m2m)")
set(SFMM_M2M_SIMD_WIDTH "2" CACHE STRING "width of m2m simd vectors")
message( STATUS "Pmin = ${SFMM_PMIN}" )
message( STATUS "Pmax = ${SFMM_PMAX}" )


configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/src/codegen.cpp
)
message(STATUS "Compiling code generator")
execute_process(
  	COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen.cpp -g -DUSE_FLOAT_FLAG=${SFMM_USE_FLOAT} -DUSE_SIMD_FLAG=${SFMM_USE_SIMD} -DUSE_DOUBLE_FLAG=${SFMM_USE_DOUBLE} -DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\" -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} -DFLOAT_SIMD_WIDTH=${SFMM_FLOAT_SIMD_WIDTH} -DDOUBLE_SIMD_WIDTH=${SFMM_DOUBLE_SIMD_WIDTH} -DM2M_SIMD_WIDTH=${SFMM_M2M_SIMD_WIDTH} -o sfmm_codegen
  	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
  	RESULT_VARIABLE RESULT
)

#add_compile_options(-D_GLIBCXX_DEBUG -fsanitize=address)
#add_link_options(-D_GLIBCXX_DEBUG -fsanitize=address)

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
	
FILE(GLOB GENERATED_HEADER_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include/*)
FILE(GLOB SRC4_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*.c )
FILE(GLOB SRC3_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*.cpp )
FILE(GLOB SRC1_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/*/*/*.cpp )
FILE(GLOB SRC2_FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_code/src/math/*.cpp)
add_library(sfmm ${SRC4_FILES} ${SRC3_FILES} ${SRC2_FILES} ${SRC1_FILES} ${GENERATED_HEADER_FILES})
target_include_directories(sfmm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_compile_options(sfmm PUBLIC -ffast-math -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX} )

add_executable(sfmm_treecode ${CMAKE_CURRENT_SOURCE_DIR}/src/tree_code.cpp )
target_include_directories(sfmm_treecode PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated_code/include)
target_link_libraries(sfmm_treecode PUBLIC sfmm pthread)
target_compile_options(sfmm_treecode PUBLIC -ffast-math -march=native -DPMIN=${SFMM_PMIN} -DPMAX=${SFMM_PMAX})
