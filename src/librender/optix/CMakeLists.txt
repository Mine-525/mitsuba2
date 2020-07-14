# Compile optix kernel
enable_language(CUDA)

add_library(optix-rt-ptx OBJECT optix_rt.cu)
set_target_properties(optix-rt-ptx PROPERTIES
CUDA_PTX_COMPILATION ON
RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
# TODO: consider --use_fast_math;
COMPILE_OPTIONS "-D__CUDACC__;-O3;-gencode;arch=compute_${ENOKI_CUDA_COMPUTE_CAPABILITY},code=compute_${ENOKI_CUDA_COMPUTE_CAPABILITY};--relocatable-device-code=true"
)
target_include_directories(optix-rt-ptx PRIVATE
  ${MTS_OPTIX_PATH}/include
  ${CMAKE_SOURCE_DIR}/src/shapes/optix
)

set(TARGET_NEW_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/optix_rt.ptx)
add_custom_command(
  OUTPUT ${TARGET_NEW_LOCATION} DEPENDS optix-rt-ptx
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/optix-rt-ptx.dir/optix_rt.ptx ${TARGET_NEW_LOCATION}
)

# add_custom_target(optix-rt-ptx-mv DEPENDS ${TARGET_NEW_LOCATION})
set(MITSUBA_DIST_OUT ${MITSUBA_DIST_OUT} ${TARGET_NEW_LOCATION} PARENT_SCOPE)