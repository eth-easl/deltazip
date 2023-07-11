file(GLOB GDEFLATE_CPU_SOURCES *gdeflate_cpu*.cpp *gdeflate_cpu*.cu)

add_compile_definitions("THRUST_CUB_WRAPPED_NAMESPACE=nvcomp")
if(BUILD_GDEFLATE_CPU)
  # Add gdeflate CPU executables 
  foreach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
    # cut off suffixes
    string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
    string(REPLACE ".cu" "" EXAMPLE_NAME ${EXAMPLE_NAME})
    get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
    add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp::nvcomp CUDA::cudart)
    target_link_libraries(${BARE_NAME} PRIVATE nvcomp::nvcomp_gdeflate_cpu)
  endforeach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
endif()

# option (BUILD_GDS_EXAMPLE "Build NVCOMP + GDS example" OFF)
# if (BUILD_GDS_EXAMPLE)
#   add_executable (nvcomp_gds nvcomp_gds.cu)
#   target_link_libraries(nvcomp_gds PRIVATE nvcomp::nvcomp CUDA::cudart cufile)
# endif()


add_executable(compress compress.cu)
  target_link_libraries(compress PRIVATE nvcomp::nvcomp CUDA::cudart)
  target_link_libraries(compress PRIVATE ZLIB::ZLIB)