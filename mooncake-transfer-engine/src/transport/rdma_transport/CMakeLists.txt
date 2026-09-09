file(GLOB RDMA_SOURCES "*.cpp")

add_library(rdma_transport OBJECT ${RDMA_SOURCES})
target_link_libraries(rdma_transport PUBLIC gflags::gflags glog::glog
                                    PRIVATE JsonCpp::JsonCpp pthread)

if(MOONCAKE_ENABLE_ADAPTIVE_CC)
  target_link_libraries(rdma_transport PUBLIC adaptive_congestion_control_core)
endif()

if(USE_MLX5DV)
  target_compile_definitions(rdma_transport PRIVATE USE_MLX5DV)
endif()
