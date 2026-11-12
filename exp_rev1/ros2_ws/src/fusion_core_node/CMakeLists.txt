cmake_minimum_required(VERSION 3.8)
project(fusion_core_node)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# The optimized fusion core: int8 weights, sparse CSR arrays, -O3.
# Built as a static library so the node and the standalone benchmarks share
# exactly one implementation.
set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../common)

add_library(fusion_core STATIC ${CORE_DIR}/fusion_core.cpp)
target_include_directories(fusion_core PUBLIC ${CORE_DIR})
target_compile_options(fusion_core PRIVATE -O3 -march=armv8.2-a -fno-math-errno)
find_package(Threads REQUIRED)
target_link_libraries(fusion_core PUBLIC Threads::Threads)

add_executable(fusion_core_node src/fusion_core_node.cpp)
target_link_libraries(fusion_core_node fusion_core)
target_compile_options(fusion_core_node PRIVATE -O3 -march=armv8.2-a)
ament_target_dependencies(fusion_core_node rclcpp sensor_msgs nav_msgs geometry_msgs std_msgs)

install(TARGETS fusion_core_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
