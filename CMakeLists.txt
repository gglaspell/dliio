###########################################################
#                                                         #
# Copyright (c)                                           #
#                                                         #
# The Verifiable & Control-Theoretic Robotics (VECTR) Lab #
# University of California, Los Angeles                   #
#                                                         #
# Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   #
# Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         #
#                                                         #
###########################################################

cmake_minimum_required(VERSION 3.12.4)
project(direct_lidar_inertial_odometry)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Find packages first
find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io registration kdtree search features surface filters)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# OpenMP
include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  set(OpenMP_LIBS gomp)
else()
  message(WARNING "OpenMP could not be found.")
endif()

# CPUID test
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)

if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()

# ROS IDL
rosidl_generate_interfaces(
    ${PROJECT_NAME}
        srv/SavePCD.srv
    DEPENDENCIES
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

# Include directories
include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# NanoFLANN
add_library(nanoflann STATIC src/nano_gicp/nanoflann.cc)
target_compile_definitions(nanoflann PRIVATE PCL_NO_PRECOMPILE)
target_include_directories(nanoflann PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(nanoflann PUBLIC ${PCL_LIBRARIES})

# NanoGICP
add_library(nano_gicp STATIC 
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
target_compile_definitions(nano_gicp PRIVATE PCL_NO_PRECOMPILE)
target_include_directories(nano_gicp PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(nano_gicp PUBLIC 
  nanoflann 
  ${PCL_LIBRARIES}
  ${OpenMP_LIBS}
)

# Odometry Node
add_executable(dlio_odom_node src/dlio/odom_node.cc src/dlio/odom.cc)
target_compile_definitions(dlio_odom_node PRIVATE PCL_NO_PRECOMPILE)
target_compile_options(dlio_odom_node PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(dlio_odom_node 
  nano_gicp
  Threads::Threads
  ${PCL_LIBRARIES}
)
ament_target_dependencies(dlio_odom_node 
  tf2_ros 
  pcl_ros 
  nav_msgs 
  pcl_conversions 
  rclcpp 
  sensor_msgs 
  geometry_msgs
)

# Mapping Node
add_executable(dlio_map_node src/dlio/map_node.cc src/dlio/map.cc)
target_compile_definitions(dlio_map_node PRIVATE PCL_NO_PRECOMPILE)
target_compile_options(dlio_map_node PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(dlio_map_node 
  ${PCL_LIBRARIES}
  ${cpp_typesupport_target}
)
ament_target_dependencies(dlio_map_node 
  tf2_ros 
  pcl_ros 
  pcl_conversions 
  rclcpp
)

# Export
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME} nano_gicp nanoflann)
ament_export_dependencies(rclcpp std_msgs sensor_msgs geometry_msgs pcl_ros)

# Install
install(TARGETS dlio_odom_node dlio_map_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY cfg launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
