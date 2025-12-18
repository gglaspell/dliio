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

find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)

# --- START OF FIX 1 ---
# Find the specific PCL components we need
find_package(PCL REQUIRED COMPONENTS common registration)
# --- END OF FIX 1 ---

find_package( Eigen3 REQUIRED )
find_package(Threads REQUIRED)
find_package(OpenMP)


set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

rosidl_generate_interfaces(
    ${PROJECT_NAME}
        srv/SavePCD.srv
    DEPENDENCIES
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)
include_directories(SYSTEM)


# NanoFLANN
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
# --- START OF FIX 2.1 ---
# Use modern CMake targets for linking
target_link_libraries(nanoflann PUBLIC PCL::common)
# --- END OF FIX 2.1 ---
ament_target_dependencies(nanoflann)


# NanoGICP
add_library(nano_gicp STATIC
  # lsq_registration is not used by the new NanoGICP
  # src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
# --- START OF FIX 2.2 ---
# Use modern CMake targets for linking. PUBLIC propagates the dependency.
target_link_libraries(nano_gicp PUBLIC PCL::registration ${OpenMP_LIBS} nanoflann)
# --- END OF FIX 2.2 ---
ament_target_dependencies(nano_gicp)


# Odometry Node
add_executable(dlio_odom_node src/dlio/odom_node.cc src/dlio/odom.cc)
# --- START OF FIX 2.3 ---
# Link the executable against the required PCL targets directly.
target_link_libraries(dlio_odom_node PRIVATE
  PCL::common
  PCL::registration
  ${OpenMP_LIBS}
  Threads::Threads
  nano_gicp
)
# --- END OF FIX 2.3 ---
ament_target_dependencies(dlio_odom_node "tf2_ros" "pcl_ros" "nav_msgs" "pcl_conversions" "rclcpp" "sensor_msgs" "geometry_msgs")


# Mapping Node
add_executable (dlio_map_node src/dlio/map_node.cc src/dlio/map.cc)
# --- START OF FIX 2.4 ---
# Also update the map node for consistency
target_link_libraries(dlio_map_node PRIVATE PCL::common PCL::registration ${cpp_typesupport_target})
# --- END OF FIX 2.4 ---
ament_target_dependencies(dlio_map_node "tf2_ros" "pcl_ros" "pcl_conversions" "rclcpp")


ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME} nano_gicp nanoflann)
ament_export_dependencies(rclcpp std_msgs sensor_msgs geometry_msgs pcl_ros)

ament_package()


# Binaries
install( TARGETS dlio_odom_node dlio_map_node
         DESTINATION lib/${PROJECT_NAME}
)

install( DIRECTORY cfg launch
         DESTINATION share/${PROJECT_NAME}
)

# Use C++17
add_compile_options(-std=c++17)

link_directories(${PCL_LIBRARY_DIRS})

if(OPENMP_FOUND)
  target_compile_options(dlio_odom_node PRIVATE ${OpenMP_CXX_FLAGS})
  target_compile_options(dlio_map_node PRIVATE ${OpenMP_CXX_FLAGS})
  target_compile_options(nano_gicp PUBLIC ${OpenMP_CXX_FLAGS})
endif()

file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
