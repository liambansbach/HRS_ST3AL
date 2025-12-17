cmake_minimum_required(VERSION 3.5)
project(ainex_description)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(DIRECTORY launch
  urdf
  meshes
  rviz
  #configs
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
