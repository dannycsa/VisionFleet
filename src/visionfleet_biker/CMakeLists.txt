cmake_minimum_required(VERSION 3.8)
project(visionfleet_biker)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/detection_node.py
  scripts/map_node.py
  scripts/gps_map_node.py
  scripts/fusion_node.py
  scripts/carla_observer_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)
ament_package() 