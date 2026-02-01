#!/bin/bash

#rm -r /log /install /build
gnome-terminal -- bash -c "colcon build --packages-select ainex_controller; source install/setup.bash; ros2 launch bringup upper_body_imitation.launch.py; exec bash"

