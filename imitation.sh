#!/bin/bash

#rm -r /log /install /build
gnome-terminal -- bash -c "rm -r build install log; colcon build; touch $LOCKFILE; source install/setup.bash; ros2 launch ainex_controller upper_body_imitation.launch.py; exec bash"

while [ ! -f $LOCKFILE ]; do
    sleep 1
done

gnome-terminal -- bash -c "sleep 30;source install/setup.bash; ros2 launch ainex_description display.launch.py; exec bash"

gnome-terminal -- bash -c "sleep 30; source install/setup.bash; ros2 topic echo /robot_imitation_targets; exec bash"