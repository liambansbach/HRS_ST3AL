import os

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    venv_site_packages = '/home/marius/Documents/TUM/Humanoid_Robotic_Systems/Project/HRS_ST3AL/src/.venv/lib/python3.12/site-packages'
    existing_python_path = os.environ.get('PYTHONPATH', '')
    new_python_path = f"{venv_site_packages}:{existing_python_path}"

    """ Defines all nodes to be launched """
    camera_sub = Node(
        package='vision',
        executable='camera_sub',
        output='screen',
        parameters=[{}],
    )

    mp_pose = Node(
        package='vision',
        executable='mp_pose',
        output='screen',
        parameters=[{}],
        additional_env={'PYTHONPATH': new_python_path}
    )

    return LaunchDescription([
        camera_sub,           
        mp_pose,      
    ])