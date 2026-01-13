import os
import sys
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    """ 
        Uses a virtual environment to launch the mp_pose node as it uses mediapipe which is not available in ROS2.
        Script finds the virtual environment (as long as it is named .venv) relative to the root of the project.
    """
    workspace_root = os.getcwd() 
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    venv_site_packages = os.path.join(
        workspace_root, 
        'src', 
        '.venv', 
        'lib', 
        py_version, 
        'site-packages'
    )

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