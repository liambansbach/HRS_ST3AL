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
    
    # Add cmeel.prefix path for pinocchio
    cmeel_site_packages = os.path.join(
        venv_site_packages,
        'cmeel.prefix',
        'lib',
        py_version,
        'site-packages'
    )

    cmeel_lib_path = os.path.join(venv_site_packages, 'cmeel.prefix', 'lib')

    existing_python_path = os.environ.get('PYTHONPATH', '')
    existing_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')

    python_path_parts = [venv_site_packages, cmeel_site_packages]
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    new_python_path = os.pathsep.join(python_path_parts)

    ld_library_path_parts = [cmeel_lib_path]
    if existing_ld_library_path:
        ld_library_path_parts.append(existing_ld_library_path)
    new_ld_library_path = os.pathsep.join(ld_library_path_parts)

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
        additional_env={
            'PYTHONPATH': new_python_path,
            'LD_LIBRARY_PATH': new_ld_library_path,
        }
    )

    human_to_ainex_basis = Node(
        package='ainex_controller',
        executable='human_to_ainex_basis',
        output='screen',
        parameters=[{}],
    )

    ainex_imitation_control_node = Node(
        package='ainex_controller',
        executable='ainex_imitation_control_node',
        output='screen',
        parameters=[{}],
        additional_env={
            'PYTHONPATH': new_python_path,
            'LD_LIBRARY_PATH': new_ld_library_path,
        }
    )

    return LaunchDescription([
        camera_sub,           
        mp_pose,     
        human_to_ainex_basis,
        ainex_imitation_control_node,
    ])
