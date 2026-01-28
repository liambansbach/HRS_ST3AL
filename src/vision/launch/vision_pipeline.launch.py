from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():

    delayed_nodes = TimerAction(
        period=1.0,   # Sekunden Verzögerung
        actions=[

            Node(
                package='vision',
                executable='camera_sub',
                name='camera_sub',
                output='screen'
            ),

            Node(
                package='vision',
                executable='detect_cubes_simple',
                name='detect_cubes_simple',
                output='screen'
            ),

            Node(
                package='vision',
                executable='identify_workspace',
                name='identify_workspace',
                output='screen'
            ),

        ]
    )

    return LaunchDescription([
        delayed_nodes
    ])
