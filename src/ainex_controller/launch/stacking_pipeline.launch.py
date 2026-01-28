from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():

    delayed_nodes = TimerAction(
        period=1.0,   # Sekunden Verzögerung
        actions=[

            Node(
                package='ainex_controller',
                executable='project_cubes_onto_workspace',
                name='project_cubes_onto_workspace',
                output='screen'
            ),

            Node(
                package='ainex_controller',
                executable='stack_cubes',
                name='stack_cubes',
                output='screen'
            ),
        ]
    )

    return LaunchDescription([
        delayed_nodes
    ])
