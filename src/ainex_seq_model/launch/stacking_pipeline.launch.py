from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():

    delayed_nodes = TimerAction(
        period=1.0,   # Sekunden Verzögerung
        actions=[

            Node(
                package='ainex_seq_model',
                executable='call_server',
                name='call_server',
                output='screen'
            ),

            Node(
                package='ainex_seq_model',
                executable='sequence_model',
                name='sequence_model',
                output='screen'
            ),
        ]
    )

    return LaunchDescription([
        delayed_nodes
    ])
