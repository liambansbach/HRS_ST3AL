from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Args
    rviz = LaunchConfiguration('rviz')
    gui = LaunchConfiguration('gui')

    # Package paths
    ainex_description = FindPackageShare('ainex_description')

    # Files
    urdf_xacro = PathJoinSubstitution([ainex_description, 'urdf', 'ainex.urdf.xacro'])
    rviz_config = PathJoinSubstitution([ainex_description, 'rviz', 'ainex.rviz'])

    # robot_description from xacro
    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        urdf_xacro
    ])

    # Joint state publishers:
    # - GUI only useful for sim/manual; for real robot keep gui:=false
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(gui),
        output='screen',
    )

    # IMPORTANT: bridges ainex_joint_states -> /joint_states
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        condition=UnlessCondition(gui),
        output='screen',
        parameters=[{'source_list': ['ainex_joint_states']}],
    )

    # Publishes TF tree from URDF using /joint_states
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    # Optional RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        condition=IfCondition(rviz),
        arguments=['-d', rviz_config],
        output='screen'
    )

    # Delay your nodes to ensure TF tree is alive
    delayed_nodes = TimerAction(
        period=0.5,
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
                executable='identify_workspace_fixed',
                name='identify_workspace_fixed',
                output='screen'
            ),
            Node(
                package='ainex_controller',
                executable='project_cubes_onto_workspace',
                name='project_cubes_onto_workspace',
                output='screen'
            ),
            Node(
                package='ainex_controller',
                executable='stack_cubes_fixed',
                name='stack_cubes_fixed',
                output='screen'
            ),
            # Node(
            #     package='ainex_controller',
            #     executable='ainex_hands_control_node',
            #     name='ainex_hands_control_node',
            #     output='screen'
            # ),
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Start RViz2 if true'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='false',
            description='Start joint_state_publisher_gui (usually false for real robot)'
        ),
        joint_state_publisher_gui,
        joint_state_publisher,
        robot_state_publisher,
        rviz_node,
        delayed_nodes
    ])
