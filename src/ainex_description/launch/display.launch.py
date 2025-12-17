from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

def generate_launch_description():
    # Arguments
    gui = LaunchConfiguration('gui')
    source_list = LaunchConfiguration('list')

    # Find package paths
    ainex_description = FindPackageShare('ainex_description')

    # Paths to files
    urdf_xacro = PathJoinSubstitution([ainex_description, 'urdf', 'ainex.urdf.xacro'])
    rviz_config = PathJoinSubstitution([ainex_description, 'rviz', 'ainex.rviz'])

    # Use xacro to generate robot_description parameter
    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        urdf_xacro
    ]) 

    # Define nodes
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(gui),
        output='screen',
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        condition=UnlessCondition(gui),
        parameters=[{'source_list': ['ainex_joint_states']}]
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    # static_tf = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     name='static_tf_base_to_camera',
    #     # args: x y z qx qy qz qw frame_id child_frame_id
    #     arguments=['0', '0.0', '0.24', '0', '0', '0', '1', 'world', 'base_link'],
    #     output='screen'
    # )
    
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false', description='Start joint_state_publisher_gui if true'),
        DeclareLaunchArgument('list', default_value="['ainex_joint_states']", description='source_list for joint_state_publisher'),
        joint_state_publisher_gui,
        joint_state_publisher,
        robot_state_publisher,
        rviz,
        # static_tf
    ])
