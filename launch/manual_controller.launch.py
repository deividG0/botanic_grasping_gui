from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='manipulator_trajectory_vision_gui',
            executable='manipulator_trajectory_vision_gui',
            output='screen'),
    ])
