from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    image_rgb_yolo_real_node = Node(
        package='manipulator_trajectory_vision_gui',
        name='image_rgb_yolo_real',
        executable='image_rgb_yolo_real'
    )

    nodes_to_start = [
        image_rgb_yolo_real_node,
    ]
    return LaunchDescription(nodes_to_start)
