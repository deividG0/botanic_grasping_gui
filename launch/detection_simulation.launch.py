from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    image_rgb_yolo_node = Node(
        package='manipulator_trajectory_vision_gui',
        name='image_rgb_yolo',
        executable='image_rgb_yolo'
    )

    nodes_to_start = [
        image_rgb_yolo_node,
    ]
    return LaunchDescription(nodes_to_start)
