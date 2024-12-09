#!/usr/bin/env python3

# Copyright (c) 2024, SENAI CIMATEC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rviz_config_file = os.path.join(
        get_package_share_directory('open_manipulator_pro_description'),
        'config',
        'omp_ros2.rviz'
    )

    omp_node = Node(
        package='manipulator_trajectory_vision_gui',
        name='omp_ros2',
        executable='omp_ros2'
    )

    gripper_traj_server = Node(
        package='manipulator_trajectory_vision_gui',
        name='gripper_joint_trajectory',
        executable='gripper_joint_trajectory'
    )

    omp_joint_trajectory = Node(
        package='manipulator_trajectory_vision_gui',
        name='omp_joint_trajectory',
        executable='omp_joint_trajectory'
    )

    node_debug_pykin = Node(
        package='manipulator_trajectory_vision_gui',
        name='node_debug_pykin_real',
        executable='node_debug_pykin_real'
    )

    node_realsense_image_reader = Node(
        package='manipulator_trajectory_vision_gui',
        name='realsense_image_reader',
        executable='realsense_image_reader'
    )

    node_cam_calibrate = Node(
        package='manipulator_trajectory_vision_gui',
        name='camera_calibration_node',
        executable='camera_calibration_node'
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    nodes_to_start = [
        omp_node,
        omp_joint_trajectory,
        gripper_traj_server,
        node_debug_pykin,
        node_realsense_image_reader,
        node_cam_calibrate,
        rviz2_node
    ]
    return LaunchDescription(nodes_to_start)
