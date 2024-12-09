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
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit


def generate_launch_description():
    # First node execution
    gazebo = ExecuteProcess(
            cmd=[
                    'gnome-terminal',
                    '--',
                    'ros2',
                    'launch',
                    'open_manipulator_pro_gazebo',
                    'simulation.launch.py'],
            output='screen'
        )

    omp = ExecuteProcess(
            cmd=[
                    'gnome-terminal',
                    '--',
                    'ros2',
                    'launch',
                    'manipulator_trajectory_vision_gui',
                    'omp_ros2.launch.py'],
            output='screen'
        )

    yolo_real = ExecuteProcess(
            cmd=[
                    'gnome-terminal',
                    '--',
                    'ros2',
                    'launch',
                    'manipulator_trajectory_vision_gui',
                    'detection_simulation.launch.py'],
            output='screen'
        )

    return LaunchDescription([
        gazebo,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gazebo,
                on_exit=[
                    omp
                ]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=omp,
                on_exit=[
                    yolo_real
                ]
            )
        )
    ])
