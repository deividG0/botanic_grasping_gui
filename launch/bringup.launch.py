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
from launch.actions import \
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare a launch argument to choose the control launch option
    type_launch_arg = DeclareLaunchArgument(
        'type_launch',
        default_value='simulation',
        description='Specify the type of launch: "real" or "simulation".'
    )

    # Get the value of the control launch argument
    type_launch_option = LaunchConfiguration('type_launch')

    # Resolve the paths to the launch files
    open_manipulator_control_path = get_package_share_directory(
        'open_manipulator_pro_control'
        )
    manipulator_vision_gui_path = get_package_share_directory(
        'manipulator_trajectory_vision_gui'
        )
    open_manipulator_gazebo_path = get_package_share_directory(
        'open_manipulator_pro_gazebo'
        )

    # Dynamixel control group
    real_group = GroupAction(actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    open_manipulator_control_path,
                    'launch',
                    'dynamixel_control.launch.py'
                    )
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    manipulator_vision_gui_path,
                    'launch',
                    'detection_real.launch.py'
                    )
            )
        ),
    ])

    # Alternative control group
    simulation_group = GroupAction(actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    open_manipulator_gazebo_path,
                    'launch',
                    'simulation.launch.py'
                    )
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    manipulator_vision_gui_path,
                    'launch',
                    'detection_simulation.launch.py'
                    )
            )
        ),
    ])

    # Common nodes
    omp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                manipulator_vision_gui_path,
                'launch',
                'omp_ros2.launch.py'
                )
        )
    )

    return LaunchDescription([
        type_launch_arg,
        # Dynamically select control group based on argument
        GroupAction(
            actions=[real_group],
            condition=IfCondition(
                PythonExpression([
                    "'", type_launch_option, "' == 'real'"
                    ])
            )
        ),
        GroupAction(
            actions=[simulation_group],
            condition=IfCondition(
                PythonExpression([
                    "'", type_launch_option, "' == 'simulation'"
                    ])
            )
        ),
        omp
    ])
