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

import os
from ament_index_python.packages import get_package_share_directory
import yaml


def load_yaml_file(filename) -> dict:
    """Load yaml file with the OMP Parameters"""
    with open(filename, 'r', encoding='UTF-8') as file:
        data = yaml.safe_load(file)
    return data


def get_ros_parameters(node_name):
    """Get the ROS2 parameters from the yaml file

    Returns
    -------
    dict
        ROS2 parameters
    list
        Declared parameters

    """
    # Get the parameters from the yaml file
    config_file = os.path.join(
        get_package_share_directory("manipulator_trajectory_vision_gui"),
        'config',
        'params.yaml'
    )
    config = load_yaml_file(config_file)
    ros_parameters = config[node_name]["ros__parameters"]

    # Declare the parameters in the ROS2 parameter server
    declared_parameters = []
    for key, value in ros_parameters.items():
        declared_parameters.append((key, value))
    return ros_parameters, declared_parameters
