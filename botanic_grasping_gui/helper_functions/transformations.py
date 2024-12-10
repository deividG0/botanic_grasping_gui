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

from math import cos, sin

import numpy as np
import tf_transformations


def geometry_msg_pose_to_htm(geometry_msg):
    """Convert a geometry_msgs.transform to a homogeneous frame matrix."""
    position = geometry_msg.translation
    orientation = geometry_msg.rotation
    translation = np.array([position.x, position.y, position.z])
    rotation = np.array([orientation.x, orientation.y,
                         orientation.z, orientation.w])
    homogeneous_transformation = tf_transformations.quaternion_matrix(rotation)
    homogeneous_transformation[0:3, 3] = translation
    return homogeneous_transformation


def htm_rotation_around_x(angle: float) -> np.matrix:
    """Generate a rotation matrix around the x-axis."""
    return np.matrix([[1, 0, 0, 0],
                     [0, cos(angle), -sin(angle), 0],
                     [0, sin(angle), cos(angle), 0],
                     [0, 0, 0, 1]])


def htm_rotation_around_y(angle: float) -> np.matrix:
    """Generate a rotation matrix around the y-axis."""
    return np.matrix([[cos(angle), 0, sin(angle), 0],
                     [0, 1, 0, 0],
                     [-sin(angle), 0, cos(angle), 0],
                     [0, 0, 0, 1]])


def htm_rotation_around_z(angle: float) -> np.matrix:
    """Generate a rotation matrix around the z-axis."""
    return np.matrix([[cos(angle), -sin(angle), 0, 0],
                     [sin(angle), cos(angle), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def htm_translation(translation_vector: np.array) -> np.matrix:
    """Generate a homogeneous transformation matrix for translation."""
    return np.matrix([[1, 0, 0, translation_vector[0]],
                     [0, 1, 0, translation_vector[1]],
                     [0, 0, 1, translation_vector[2]],
                     [0, 0, 0, 1]])


def get_desired_pose_htm(
    position: np.array,
    roll: float,
    pitch: float,
    yaw: float
):
    """Compute the desired pose HTM for the end effector."""
    rot_x = htm_rotation_around_x(np.radians(roll))
    rot_y = htm_rotation_around_y(np.radians(pitch))
    rot_z = htm_rotation_around_z(np.radians(yaw))
    desired_pose = rot_z * rot_y * rot_x

    # position vector is the desired position of the end effector
    # in the base_link_inertia frame
    position_vector = position.reshape(3, 1)

    # set the desired pose to have the same position as the position vector
    desired_pose[:3, -1] = position_vector

    return desired_pose
