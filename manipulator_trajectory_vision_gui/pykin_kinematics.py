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
import numpy as np
from manipulator_trajectory_vision_gui.pykin_omp.single_arm import SingleArm
from ament_index_python.packages import get_package_share_directory


class IKCalculatorOMP:
    def __init__(self, start="world", end="end_link"):
        # Default variables
        share_directory = get_package_share_directory("manipulator_trajectory_vision_gui")
        file_path = os.path.join(
            share_directory,
            "resource",
            "total_description",
            "open_manipulator_pro.urdf",
        )
        self.robot = SingleArm(
            str(file_path)
        )
        self.robot.setup_link_name(start, end)

    def calculate_ik(
        self,
        pose,
        current_thetas=np.random.randn(7),
        max_iter=100,
        method="LM_modified",
        joint_limits=[],
    ):
        """
            Pose formato: [x, y, z, w, qx, qy, qz]
        """
        target_pose = np.array(pose)
        current_thetas = np.array(current_thetas)

        ik_LM_result = self.robot.inverse_kin(
            current_thetas,
            target_pose,
            method=method,
            max_iter=max_iter,
            joint_limits=joint_limits,
        )

        return ik_LM_result

    def calculate_fk(self, current_thetas):
        """
        Formato de retorno: [x, y, z]
        """
        return self.get_tf(current_thetas)["end_link"]

    def get_tf(self, current_thetas):
        """
        Formato de retorno: [x, y, z]
        """
        current_thetas = np.array(current_thetas)

        fk_result = self.robot.forward_kin(
            current_thetas,
        )

        return fk_result
