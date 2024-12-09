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
from pykin.robots.single_arm import SingleArm
from pykin.kinematics import jacobian as jac
from pykin.utils import transform_utils as t_utils
from pykin.utils.kin_utils import calc_pose_error
from ament_index_python.packages import get_package_share_directory


class IKCalculatorOMP:
    def __init__(self, start="world", end="end_link"):
        # Default variables
        share_directory = get_package_share_directory(
            "manipulator_trajectory_vision_gui"
            )
        file_path = os.path.join(
            share_directory,
            "resource",
            "total_description",
            "open_manipulator_pro.urdf",
        )
        self.joint_limits = {
                'joint1': (-np.pi/2, np.pi/2),
                'joint2': (-np.pi/2, np.pi/2),
                'joint3': (-np.pi/2, 3*np.pi/4),
                'joint4': (-np.pi/2, np.pi/2),
                'joint5': (-np.pi/2, np.pi/2),
                'joint6': (-np.pi/2, np.pi/2)
                }
        self.robot = SingleArm(
            str(file_path)
        )
        self.robot.kin._compute_IK_LM = self.compute_IK_LM_modified
        self.robot.setup_link_name(start, end)

    def compute_IK_LM_modified(
            self, frames, current_joints, target_pose, max_iter,
            ):
        """Compute inverse kinematics using Levenberg-Marquatdt method."""
        iterator = 1
        EPS = float(1e-12)
        dof = len(current_joints)
        joint_limits = list(self.joint_limits.values())
        wn_pos = 1 / 0.3
        wn_ang = 1 / (2 * np.pi)
        We = np.diag([wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang])
        Wn = np.eye(dof)

        target_pose = t_utils.get_h_mat(target_pose[:3], target_pose[3:])

        cur_fk = self.robot.kin.forward_kinematics(frames, current_joints)
        cur_pose = list(cur_fk.values())[-1].h_mat

        err = calc_pose_error(target_pose, cur_pose, EPS)
        Ek = float(np.dot(np.dot(err.T, We), err)[0])

        while Ek > EPS:
            iterator += 1
            if iterator > max_iter:
                break

            lamb = Ek + 0.002

            J = jac.calc_jacobian(frames, cur_fk, len(current_joints))
            J_dls = np.dot(np.dot(J.T, We), J) + np.dot(Wn, lamb)

            gerr = np.dot(np.dot(J.T, We), err)
            dq = np.dot(np.linalg.inv(J_dls), gerr)
            current_joints = [current_joints[i] + dq[i] for i in range(dof)]

            # Constrain the joints within their limits
            for i in range(dof):
                min_limit, max_limit = joint_limits[i]
                current_joints[i] = max(
                    min(current_joints[i], max_limit), min_limit
                    )

            cur_fk = self.robot.kin.forward_kinematics(frames, current_joints)
            cur_pose = list(cur_fk.values())[-1].h_mat
            err = calc_pose_error(target_pose, cur_pose, EPS)
            Ek2 = float(np.dot(np.dot(err.T, We), err)[0])

            if Ek2 < Ek:
                Ek = Ek2
            else:
                current_joints = [
                    current_joints[i] - dq[i] for i in range(dof)
                    ]
                cur_fk = self.robot.kin.forward_kinematics(
                    frames, current_joints
                    )
                break

        current_joints = np.array(
            [float(current_joint) for current_joint in current_joints]
        )
        return current_joints

    def calculate_ik(
        self,
        pose,
        current_thetas=np.random.randn(7),
        max_iter=100,
        method="LM"
    ):
        pose = np.array(pose)
        current_thetas = np.array(current_thetas)

        ik_LM_result = self.robot.inverse_kin(
            current_joints=current_thetas,
            target_pose=pose,
            method=method,
            max_iter=max_iter
        )

        return ik_LM_result

    def calculate_fk(self, current_thetas):
        return self.get_tf(current_thetas)["end_link"]

    def get_tf(self, current_thetas):
        current_thetas = np.array(current_thetas)

        fk_result = self.robot.forward_kin(
            current_thetas,
        )

        return fk_result
