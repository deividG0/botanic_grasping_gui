# MIT License

# Copyright (c) 2021 DaeJong Jin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from collections import OrderedDict
from pykin.kinematics import jacobian as jac
from pykin.utils import transform_utils as t_utils
from pykin.utils.kin_utils import \
      calc_pose_error, convert_thetas_to_dict, logging_time


class Kinematics:
    """
    Class of Kinematics

    Args:
        robot_name (str): robot's name
        offset (Transform): robot's offset
        active_joint_names (list): robot's actuated joints
        base_name (str): reference link's name
        eef_name (str): end effector's name
    """

    def __init__(
        self,
        robot_name,
        offset,
        active_joint_names=[],
        base_name="base",
        eef_name=None,
    ):
        self.robot_name = robot_name
        self.offset = offset
        self.active_joint_names = active_joint_names
        self.base_name = base_name
        self.eef_name = eef_name

    def forward_kinematics(self, frames, thetas):
        """
        Returns transformations obtained by computing fk

        Args:
            frames (list or Frame()): robot's frame for forward kinematics
            thetas (sequence of float): input joint angles

        Returns:
            fk (OrderedDict): transformations
        """

        if not isinstance(frames, list):
            thetas = convert_thetas_to_dict(self.active_joint_names, thetas)
        fk = self._compute_FK(frames, self.offset, thetas)
        return fk

    @logging_time
    def inverse_kinematics(
        self,
        frames,
        current_joints,
        target_pose,
        method="LM2",
        max_iter=1000,
        joint_limits=[],
    ):
        """
        Returns joint angles obtained by computing IK

        Args:
            frames (Frame()): robot's frame for invers kinematics
            current_joints (sequence of float): input joint angles
            target_pose (np.array): goal pose to achieve
            method (str): two methods to calculate IK
                (LM: Levenberg-marquardt, NR: Newton-raphson)
            max_iter (int): Maximum number of calculation iterations

        Returns:
            joints (np.array): target joint angles
        """
        print("my kinematics 1")
        if method == "LM_modified":
            joints = self._compute_IK_LM_modified(
                frames,
                current_joints,
                target_pose,
                max_iter=max_iter,
                joint_limits=joint_limits,
            )
        return joints

    def _compute_FK(self, frames, offset, thetas):
        """
        Computes forward kinematics

        Args:
            frames (list or Frame()): robot's frame for forward kinematics
            offset (Transform): robot's offset
            thetas (sequence of float): input joint angles

        Returns:
            fk (OrderedDict): transformations
        """
        fk = OrderedDict()
        if not isinstance(frames, list):
            trans = offset * frames.get_transform(
                thetas.get(frames.joint.name, 0.0)
                )
            fk[frames.link.name] = trans
            for child in frames.children:
                fk.update(self._compute_FK(child, trans, thetas))
        else:
            # To compute IK
            cnt = 0
            trans = offset
            for frame in frames:
                trans = trans * frame.get_transform(thetas[cnt])
                fk[frame.link.name] = trans

                if frame.joint.dtype != "fixed":
                    cnt += 1

                if cnt >= len(thetas):
                    cnt -= 1

        return fk

    def _compute_IK_LM_modified(
        self, frames, current_joints, target_pose, max_iter, joint_limits
    ):
        """
        Computes inverse kinematics using Levenberg-Marquatdt method

        Args:
            frames (list or Frame()): robot's frame for inverse kinematics
            current_joints (sequence of float): input joint angles
            target_pose (np.array): goal pose to achieve
            max_iter (int): Maximum number of calculation iterations
            joint_limits (sequence of tuple):
                list of (min, max) pairs for each joint

        Returns:
            joints (np.array): target joint angles
        """
        print("my kinematics 2")
        iterator = 1
        EPS = float(1e-12)
        dof = len(current_joints)
        wn_pos = 1 / 0.3
        wn_ang = 1 / (2 * np.pi)
        We = np.diag([wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang])
        Wn = np.eye(dof)

        target_pose = t_utils.get_h_mat(target_pose[:3], target_pose[3:])

        cur_fk = self.forward_kinematics(frames, current_joints)
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

            cur_fk = self.forward_kinematics(frames, current_joints)
            cur_pose = list(cur_fk.values())[-1].h_mat
            err = calc_pose_error(target_pose, cur_pose, EPS)
            Ek2 = float(np.dot(np.dot(err.T, We), err)[0])

            if Ek2 < Ek:
                Ek = Ek2
            else:
                current_joints = [
                    current_joints[i] - dq[i] for i in range(dof)
                    ]
                cur_fk = self.forward_kinematics(frames, current_joints)
                break

        current_joints = np.array(
            [float(current_joint) for current_joint in current_joints]
        )
        return current_joints
