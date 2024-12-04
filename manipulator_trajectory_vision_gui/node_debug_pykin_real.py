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

import math
import random
import numpy as np
import rclpy
from rclpy.node import Node
from .pykin_kinematics import IKCalculatorOMP
from builtin_interfaces.msg import Duration
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory


class DebugPykin(Node):
    def __init__(self):
        super().__init__("node_debug_pykin_real")
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        while not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Action server not available, waiting...")

        self.marker_namespace = "/marker"

        self.real_point = self.create_publisher(
            Marker, f"{self.marker_namespace}/real_point", 10
        )
        self.real_orientation = self.create_publisher(
            Marker, f"{self.marker_namespace}/real_orientation", 10
        )
        self.fk_point = self.create_publisher(
            Marker, f"{self.marker_namespace}/fk_point", 10
        )
        self.fk_rot = self.create_publisher(
            Marker, f"{self.marker_namespace}/fk_rot", 10
        )
        self.wall_marker = self.create_publisher(
            Marker, f"{self.marker_namespace}/wall", 10
        )
        self.valid_joint_config_found = False
        self.ik_calculator_omp = IKCalculatorOMP()

        self.time_period = 7
        self.create_timer(self.time_period, self.execute_trajectory)
        self.create_timer(0.1, self.define_point)

        self.joint_states = np.random.uniform(-np.pi / 3, np.pi / 3, 6)
        self.create_subscription(
            JointState, "/joint_states", self.update_joint_state, 10
        )

        self.mode = "off"
        self.create_subscription(String, "/switch_debug_mode", self.switch, 10)

        self.target_position = None
        self.target_orientation = [1.0, 0.0, 0.0, 0.0]
        self.joint_angles = [0.0, 0.0, 0.78, 1.57, 0.0, 0.0, 0.2]

        self.joint_limits = {
            "joint1": (-np.pi / 2, np.pi / 2),
            "joint2": (-0.25, 0.35),
            "joint3": (0.25, 1.05),
            "joint4": (-np.pi / 2, np.pi / 2),
            "joint5": (-np.pi / 2, np.pi / 2),
            "joint6": (-np.pi / 2, np.pi / 2),
        }

    def switch(self, msg):
        self.mode = msg.data

    def update_joint_state(self, msg):
        if msg.name[0] == "gripper":
            self.gripper_position = msg.position[0]
            self.joint_states[0] = msg.position[3]
            self.joint_states[1] = msg.position[1]
            self.joint_states[2] = msg.position[2]
            self.joint_states[3] = msg.position[4]
            self.joint_states[4] = msg.position[5]
            self.joint_states[5] = msg.position[6]
            self.joint_states = np.array(self.joint_states)
        else:
            self.joint_states[0] = msg.position[2]
            self.joint_states[1] = msg.position[0]
            self.joint_states[2] = msg.position[1]
            self.joint_states[3] = msg.position[3]
            self.joint_states[4] = msg.position[4]
            self.joint_states[5] = msg.position[5]
            self.gripper_position = msg.position[6]
            self.joint_states = np.array(self.joint_states)

    def get_random_point(self):
        operating_space_min = [
            0.35,
            -0.25,
            0.2,
        ]  # Replace with your min bound coordinates
        operating_space_max = [
            0.45,
            0.25,
            0.5,
        ]  # Replace with your max bound coordinates

        x = random.uniform(operating_space_min[0], operating_space_max[0])
        y = random.uniform(operating_space_min[1], operating_space_max[1])
        z = random.uniform(operating_space_min[2], operating_space_max[2])

        # Wall marker
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = (
            operating_space_min[0] + operating_space_max[0]
            ) / 2
        marker.pose.position.y = (
            operating_space_min[1] + operating_space_max[1]
            ) / 2
        marker.pose.position.z = (
            operating_space_min[2] + operating_space_max[2]
            ) / 2
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = (
            operating_space_max[0] - operating_space_min[0]
        )  # Length of the plane
        marker.scale.y = (
            operating_space_max[1] - operating_space_min[1]
        )  # Width of the plane
        marker.scale.z = (
            operating_space_max[2] - operating_space_min[2]
        )  # Height of the plane, making it very thin
        marker.color.a = 0.6
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        self.wall_marker.publish(marker)

        # Real point marker
        m1 = Marker()
        m1.header.frame_id = "world"  # Assuming the frame_id in RViz2
        m1.type = Marker.SPHERE
        m1.action = Marker.ADD
        m1.scale.x = 0.1
        m1.scale.y = 0.1
        m1.scale.z = 0.1
        m1.color.a = 0.5
        m1.color.r = 0.0
        m1.color.g = 1.0
        m1.color.b = 0.0
        m1.pose.position.x = x  # X coordinate of the point
        m1.pose.position.y = y  # Y coordinate of the point
        m1.pose.position.z = z  # Z coordinate of the point
        self.real_point.publish(m1)

        # Real orientation marker
        marker = Marker()
        marker.header.frame_id = "world"  # Assuming the frame_id in RViz2
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.07
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = x  # X coordinate of the point
        marker.pose.position.y = y  # Y coordinate of the point
        marker.pose.position.z = z  # Z coordinate of the point
        marker.pose.orientation.w = self.target_orientation[
            0
        ]  # X coordinate of the point
        marker.pose.orientation.x = self.target_orientation[
            1
        ]  # X coordinate of the point
        marker.pose.orientation.y = self.target_orientation[
            2
        ]  # Y coordinate of the point
        marker.pose.orientation.z = self.target_orientation[
            3
        ]  # Z coordinate of the point
        self.real_orientation.publish(marker)

        return x, y, z

    def define_point(self):
        if not self.valid_joint_config_found:
            x, y, z = self.get_random_point()
            random_angles = np.random.uniform(-np.pi / 3, np.pi / 3, 6)

            joint_angles = self.ik_calculator_omp.calculate_ik(
                pose=[x, y, z] + self.target_orientation,
                current_thetas=random_angles,
                max_iter=100,
                joint_limits=list(self.joint_limits.values()),
                method="LM_modified",
            )

            if self.is_joint_valid(joint_angles):
                if self.is_ik_acceptable(
                    c=self.ik_calculator_omp,
                    joint_angles=joint_angles,
                    required_position=[x, y, z],
                    required_orientation=self.target_orientation,
                ):
                    self.target_position = [x, y, z]
                    self.joint_angles = joint_angles
                    self.valid_joint_config_found = True
                else:
                    return
            else:
                return

    def execute_trajectory(self):
        if self.mode == "on":
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "gripper",
            ]
            duration = Duration(sec=self.time_period - 1)
            goal.trajectory.points.append(
                JointTrajectoryPoint(
                    positions=list(self.joint_angles) + [0.2],
                    velocities=[0.0] * 7,
                    accelerations=[0.0] * 7,
                    time_from_start=duration,
                )
            )
            self._action_client.wait_for_server(timeout_sec=10.0)
            send_goal_future = self._action_client.send_goal_async(
                goal, feedback_callback=self.feedback_callback
            )
            send_goal_future.add_done_callback(self.goal_response_callback)

    def is_ik_acceptable(
        self, c, joint_angles, required_position, required_orientation
    ):
        """
            Verifying if the position of the real end
            effector is equal (or close enough) to the required one.
        """
        fk = c.calculate_fk(current_thetas=joint_angles)
        fk_position = fk.pos
        fk_rot = fk.rot

        marker = Marker()
        marker.header.frame_id = "world"  # Assuming the frame_id in RViz2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = fk_position[0]  # X coordinate of the point
        marker.pose.position.y = fk_position[1]  # Y coordinate of the point
        marker.pose.position.z = fk_position[2]  # Z coordinate of the point
        self.fk_point.publish(marker)

        marker = Marker()
        marker.header.frame_id = "world"  # Assuming the frame_id in RViz2
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.07
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = fk_position[0]  # X coordinate of the point
        marker.pose.position.y = fk_position[1]  # Y coordinate of the point
        marker.pose.position.z = fk_position[2]  # Z coordinate of the point
        marker.pose.orientation.w = fk_rot[0]  # X coordinate of the point
        marker.pose.orientation.x = fk_rot[1]  # X coordinate of the point
        marker.pose.orientation.y = fk_rot[2]  # Y coordinate of the point
        marker.pose.orientation.z = fk_rot[3]  # Z coordinate of the point
        self.fk_rot.publish(marker)

        offset_orientation = 0.001

        distance = math.sqrt(
            (fk_position[0] - required_position[0]) ** 2
            + (fk_position[1] - required_position[1]) ** 2
            + (fk_position[2] - required_position[2]) ** 2
        )
        if (
            distance < 0.001
            and required_orientation[0] - offset_orientation
            < fk_rot[0]
            < required_orientation[0] + offset_orientation
            and required_orientation[1] - offset_orientation
            < fk_rot[1]
            < required_orientation[1] + offset_orientation
            and required_orientation[2] - offset_orientation
            < fk_rot[2]
            < required_orientation[2] + offset_orientation
            and required_orientation[3] - offset_orientation
            < fk_rot[3]
            < required_orientation[3] + offset_orientation
        ):
            return True
        else:
            return False

    def is_joint_valid(self, target):
        if (
            self.joint_limits["joint1"][0]
            < target[0]
            < self.joint_limits["joint1"][1]
            and self.joint_limits["joint2"][0]
            < target[1]
            < self.joint_limits["joint2"][1]
            and self.joint_limits["joint3"][0]
            < target[2]
            < self.joint_limits["joint3"][1]
            and self.joint_limits["joint4"][0]
            < target[3]
            < self.joint_limits["joint4"][1]
            and self.joint_limits["joint5"][0]
            < target[4]
            < self.joint_limits["joint5"][1]
            and self.joint_limits["joint6"][0]
            < target[5]
            < self.joint_limits["joint6"][1]
        ):
            return True
        else:
            return False

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback))

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result))
        self.valid_joint_config_found = False


def main(args=None):
    rclpy.init(args=args)
    debug_pykin = DebugPykin()
    try:
        rclpy.spin(debug_pykin)
    except KeyboardInterrupt:
        pass

    debug_pykin.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
