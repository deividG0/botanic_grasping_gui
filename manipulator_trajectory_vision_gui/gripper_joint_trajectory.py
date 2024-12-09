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

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Bool


class OMPJointController(Node):
    joint_names = ["gripper", "gripper_sub"]

    def __init__(self):
        super().__init__("omp_joint_trajectory")

        client_cb_group = ReentrantCallbackGroup()
        self.omp_states = [0.0] * 6
        self.joint_states = 0.0
        self.create_subscription(
            JointState,
            "/joint_states",
            self.update_joint_state,
            10,
            callback_group=client_cb_group,
        )

        self.gripper_controller_sub = self.create_subscription(
            Float32MultiArray,
            "/gripper_controller_topic",
            self.listener_callback,
            10
        )

        self.step_grasping_pub = self.create_publisher(
            Bool,
            "step_grasping",
            10
            )

        self.action_gripper_controller = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

        self.trajectory_position_list = []
        self.trajectory_velocity_list = []
        self.trajectory_acceleration_list = []
        self.time_t0 = 0.0
        self.time_in_sec = 1.0
        self.time_step = 0.01  # time step
        self.time = []
        self.coeffs = []
        self.trajectory_in_execution = False

        self.position = []

    def listener_callback(self, msg):
        self.position = msg.data
        # Call the get the trajectories points function
        self.calculate_trajectory_points()

    def calculate_trajectory_points(
        self,
    ):
        self.time_real = np.array([])
        self.real_joint_values = []

        # Create a JointTrajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]

        # Define time parameters
        t0 = 0.0  # start time
        tf = self.time_in_sec  # end time
        self.time = np.arange(t0, tf, self.time_step)

        # Define initial and final conditions for each DOF
        init_final_conditions = {
            "p0": list([self.joint_states]),
            "pf": list(self.position),
            "v0": [0.0],
            "vf": [0.0],
            "a0": [0.0],
            "af": [0.0],
        }

        # Calculate the coefficients for each DOF
        coefficients = []
        for i in range(1):
            coeffs = self.compute_quintic_coefficients(
                init_final_conditions["p0"][i],
                init_final_conditions["pf"][i],
                init_final_conditions["v0"][i],
                init_final_conditions["vf"][i],
                init_final_conditions["a0"][i],
                init_final_conditions["af"][i],
                t0,
                tf,
            )
            coefficients.append(coeffs)

        # Calculate position, velocity, and acceleration for each DOF
        positions = np.zeros((len(self.time), 1))

        self.get_logger().info("[GJT] Planning trajectory ...")
        for i in range(1):
            coeffs = coefficients[i]
            positions[:, i] = (
                coeffs[0]
                + coeffs[1] * self.time
                + coeffs[2] * self.time**2
                + coeffs[3] * self.time**3
                + coeffs[4] * self.time**4
                + coeffs[5] * self.time**5
            )

        for i, position in enumerate(positions):
            p = JointTrajectoryPoint()
            p.positions = self.omp_states + list(
                position
            )  # Adding current omp angles values
            p.time_from_start = rclpy.duration.Duration(
                seconds=self.time[i]).to_msg()
            trajectory.points.append(p)

        # Send the trajectory to the controller
        self.send_trajectory(trajectory)

    def update_joint_state(self, msg: JointState):
        """Get simulated OMP joint angles in the following order:

        shoulder_pan_joint, shoulder_lift_joint, elbow_joint
        wrist_1_joint, wrist_2_joint, wrist_3_joint

        Parameters
        ----------
        msg : JointState
            JointState message from the /joint_states topic

        """
        if msg.name[0] == "gripper":
            self.joint_states = msg.position[0]
            self.omp_states[0] = msg.position[3]
            self.omp_states[1] = msg.position[1]
            self.omp_states[2] = msg.position[2]
            self.omp_states[3] = msg.position[4]
            self.omp_states[4] = msg.position[5]
            self.omp_states[5] = msg.position[6]
        else:
            self.omp_states[0] = msg.position[2]
            self.omp_states[1] = msg.position[0]
            self.omp_states[2] = msg.position[1]
            self.omp_states[3] = msg.position[3]
            self.omp_states[4] = msg.position[4]
            self.omp_states[5] = msg.position[5]
            self.joint_states = msg.position[6]

    def compute_quintic_coefficients(self, p0, pf, v0, vf, a0, af, t0, tf):
        """
            This function uses the values that we already
            now about the polynomial equation to find the coeffs.
            The values that we already now are the initial and final values.
        """
        A = np.array(
            [
                [1, t0, t0**2, t0**3, t0**4, t0**5],
                [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
                [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
            ]
        )
        b = np.array([p0, v0, a0, pf, vf, af])
        coeffs = np.linalg.solve(A, b)
        self.coeffs = coeffs
        return coeffs

    def send_trajectory(self, trajectory):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        self.action_gripper_controller.wait_for_server()

        self._send_goal_future \
            = self.action_gripper_controller.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback
            )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        feedback
        # self.get_logger().info(f"Received feedback: {feedback}")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        result
        # self.get_logger().info(f'Result: {result}')

        # Sending end of the movimentation to the omp_ros2
        msg = Bool()
        msg.data = True
        self.step_grasping_pub.publish(msg)


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    omp_server = OMPJointController()
    executor = MultiThreadedExecutor()
    executor.add_node(omp_server)

    try:
        omp_server.get_logger().info(
            "[GJT] Beginning client, shut down with CTRL-C"
            )
        executor.spin()
    except KeyboardInterrupt:
        omp_server.get_logger().info(
            "[GJT] Keyboard interrupt, shutting down.\n"
            )
    omp_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
