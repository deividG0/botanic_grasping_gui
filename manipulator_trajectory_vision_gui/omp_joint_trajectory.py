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
import rclpy
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray, Bool

import matplotlib.pyplot as plt
from datetime import datetime

from manipulator_trajectory_vision_gui.\
    checking_self_collision.\
    manual_check_self_collision import (
        OMPCollisionChecker,
    )


class OMPJointController(Node):
    omp_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]

    def __init__(self):
        super().__init__("omp_joint_trajectory")

        ###############################
        # PLOT VARIABLES
        ###############################
        self.plot_path = f"{os.getenv('PWD')}"
        self.real_joint_values = []

        self.start_time = None
        self.timer = None
        self.time_real = np.array([])
        self.save_plots = False

        self.omp_collision_checker = OMPCollisionChecker()

        client_cb_group = ReentrantCallbackGroup()
        self.joint_states = [0.0] * 6
        self.gripper_position = None
        self.create_subscription(
            JointState,
            "/joint_states",
            self.update_joint_state,
            10,
            callback_group=client_cb_group,
        )

        self.omp_controller_sub = self.create_subscription(
            Float32MultiArray,
            "/omp_controller_topic",
            self.listener_callback,
            10
        )

        self.step_grasping_pub = self.create_publisher(
            Bool, "step_grasping", 10
            )

        self.action_arm_controller = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

        self.trajectory_position_list = []
        self.trajectory_velocity_list = []
        self.trajectory_acceleration_list = []
        self.time_t0 = 0.0
        self.time_in_sec = 0.0
        self.time_step = 0.01  # time step
        self.time = []
        self.coeffs = []
        self.trajectory_in_execution = False
        self.check_self_collision = True

        self.trajectory_points = []
        self.ik_target = []

        # Markers variables

        self.marker_namespace = "/marker"
        self.markers_publisher = self.create_publisher(
            Marker,
            f"{self.marker_namespace}/collision_checker_visualization",
            10
        )
        timer_period = 0.5  # seconds
        self.create_timer(timer_period, self.timer_callback_marker)
        self.timer_collect_joint_states = None
        self.counter = 0
        self.all_parallelepipeds_trajectory = []
        self.all_omp_in_collision_bool = []
        self.index = 0

    def timer_callback_marker(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "parallelepiped"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.005
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        if self.index < len(self.all_parallelepipeds_trajectory):
            for parallelepiped in \
                    self.all_parallelepipeds_trajectory[self.index]:
                list_of_edges = [
                    (parallelepiped[0], parallelepiped[1]),
                    (parallelepiped[1], parallelepiped[2]),
                    (parallelepiped[2], parallelepiped[3]),
                    (parallelepiped[3], parallelepiped[0]),
                    (parallelepiped[4], parallelepiped[5]),
                    (parallelepiped[5], parallelepiped[6]),
                    (parallelepiped[6], parallelepiped[7]),
                    (parallelepiped[7], parallelepiped[4]),
                    (parallelepiped[0], parallelepiped[4]),
                    (parallelepiped[1], parallelepiped[5]),
                    (parallelepiped[2], parallelepiped[6]),
                    (parallelepiped[3], parallelepiped[7]),
                ]
                for edge in list_of_edges:
                    p1, p2 = edge
                    marker.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
                    marker.points.append(Point(x=p2[0], y=p2[1], z=p2[2]))

            self.index += 1
        else:
            self.index = 0

        self.markers_publisher.publish(marker)
        self.counter += 1

    def listener_callback(self, msg):
        self.ik_target = msg.data[:6]
        self.time_in_sec = msg.data[-2]
        self.check_self_collision = True if msg.data[-1] == 1.0 else False
        # Call the get the trajectories points function
        self.calculate_trajectory_points()

    def start_joint_states_collection(self):
        self.start_time = self.get_clock().now().to_msg().sec
        self.timer_collect_joint_states = self.create_timer(
            self.time_step, self.collect_joint_states
        )

    def collect_joint_states(self):
        try:
            self.real_joint_values.append(self.joint_states)

            current_time = self.get_clock().now().to_msg().sec
            elapsed_time = current_time - self.start_time
            if len(self.time_real) == 0:
                self.time_real = np.append(self.time_real, self.time_step)
            else:
                self.time_real = np.append(
                    self.time_real, self.time_real[-1] + self.time_step
                )

            if elapsed_time >= self.time_in_sec:
                self.get_logger().info(
                    "Finished collecting joint states for plotting."
                    )
                self.plot_real_graph(
                    time=self.time_real, positions=self.real_joint_values
                )
                self.timer_collect_joint_states.cancel()
                self.timer_collect_joint_states = None  # Delete the timer
        except Exception as e:
            self.get_logger().info(f"Erro on plot graphs function: {e}")

    def plot_graphs(self, time, positions, velocities, accelerations):
        try:
            # Plotting the results
            _, axes = plt.subplots(3, 1, figsize=(10, 15))

            positions = np.array(positions)

            # Plot positions
            for i in range(6):
                axes[0].plot(time, positions[:, i], label=f"Joint {i+1}")
            axes[0].set_title("Position Trajectories")
            axes[0].set_xlabel("Time [s]")
            axes[0].set_ylabel("Position [units]")
            axes[0].legend()
            axes[0].grid()

            # Plot velocities
            for i in range(6):
                axes[1].plot(time, velocities[:, i], label=f"Joint {i+1}")
            axes[1].set_title("Velocity Trajectories")
            axes[1].set_xlabel("Time [s]")
            axes[1].set_ylabel("Velocity [units/s]")
            axes[1].legend()
            axes[1].grid()

            # Plot accelerations
            for i in range(6):
                axes[2].plot(time, accelerations[:, i], label=f"Joint {i+1}")
            axes[2].set_title("Acceleration Trajectories")
            axes[2].set_xlabel("Time [s]")
            axes[2].set_ylabel("Acceleration [units/s^2]")
            axes[2].legend()
            axes[2].grid()

            plt.tight_layout()

            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
            plot_file_name = f"plot_planned_traj({dt_string}).png"

            # Saving
            plt.savefig(self.plot_path + plot_file_name)
            plt.clf()
        except Exception as e:
            self.get_logger().info("Erro on plot graphs function", e)

    def plot_real_graph(self, time, positions):
        try:
            # Plotting the results
            _, axes = plt.subplots(1, 1, figsize=(15, 10))

            positions = np.array(positions)
            time = np.array(time)

            # Plot positions
            for i in range(6):
                axes.plot(time, positions[:, i], label=f"Joint {i+1}")
            axes.set_title("Position Trajectories")
            axes.set_xlabel("Time [s]")
            axes.set_ylabel("Position [units]")
            axes.legend()
            axes.grid()

            plt.tight_layout()

            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
            plot_file_name = f"plot_real_traj({dt_string}).png"

            # Saving
            plt.savefig(self.plot_path + plot_file_name)
            plt.clf()
        except Exception as e:
            self.get_logger().info(f"Erro on plot real function {e}")

    def calculate_trajectory_points(
        self,
    ):
        self.all_parallelepipeds_trajectory = []
        self.all_omp_in_collision_bool = []
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
            "p0": list(self.joint_states),
            "pf": list(self.ik_target),
            "v0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "vf": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "a0": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "af": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }

        # Calculate the coefficients for each DOF
        coefficients = []
        for i in range(6):
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
        positions = np.zeros((len(self.time), 6))
        velocities = np.zeros((len(self.time), 6))
        accelerations = np.zeros((len(self.time), 6))

        self.get_logger().info("[OJT] Planning trajectory ...")
        for i in range(6):
            coeffs = coefficients[i]
            positions[:, i] = (
                coeffs[0]
                + coeffs[1] * self.time
                + coeffs[2] * self.time**2
                + coeffs[3] * self.time**3
                + coeffs[4] * self.time**4
                + coeffs[5] * self.time**5
            )
            velocities[:, i] = (
                coeffs[1]
                + 2 * coeffs[2] * self.time
                + 3 * coeffs[3] * self.time**2
                + 4 * coeffs[4] * self.time**3
                + 5 * coeffs[5] * self.time**4
            )
            accelerations[:, i] = (
                2 * coeffs[2]
                + 6 * coeffs[3] * self.time
                + 12 * coeffs[4] * self.time**2
                + 20 * coeffs[5] * self.time**3
            )

        if self.check_self_collision:
            self.get_logger().info(
                "[OJT] Checking for possible self collision ..."
                )
            for t in range(0, len(self.time), 100):
                # Check possible self collision
                joint_angles = {
                    "world_fixed": 0.0,
                    "joint1": positions[t, 0],
                    "joint2": positions[t, 1],
                    "joint3": positions[t, 2],
                    "joint4": positions[t, 3],
                    "joint5": positions[t, 4],
                    "joint6": positions[t, 5],
                    "camera_joint": 0.0,
                    "camera_optical_joint": 0.0,
                    "base_link_gripper": 0.0,
                    "gripper": 0.0,
                    "gripper_sub": 0.0,
                }
                omp_in_collision, parallelepipeds = (
                    self.omp_collision_checker.check_self_collision(
                        joint_angles
                        )
                )
                if omp_in_collision:
                    self.get_logger().warn("#" * 100)
                    self.get_logger().warn(f"{omp_in_collision, joint_angles}")

                self.all_parallelepipeds_trajectory.append(parallelepipeds)
                self.all_omp_in_collision_bool.append(omp_in_collision)

            if any(self.all_omp_in_collision_bool):
                self.get_logger().warn(
                    "[OJT] Self collision detected in built trajectory."
                )
                return
            else:
                self.get_logger().info("[OJT] No self collision detected ...")

        for i, position in enumerate(positions):
            p = JointTrajectoryPoint()
            p.positions = list(position) + [
                self.gripper_position
            ]  # Adding current gripper angle value
            p.time_from_start = rclpy.duration.Duration(
                seconds=self.time[i]
                ).to_msg()
            trajectory.points.append(p)

        # Send the trajectory to the controller
        self.send_trajectory(trajectory)
        if self.save_plots:
            self.start_joint_states_collection()
            self.plot_graphs(self.time, positions, velocities, accelerations)

    def update_joint_state(self, msg: JointState):
        """Get simulated OMP joint angles."""
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

    def compute_quintic_coefficients(self, p0, pf, v0, vf, a0, af, t0, tf):
        # This function uses the values that we already
        # now about the polynomial equation to find the coeffs.
        # The values that we already now are the initial and final values.
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

        self.action_arm_controller.wait_for_server()

        self._send_goal_future = self.action_arm_controller.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        feedback
        # self.get_logger().info(f"[OJT] Received feedback: {feedback}")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        result
        # self.get_logger().info(f"Result: {result}")

        # Sending end of the movimentation to the omp_ros2
        msg = Bool()
        msg.data = True
        self.step_grasping_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    omp_server = OMPJointController()
    executor = MultiThreadedExecutor()
    executor.add_node(omp_server)

    try:
        omp_server.get_logger().info(
            "[OMP] Beginning client, shut down with CTRL-C"
            )
        executor.spin()
    except KeyboardInterrupt:
        omp_server.get_logger().info(
            "[OMP] Keyboard interrupt, shutting down.\n"
            )
    omp_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
