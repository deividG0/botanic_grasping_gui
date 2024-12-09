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
import numpy as np
import rclpy
import random
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.parameter import Parameter
from .pykin_kinematics import IKCalculatorOMP
from std_msgs.msg import Float32MultiArray, Bool
from visualization_msgs.msg import Marker

# Image related imports
from manipulator_trajectory_vision_gui.helper_functions.load_ros_parameters \
    import get_ros_parameters
from .tkinter_gui import TkinterGui


class OpenManipulatorPro(Node):
    """Class to simulate OMP Robot in"""

    omp_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ]

    gripper_joint_names = ["gripper", "gripper_sub"]

    def __init__(self):
        """Initialize the OMP simulation"""

        node_name = "omp_ros2"
        super().__init__(node_name)

        self.marker_namespace = "/marker"

        ###############################
        # OMP ACTION CLIENT
        ###############################
        self.switch_debug_pykin = self.create_publisher(
            String, "/switch_debug_mode", 10
        )
        self.switch_grasp_mode = self.create_publisher(
            String, "/switch_grasp_mode", 10
            )
        self.switch_cam_calibrate_pub = self.create_publisher(
            String, "/switch_cam_calibration_mode", 10
        )
        self.switch_mirroring_effect_pub = self.create_publisher(
            String, "/switch_mirroring_effect", 10
        )

        self.omp_controller_publisher = self.create_publisher(
            Float32MultiArray, "/omp_controller_topic", 10
        )
        self.gripper_controller_publisher = self.create_publisher(
            Float32MultiArray, "/gripper_controller_topic", 10
        )
        self.step_grasping_sub = self.create_subscription(
            Bool, "step_grasping", self.step_grasping, 10
        )

        self.auto_grasping = False

        self.joint_limits = {
            "joint1": (-np.pi / 2, np.pi / 2),
            "joint2": (-np.pi / 2, np.pi / 2),
            "joint3": (-np.pi / 2, 3 * np.pi / 4),
            "joint4": (-np.pi / 2, np.pi / 2),
            "joint5": (-np.pi / 2, np.pi / 2),
            "joint6": (-np.pi / 2, np.pi / 2),
        }

        self.selected_point_x, self.selected_point_y, self.selected_point_z = (
            None,
            None,
            None,
        )

        # Gripper action client
        self._gripper_action_client = ActionClient(
            self, FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory"
        )

        # Ros parameters
        self.ros_parameters, declared_parameters = get_ros_parameters(
            node_name
            )
        self.declare_parameters(namespace="", parameters=declared_parameters)
        self.get_logger().info(f"{declared_parameters}")
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.closed_gripper = self.ros_parameters["gripper_closed_position"]
        self.open_gripper = self.ros_parameters["gripper_open_position"]

        # Camera
        self.rgb_image = None
        self.depth_image = None
        self.H, self.W = None, None
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0

        self.grasping_point_sub = self.create_subscription(
            Float32MultiArray,
            "/grasping_point",
            self.grasping_point_listener,
            10
        )

        self.number_defined = False

        ###############################
        # DETECTION MODEL & GRASPING
        ###############################
        self.pre_grasping_point = []
        self.grasping_point = []
        self.pre_grasp_offset = 0.1
        self.grasping_steps_list = []
        self.number_items = 0
        self.old_number_items = 0

        ###############################
        # TKINTER GUI
        ###############################
        self.tkinter = TkinterGui(
            self.send_goal,
            self.send_goal_gripper,
            self.switch_auto_grasp,
            self.is_in_workspace,
            self.switch_debug_mode,
            self.switch_calibrate_cam,
            self.switch_mirroring_effect,
        )

        self.inv_kin = False
        self.debug_inv_kin = False
        self.goal_prim = ""

        self.root = self.tkinter.build_frames()
        # timer_period = 1/60
        timer_period = 0.2
        self.timer = self.create_timer(timer_period, self.simulate)

        self.omp_ik_calculator = IKCalculatorOMP()

        self.point_publisher = self.create_publisher(
            Marker,
            f"{self.marker_namespace}/selected_point_marker",
            10
        )

        ###############################
        # MARKERS
        ###############################
        self.workspace = self.create_publisher(
            Marker, f"{self.marker_namespace}/workspace", 10
        )
        self.unallowed_space = self.create_publisher(
            Marker, f"{self.marker_namespace}/unallowed_space", 10
        )
        self.floor_marker = self.create_publisher(
            Marker, f"{self.marker_namespace}/floor", 10
        )
        self.marker_publisher_grasping = self.create_publisher(
            Marker, f"{self.marker_namespace}/grasping", 10
        )
        self.marker_publisher_pre_grasping = self.create_publisher(
            Marker, f"{self.marker_namespace}/pre_grasping", 10
        )
        self.timer = self.create_timer(1.0, self.publish_marker)
        self.center = (0.0, 0.0, 0.2)
        self.radius_smaller_sphere = 0.15
        self.radius_bigger_sphere = 0.75
        self.floor = 0.1
        self.fk_point = self.create_publisher(
            Marker, f"{self.marker_namespace}/fk_point", 10
        )
        self.fk_rot = self.create_publisher(
            Marker, f"{self.marker_namespace}/fk_rot", 10
        )

        ###############################
        # AUTO GRASP STATE MACHINE
        ###############################
        self.state = "selecting_fruit"
        self.timer = self.create_timer(0.1, self.auto_grasp_state_machine)
        self.current_pre_grasping_point = None
        self.current_grasping_point = None
        self.finish_traj = True

    def switch_debug_mode(self, str):
        msg = String()
        msg.data = f"{str}"
        self.switch_debug_pykin.publish(msg)

    def switch_mirroring_effect(self, str):
        msg = String()
        msg.data = f"{str}"
        self.switch_mirroring_effect_pub.publish(msg)

    def switch_calibrate_cam(self, str):
        msg = String()
        msg.data = f"{str}"
        self.switch_cam_calibrate_pub.publish(msg)

    def switch_auto_grasp(self, str):
        if str == "on":
            self.auto_grasping = True
        elif str == "off":
            self.auto_grasping = False

        msg = String()
        msg.data = f"{str}"
        self.switch_grasp_mode.publish(msg)

    # def number_items_listener(self, msg):
    #     self.number_items = msg.data

    def grasping_point_listener(self, msg):
        self.grasping_point = np.array(msg.data).tolist()
        if len(self.grasping_point) == 0:
            self.number_items = 0
        else:
            grasping_offset_x = 0.08
            grasping_offset_y = -0.03
            self.grasping_point[0] = (
                self.grasping_point[0] - grasping_offset_x
            )  # For not going too forward
            self.grasping_point[1] = (
                self.grasping_point[1] - grasping_offset_y
            )  # For not going too forward

            self.pre_grasping_point = self.grasping_point.copy()
            pre_grasping_offset_x = 0.08
            self.pre_grasping_point[0] = (
                self.pre_grasping_point[0] - pre_grasping_offset_x
            )
            self.number_items = 1

    def publish_marker(self):
        if len(self.grasping_point) > 0:
            marker = Marker()
            marker.header.frame_id = "world"  # Assuming the frame_id in RViz2
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.pose.position.x = float(
                self.grasping_point[0]
            )  # X coordinate of the point
            marker.pose.position.y = float(
                self.grasping_point[1]
            )  # Y coordinate of the point
            marker.pose.position.z = float(
                self.grasping_point[2]
            )  # Z coordinate of the point

            marker.pose.orientation.w = 1.0  # X coordinate of the point
            marker.pose.orientation.x = 0.0  # X coordinate of the point
            marker.pose.orientation.y = 0.0  # Y coordinate of the point
            marker.pose.orientation.z = 0.0  # Z coordinate of the point

            self.marker_publisher_grasping.publish(marker)

            marker = Marker()
            marker.header.frame_id = "world"  # Assuming the frame_id in RViz2
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose.position.x = float(
                self.pre_grasping_point[0]
            )  # X coordinate of the point
            marker.pose.position.y = float(
                self.pre_grasping_point[1]
            )  # Y coordinate of the point
            marker.pose.position.z = float(
                self.pre_grasping_point[2]
            )  # Z coordinate of the point

            marker.pose.orientation.w = 1.0  # X coordinate of the point
            marker.pose.orientation.x = 0.0  # X coordinate of the point
            marker.pose.orientation.y = 0.0  # Y coordinate of the point
            marker.pose.orientation.z = 0.0  # Z coordinate of the point

            self.marker_publisher_pre_grasping.publish(marker)

        marker = Marker()
        marker.header.frame_id = "world"  # Assuming the frame_id in RViz2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = self.radius_bigger_sphere * 2
        marker.scale.y = self.radius_bigger_sphere * 2
        marker.scale.z = self.radius_bigger_sphere * 2
        marker.color.a = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.position.x = self.center[0]  # X coordinate of the point
        marker.pose.position.y = self.center[1]  # Y coordinate of the point
        marker.pose.position.z = self.center[2]  # Z coordinate of the point

        # Publish marker
        self.workspace.publish(marker)

        marker = Marker()
        marker.header.frame_id = "world"  # Assuming the frame_id in RViz2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = self.radius_smaller_sphere * 2
        marker.scale.y = self.radius_smaller_sphere * 2
        marker.scale.z = self.radius_smaller_sphere * 2
        marker.color.a = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.position.x = self.center[0]  # X coordinate of the point
        marker.pose.position.y = self.center[1]  # Y coordinate of the point
        marker.pose.position.z = self.center[2]  # Z coordinate of the point

        # Publish marker
        self.unallowed_space.publish(marker)

        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = self.floor / 2
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 3.0  # Length of the plane
        marker.scale.y = 3.0  # Width of the plane
        marker.scale.z = self.floor  # Height of the plane, making it very thin
        marker.color.a = 0.2  # Alpha value (opacity)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.floor_marker.publish(marker)

    def publish_point_marker(self):
        if (
            self.selected_point_x is not None
            and self.selected_point_y is not None
            and self.selected_point_z is not None
        ):
            marker = Marker()
            marker.header.frame_id = "world"  # Assuming the frame_id in RViz2
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 0.8
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position.x = self.selected_point_x
            marker.pose.position.y = self.selected_point_y
            marker.pose.position.z = self.selected_point_z

            self.point_publisher.publish(marker)

    def is_in_workspace(self, point):
        try:
            if (
                (point[0] - self.center[0]) ** 2
                + (point[1] - self.center[1]) ** 2
                + (point[2] - self.center[2]) ** 2
                > self.radius_smaller_sphere**2
                and (point[0] - self.center[0]) ** 2
                + (point[1] - self.center[1]) ** 2
                + (point[2] - self.center[2]) ** 2
                < self.radius_bigger_sphere**2
                and point[2] > self.floor
            ):
                return True
        except Exception as e:
            self.get_logger().warn(
                f"Error in checking point to workspace: {e}."
                " Point shape: {point}"
            )
            return False

    ##############################################################
    # FUNCTIONS RELATED TO AUTOMATED GRASPING
    ##############################################################

    def auto_grasp_state_machine(self):
        if self.auto_grasping is True and self.finish_traj is True:
            self.get_logger().info("Auto grasp mode")
            self.get_logger().info(
                f"Number items {self.number_items} || "
                "self.state {self.state}  || "
                "self.finish_traj {self.finish_traj}"
            )

            # State Machine
            if self.state == "selecting_fruit":
                self.current_pre_grasping_point = self.pre_grasping_point
                self.current_grasping_point = self.grasping_point
                self.send_goal_gripper([self.open_gripper])

                if self.number_items > 0:
                    self.state = "pre_grasp"
                if self.number_items <= 0:
                    self.state = "selecting_fruit"
            elif self.state == "pre_grasp":
                self.send_goal(
                    self.current_pre_grasping_point, "slow", True, 1.0
                )  # Going pre grasping position

                self.state = "grasp"
            elif self.state == "grasp":
                self.send_goal(
                    self.current_grasping_point, "slow", True, 0.0
                )  # Going pre grasping position

                self.state = "close_gripper"
            elif self.state == "close_gripper":
                self.send_goal_gripper([self.closed_gripper])

                self.state = "returning"
            elif self.state == "returning":
                self.send_goal(
                    [0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0], "slow", True, 1.0
                )  # Returning with the object

                self.state = "depose_pose"
            elif self.state == "depose_pose":
                self.send_goal(
                    [0.2, 0.2, 0.25, 0.707, 0.0, 0.707, 0.0], "slow", True, 1.0
                )

                # if self.finish_traj==True:
                self.state = "open_gripper"
            elif self.state == "open_gripper":
                self.send_goal_gripper([self.open_gripper])

                self.state = "waiting_fruit"
            elif self.state == "waiting_fruit":
                self.send_goal(
                    [0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0], "slow", True, 1.0
                    )

                if self.number_items <= 0:
                    self.state = "waiting_fruit"
                if self.number_items > 0:
                    self.state = "selecting_fruit"

    ##############################################################
    # FUNCTIONS RELATED TO MOTION PLANNING
    ##############################################################
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

    def is_ik_acceptable(
            self, joint_angles, required_position, required_orientation
            ):
        """
            Verifying if the position of the real end effector
            is equal (or close enough) to the required one.
        """
        fk = self.omp_ik_calculator.calculate_fk(current_thetas=joint_angles)
        fk_position = fk.pos
        fk_rot = fk.rot

        fk_p = Marker()
        fk_p.header.frame_id = "world"  # Assuming the frame_id in RViz2
        fk_p.type = Marker.SPHERE
        fk_p.action = Marker.ADD
        fk_p.scale.x = 0.05
        fk_p.scale.y = 0.05
        fk_p.scale.z = 0.05
        fk_p.color.a = 0.5
        fk_p.color.r = 0.0
        fk_p.color.g = 0.0
        fk_p.color.b = 0.0
        fk_p.pose.position.x = fk_position[0]  # X coordinate of the point
        fk_p.pose.position.y = fk_position[1]  # Y coordinate of the point
        fk_p.pose.position.z = fk_position[2]  # Z coordinate of the point
        self.fk_point.publish(fk_p)

        fk_r = Marker()
        fk_r.header.frame_id = "world"  # Assuming the frame_id in RViz2
        fk_r.type = Marker.ARROW
        fk_r.action = Marker.ADD
        fk_r.scale.x = 0.07
        fk_r.scale.y = 0.02
        fk_r.scale.z = 0.02
        fk_r.color.a = 0.5
        fk_r.color.r = 0.0
        fk_r.color.g = 0.0
        fk_r.color.b = 0.0
        fk_r.pose.position.x = fk_position[0]  # X coordinate of the point
        fk_r.pose.position.y = fk_position[1]  # Y coordinate of the point
        fk_r.pose.position.z = fk_position[2]  # Z coordinate of the point
        fk_r.pose.orientation.w = fk_rot[0]  # X coordinate of the point
        fk_r.pose.orientation.x = fk_rot[1]  # X coordinate of the point
        fk_r.pose.orientation.y = fk_rot[2]  # Y coordinate of the point
        fk_r.pose.orientation.z = fk_rot[3]  # Z coordinate of the point
        self.fk_rot.publish(fk_r)

        offset_orientation = 0.05

        distance = math.sqrt(
            (fk_position[0] - required_position[0]) ** 2
            + (fk_position[1] - required_position[1]) ** 2
            + (fk_position[2] - required_position[2]) ** 2
        )

        if (
            distance < 0.05
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

    def find_joint_configuration(self, target_position, target_orientation):
        try:
            max_it = 100
            it = 0
            while True:
                random_angles = np.random.uniform(-np.pi / 3, np.pi / 3, 6)
                joint_angles = self.omp_ik_calculator.calculate_ik(
                    pose=target_position + target_orientation,
                    current_thetas=random_angles,
                    max_iter=random.randint(100, 200),
                    method="LM"
                )

                if self.is_ik_acceptable(
                    joint_angles=joint_angles,
                    required_position=target_position,
                    required_orientation=target_orientation,
                ) and self.is_joint_valid(joint_angles):
                    self.get_logger().info(
                        "A valid joint configuration was found."
                        )
                    return joint_angles

                if it > max_it:
                    self.get_logger().info(
                        "A valid joint configuration was NOT found."
                    )
                    return None
                it += 1
        except Exception as e:
            self.get_logger().error(
                f"Error trying to find a joint configuration: {e}"
                )

    def step_grasping(self, msg):
        if msg.data:
            self.finish_traj = True

    def send_goal(
        self,
        target: list,
        movement: str = "slow",
        inv_kin: bool = False,
        check_self_collision: float = 1.0,  # It means True
    ):
        """
        Send trajectory to the OMP robot in .

        Parameters
        ----------
        target : list
            Target joint positions or
            target pose [x, y, z, w, qx, qy, qz] in degrees (if inv_kin
            is True).
        movement : str
            Type of movement (slow, fast).
        inv_kin : bool
            If True, the target is the desired pose
            (default: {False}).
        """
        self.get_logger().info(f"In send_goal: {target}.")

        self.goal_prim = "OMP"
        if movement == "slow":
            time_in_sec = self.ros_parameters["trajectory_time_slow"]
        elif movement == "fast":
            time_in_sec = self.ros_parameters["trajectory_time_fast"]

        joint_angles = target

        if inv_kin:
            self.selected_point_x,
            self.selected_point_y,
            self.selected_point_z = (
                joint_angles[:3]
            )
            self.publish_point_marker()

            # Verify if point is in the robot workspace
            if not self.is_in_workspace(target[:3]):
                self.get_logger().info(
                    f"Point [x, y, z]: {target[:3]} not in OMP workspace."
                )
                return

            target_position = joint_angles[:3]
            target_orientation = joint_angles[3:]
            self.get_logger().info("Target position [x, y, z]:"
                                   f" {target_position}")
            self.get_logger().info(
                f"Target orientation [w, qx, qy, qz]: {target_orientation}"
            )

            result = self.find_joint_configuration(
                target_position, target_orientation
                )
            if result is None:
                return
            else:
                joint_angles = result

        self.get_logger().info(
            f"Joint angles: {joint_angles}, "
            "time in sec for the trajectory {[time_in_sec]}"
        )

        self.finish_traj = False
        # Sending through topic
        msg = Float32MultiArray()
        msg.data = (
            list(joint_angles) + [float(time_in_sec)] + [check_self_collision]
        )  # Example float array
        self.omp_controller_publisher.publish(msg)

    def send_goal_gripper(self, position: list):
        """
        Send trajectory to the OMP gripper in .

        Parameters
        ----------
        time_in_sec : float
            Time in seconds to complete the action.
        position : list
            List of joint positions.

        """
        self.finish_traj = False
        # Sending through topic
        msg = Float32MultiArray()
        msg.data = list(position)  # Example float array
        self.gripper_controller_publisher.publish(msg)

    def parameters_callback(self, params):
        """
        Update ROS2 parameters according to the config/params.yaml file.

        Parameters
        ----------
        params : list
            List of parameters to be updated

        Returns
        -------
        SetParametersResult
            Result of the update of the parameters

        """
        for param in params:
            if param.name == "home_joint_array":
                if param.type_ in [Parameter.Type.DOUBLE_ARRAY]:
                    self.ros_parameters["home_joint_array"] = list(param.value)
                else:
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

    def simulate(self) -> None:
        """OMP Simulation."""
        self.root.update()


def main(args=None):
    """Main function to run the OMP  simulation."""
    rclpy.init(args=args)

    ros2_publisher = OpenManipulatorPro()
    rclpy.spin(ros2_publisher)

    ros2_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
