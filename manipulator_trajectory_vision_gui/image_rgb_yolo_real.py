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
import rclpy
import cv2
import struct
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import Float32MultiArray, Header
from tf2_ros import TransformListener, Buffer
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from sensor_msgs_py import point_cloud2
from ament_index_python.packages import get_package_share_directory

from visualization_msgs.msg import Marker

from manipulator_trajectory_vision_gui.helper_functions.utils import (
    generate_oriented_rectangle,
    get_min_depth_coordinates,
    get_cropped_img,
    calculate_midpoint,
    draw_rotated_rectangle,
)


class ImageRGBYOLO(Node):
    def __init__(self):
        super().__init__("image_depth_yolo")

        self.model = YOLO(
            os.path.join(
                get_package_share_directory(
                    'manipulator_trajectory_vision_gui'
                    ),
                'yolo_weight',
                'best.pt'
                )
            )

        self.depth_image = None
        self.rgb_image = None
        self.point_cloud_mask = None

        self.fx = 528.433756558705  # Focal length in x
        self.fy = 528.433756558705  # Focal length in y
        self.cx = 320.5  # Optical center x
        self.cy = 240.5  # Optical center y

        self.W = 640
        self.H = 480

        self.end_link_x, self.end_link_y, self.end_link_z = 0, 0, 0
        self.pre_grasp_offset = 0.10
        self.image_shown = np.zeros((480, 640))
        self.image_shown = self.image_shown.astype(np.float64)
        self.space_print = "##" * 20

        # Point Cloud Publisher
        self.point_cloud_publisher = self.create_publisher(
            PointCloud2, "/realsense/point_cloud", 10
        )
        self.grasping_point_pub = self.create_publisher(
            Float32MultiArray, "/grasping_point", 10
        )

        self.create_timer(0.1, self.image_analysis)
        self.create_timer(0.1, self.get_transform)
        self.create_timer(0.1, self.pub_arrow_marker)
        self.create_timer(1.0, self.create_point_cloud)
        self.create_timer(0.1, self._show)

        self.mode = "on"
        self.create_subscription(String, "/switch_grasp_mode", self.switch, 10)

        # Create a buffer to store transforms and
        # a TransformListener to fill the buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        # Subscriptions to RGB and Depth topics
        self.rgb_sub = self.create_subscription(
            Image, "realsense/rgb/image_raw", self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, "realsense/depth/image_raw", self.depth_callback, 10
        )

        self.marker_namespace = "/marker"
        self.marker_publisher = self.create_publisher(
            Marker, f"{self.marker_namespace}/visualization_marker", 10
        )
        self.grasping_point = None

    def switch(self, msg):
        self.mode = msg.data

    def rgb_callback(self, msg):
        try:
            # Convert the ROS2 Image message to an OpenCV image
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"Error in getting camera RGB frames: {e}")

    def depth_callback(self, msg):
        try:
            # Convert the ROS2 Image message to an OpenCV image
            d_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            d_image = d_image.astype(np.float32) / 1000.0  # Example conversion
            self.depth_image = cv2.resize(d_image, (self.W, self.H))
        except Exception as e:
            self.get_logger().warn(
                f"Error in getting camera depth frames: {e}"
                )

    def pub_arrow_marker(
        self,
    ):
        if self.grasping_point is not None and self.mode == "on":
            # Publishing arrow of grasping
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
            marker.color.b = 0.0
            marker.pose.position.x = float(self.grasping_point[0])
            marker.pose.position.y = float(self.grasping_point[1])
            marker.pose.position.z = float(self.grasping_point[2])

            marker.pose.orientation.w = 1.0  # X coordinate of the point
            marker.pose.orientation.x = 0.0  # X coordinate of the point
            marker.pose.orientation.y = 0.0  # Y coordinate of the point
            marker.pose.orientation.z = 0.0  # Z coordinate of the point

            self.marker_publisher.publish(marker)

    def _show(self):
        if self.mode == "on":
            # self.timer = self.create_timer(0.5, self.get_transform)
            cv2.imshow("Depth image with grasping", self.image_shown)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()

    def get_transform(self):
        if self.mode == "on":
            try:
                # Lookup transform from 'base_link' to 'target_link'
                trans = self.tf_buffer.lookup_transform(
                    "world", "camera_link", rclpy.time.Time()
                )
                self.print_coordinates(transform=trans)
            except Exception as e:
                self.get_logger().warn(f"Could not transform: {e}")

    def print_coordinates(self, transform: TransformStamped):
        translation = transform.transform.translation
        # rotation = transform.transform.rotation
        self.end_link_x, self.end_link_y, self.end_link_z = (
            translation.x,
            translation.y,
            translation.z,
        )

    def image_analysis(
        self,
    ):
        if (
            self.mode == "on"
            and self.depth_image is not None
            and self.rgb_image is not None
        ):
            self.image_shown = self.rgb_image
            try:
                # Convert ROS Image message to OpenCV image
                results = self.model(self.rgb_image)
            except Exception as e:
                self.get_logger().error(
                    "Error processing frame through YOLO model: {}".format(e)
                )

            for result in results:
                if result.masks is None:
                    self.get_logger().info("No detections")
                    msg = Float32MultiArray()
                    msg.data = []  # Example float array
                    self.grasping_point_pub.publish(msg)
                    break

                for _, mask in enumerate(result.masks.data):
                    mask = mask.cpu().numpy() * 255
                    mask = cv2.resize(mask, (self.W, self.H))
                    mask = mask.astype(np.uint8)
                    self.point_cloud_mask = mask
                    end_link_coordinates = [
                        self.end_link_x, self.end_link_y, self.end_link_z
                        ]

                    try:
                        self.generate_grasp(
                            self.depth_image,
                            mask,
                            end_link_coordinates,
                            [self.fx, self.fy, self.cx, self.cy],
                        )

                    except Exception as e:
                        self.get_logger().error(
                            "Error generating grasping: {}".format(e)
                        )
                    break

    def generate_grasp(
        self, depth_image, mask, coordinates, intrisic, interest="center"
    ):
        """
        Generate grasp.

        Returns the grasp position [x, y, z, angle] using object
        mask, depth image, and camera intrinsic parameters.
        """
        m = mask

        # Creating normalized depth image just to visualize.
        normalized_depth_image = depth_image.copy()

        # Getting cropped image with mask
        cropped_depth_image = get_cropped_img(depth_image, mask, show=False)

        # Finding min depth value
        min_depth_value, min_depth_coordinates = get_min_depth_coordinates(
            depth_image=cropped_depth_image, mask=mask
        )
        rect, box = generate_oriented_rectangle(mask)
        if len(min_depth_coordinates) == 0:
            self.get_logger().error(
                "Problem with getting depth min values from"
                " depth image in get_min_depth_coordinates function."
            )
            return

        if rect is not None:
            center, dimensions, angle = rect
            center = tuple(map(int, center))
            width, height = dimensions

            # Ensure the angle is in relation to the x-axis of the image
            if width < height:
                angle -= 90

            grasping_angle = 90 - angle

            first_min_depth_coordinates = (
                min_depth_coordinates[0][1],
                min_depth_coordinates[0][0],
            )
            midpoint = calculate_midpoint(center, first_min_depth_coordinates)

            center_depth_value = cropped_depth_image[center[1], center[0]]
            midpoint_depth_value = cropped_depth_image[
                midpoint[1], midpoint[0]
                ]

            # Drawing points
            cv2.circle(normalized_depth_image, center, 5, (255, 0, 0), -1)
            cv2.circle(
                normalized_depth_image,
                first_min_depth_coordinates, 5, (255, 0, 0), -1
            )
            cv2.circle(normalized_depth_image, midpoint, 5, (255, 0, 0), -1)

            # Create an output image to visualize the result
            output_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Draw the oriented rectangle on the output image
            cv2.drawContours(output_mask, [box], 0, (0, 255, 0), 2)

            # Drawing points
            cv2.circle(output_mask, center, 5, (255, 0, 0), -1)
            cv2.circle(output_mask,
                       first_min_depth_coordinates, 5, (0, 255, 0), -1)
            cv2.circle(output_mask, midpoint, 5, (0, 0, 255), -1)

            if interest == "center":
                point_of_interest = center
                depth_of_interest = center_depth_value
            elif interest == "min_depth":
                point_of_interest = first_min_depth_coordinates
                depth_of_interest = min_depth_coordinates
            elif interest == "midpoint":
                point_of_interest = midpoint
                depth_of_interest = midpoint_depth_value

            # Visualization
            alpha = 0.5
            mask = mask.astype(np.float32)
            mask = mask * 0.001
            normalized_depth_image = cv2.addWeighted(
                mask, 1 - alpha, normalized_depth_image, alpha, 0
            )

            mask_3d = cv2.merge((m, m, m))
            normalized_rgb_image = cv2.addWeighted(
                self.rgb_image, 1, mask_3d, 0.5, 0
                )
            normalized_depth_image = cv2.addWeighted(
                mask, 1 - alpha, normalized_depth_image, alpha, 0
            )

            draw_rotated_rectangle(
                image=normalized_rgb_image,
                center=point_of_interest,
                width=200,
                height=50,
                angle=grasping_angle,
                color=(0, 255, 255),
                thickness=2,
            )
            self.image_shown = normalized_rgb_image

            # Camera intrinsic parameters
            f_x, f_y, c_x, c_y = (
                intrisic
            )
            x_cam, y_cam, z_cam = coordinates

            x_pixel = point_of_interest[0]
            y_pixel = point_of_interest[1]
            D = depth_of_interest
            X_real = (x_pixel - c_x) * D / f_x
            Y_real = (y_pixel - c_y) * D / f_y
            Z_real = D

            # Keep gripper facing up
            pitch_angle = (grasping_angle + 180) % 360 - 180

            if pitch_angle > 90:
                pitch_angle -= 180
            elif pitch_angle < -90:
                pitch_angle += 180

            grasping_angle = pitch_angle
            grasping_angle = 180

            quaternion = [1.0, 0.0, 0.0, 0.0]
            self.grasping_point = [
                x_cam + Z_real,
                y_cam + (-X_real),
                z_cam + (-Y_real),
            ] + quaternion

            if (
                0.0 < self.grasping_point[0] < 0.7
                and -0.6 < self.grasping_point[1] < 0.6
                and 0.0 < self.grasping_point[2] < 0.5
            ):
                self.get_logger().info(
                    "Object real coordinates "
                    f"X: {self.grasping_point[0]} "
                    f"Y: {self.grasping_point[1]} "
                    f"Z: {self.grasping_point[2]}"
                )
                self.get_logger().info(f"{self.space_print}")

                msg = Float32MultiArray()
                msg.data = list(self.grasping_point)
                self.grasping_point_pub.publish(msg)
            else:
                self.get_logger().info("Point out of allowed space.")
                msg = Float32MultiArray()
                msg.data = []
                self.grasping_point_pub.publish(msg)
        else:
            self.get_logger().warn("No contours found in the mask.")

    def create_point_cloud(self):
        step_size = 4
        if self.mode == "on" and self.point_cloud_mask is not None:
            points = []
            for v in range(0, self.H, step_size):
                for u in range(0, self.W, step_size):
                    if self.point_cloud_mask[v, u] == 0:
                        continue

                    z = self.depth_image[v, u]
                    if z == 0:
                        continue
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    b, g, r = self.rgb_image[v, u]

                    rgb = struct.unpack(
                        "I", struct.pack("BBBB", b, g, r, 255)
                        )[0]

                    aux = x
                    x = z
                    z = aux

                    aux = y
                    y = -z
                    z = -aux

                    points.append([x, y, z, rgb])

            header = Header()
            header.frame_id = "camera_link"

            fields = [
                point_cloud2.PointField(
                    name="x",
                    offset=0,
                    datatype=point_cloud2.PointField.FLOAT32,
                    count=1,
                ),
                point_cloud2.PointField(
                    name="y",
                    offset=4,
                    datatype=point_cloud2.PointField.FLOAT32,
                    count=1,
                ),
                point_cloud2.PointField(
                    name="z",
                    offset=8,
                    datatype=point_cloud2.PointField.FLOAT32,
                    count=1,
                ),
                point_cloud2.PointField(
                    name="rgb",
                    offset=12,
                    datatype=point_cloud2.PointField.UINT32,
                    count=1,
                ),
            ]
            header.stamp = self.get_clock().now().to_msg()
            point_cloud = point_cloud2.create_cloud(header, fields, points)
            self.point_cloud_publisher.publish(point_cloud)


def main(args=None):
    rclpy.init(args=args)
    image_depth_yolo = ImageRGBYOLO()
    rclpy.spin(image_depth_yolo)
    image_depth_yolo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
