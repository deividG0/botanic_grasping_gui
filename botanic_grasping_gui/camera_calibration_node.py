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
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String
from skimage.metrics import structural_similarity as ssim


class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration_node')
        # Load the reference image (Make sure it's in the correct path)
        share_directory = get_package_share_directory(
            "botanic_grasping_gui"
            )
        root_path = os.path.join(
            share_directory, "resource", "images", "reference_image.jpg"
            )
        self.reference_image = cv2.imread(root_path)

        if self.reference_image is None:
            self.get_logger().error("Reference image not found! "
                                    "Ensure the path is correct.")
            raise FileNotFoundError("Reference image not found!")

        self.get_logger().info("Reference image loaded successfully.")

        self.bridge = CvBridge()
        self.rgb_sub = self.create_subscription(
            Image, 'realsense/rgb/image_raw', self.rgb_callback, 10
            )

        self.mode = 'off'
        self.create_subscription(String,
                                 '/switch_cam_calibration_mode',
                                 self.switch,
                                 10)

    def switch(self, msg):
        self.mode = msg.data

    def rgb_callback(self, msg):
        if self.mode == 'on':
            # Convert the ROS2 Image message to an OpenCV image
            color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Calculate similarity between
            # the reference image and the current frame
            similarity_score = self.calculate_similarity(
                self.reference_image, color_image
                )

            # Overlay the reference image onto the current frame
            overlay = cv2.addWeighted(
                self.reference_image, 0.4, color_image, 0.6, 0
                )

            # Display similarity score on the overlay
            cv2.putText(
                overlay, f'Similarity: {similarity_score:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                )

            # Show the overlay in an OpenCV window
            cv2.imshow('RealSense Overlay with Similarity', overlay)
            cv2.waitKey(1)

        else:
            cv2.destroyAllWindows()

    def calculate_similarity(self, frame1, frame2):
        """Calculate similarity using SSIM."""
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Resize frame2 (current frame) to match the reference image dimensions
        gray_frame2_resized = cv2.resize(
            gray_frame2, (gray_frame1.shape[1], gray_frame1.shape[0])
            )

        # Compute SSIM
        score, _ = ssim(gray_frame1, gray_frame2_resized, full=True)
        return score

    def destroy(self):
        """Cleanup method to stop RealSense pipeline and close windows."""
        self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy()


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
