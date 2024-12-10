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

import rclpy
import numpy as np
import pyrealsense2 as rs
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher_node')

        # Publishers for RGB and Depth images
        self.rgb_pub = self.create_publisher(
            Image, 'realsense/rgb/image_raw', 10
            )
        self.depth_pub = self.create_publisher(
            Image, 'realsense/depth/image_raw', 10
            )

        try:
            # Initialize RealSense pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(config)

            align_to = rs.stream.color
            self.align = rs.align(align_to)

            # Bridge to convert between OpenCV and ROS2 images
            self.bridge = CvBridge()

            # Timer to publish frames at 30Hz
            self.timer = self.create_timer(1/30.0, self.publish_frames)
            self.get_logger().info('Realsense frames being published.')
        except Exception as e:
            self.get_logger().error(
                f'Error trying to load realsense camera: {e}.'
                )

    def publish_frames(self):
        # Wait for a coherent pair of frames: RGB and Depth
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not color_frame or not depth_frame:
            return

        # Convert RealSense frames to OpenCV format
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert OpenCV images to ROS2 Image messages
        rgb_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')

        # Publish the images
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)

    def destroy(self):
        self.pipeline.stop()
        super().destroy()


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
