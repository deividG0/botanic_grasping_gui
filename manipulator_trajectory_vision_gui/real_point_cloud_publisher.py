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

import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
from std_msgs.msg import (
    Float32MultiArray,
    Int32MultiArray,
)
import numpy as np
import struct


class PointCloudPublisher(Node):

    def __init__(self):
        super().__init__("point_cloud_publisher")
        self.point_cloud_publisher = self.create_publisher(
            PointCloud2, "/camera/point_cloud", 100
        )
        self.depth_image_pub = self.create_publisher(
            Float32MultiArray, "depth_image", 10
        )
        self.rgb_image_pub = self.create_publisher(
            Int32MultiArray, "rgb_image", 10
            )

        self.bridge = CvBridge()

        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipe.start(self.cfg)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.timer = self.create_timer(0.1, self.capture_and_publish)

    def capture_and_publish(self):
        frame = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frame)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_frame = np.asanyarray(depth_frame.get_data())
        rgb_frame = np.asanyarray(color_frame.get_data())

        # Assuming depth_frame is already in meters or needs to be converted
        depth_frame = depth_frame.astype(np.float32) / 1000.0

        print(
            f"Publishing: depth_frame: {depth_frame}, rgb_frame: {rgb_frame}"
            )

        h, w = depth_frame.shape
        fx = 528.433756558705
        fy = 528.433756558705
        cx = 320.5
        cy = 240.5

        points = []
        for v in range(h):
            for u in range(w):
                z = depth_frame[v, u]
                if z == 0:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                b, g, r = rgb_frame[v, u]

                rgb = struct.unpack("I", struct.pack("BBBB", b, g, r, 255))[0]

                aux = x
                x = z
                z = aux

                aux = y
                y = z
                z = aux

                points.append([x, y, z, rgb])

        header = Header()
        header.frame_id = "camera_link"

        fields = [
            point_cloud2.PointField(
                name="x",
                offset=0,
                datatype=point_cloud2.PointField.FLOAT32,
                count=1
            ),
            point_cloud2.PointField(
                name="y",
                offset=4,
                datatype=point_cloud2.PointField.FLOAT32,
                count=1
            ),
            point_cloud2.PointField(
                name="z",
                offset=8,
                datatype=point_cloud2.PointField.FLOAT32,
                count=1
            ),
            point_cloud2.PointField(
                name="rgb",
                offset=12,
                datatype=point_cloud2.PointField.UINT32,
                count=1
            ),
        ]

        header.stamp = self.get_clock().now().to_msg()
        point_cloud = point_cloud2.create_cloud(header, fields, points)
        self.point_cloud_publisher.publish(point_cloud)

    def destroy(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    point_cloud_publisher = PointCloudPublisher()
    try:
        rclpy.spin(point_cloud_publisher)
    except KeyboardInterrupt:
        pass
    point_cloud_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
