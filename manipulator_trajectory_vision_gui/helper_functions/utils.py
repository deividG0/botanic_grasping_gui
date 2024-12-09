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
import cv2


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion representation.

    Parameters:
        roll: float - Roll angle in radians.
        pitch: float - Pitch angle in radians.
        yaw: float - Yaw angle in radians.

    Returns:
        numpy.array - Quaternion [w, x, y, z].
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]


def show_img(img, title="Depth Image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)


def draw_rotated_rectangle(
                        image,
                        center,
                        width,
                        height,
                        angle,
                        color,
                        thickness
                        ):
    """
    Draw a rotated rectangle on an image.

    :param image: The image to draw the rectangle on.
    :param center: The center of the rectangle (x, y).
    :param width: The width of the rectangle.
    :param height: The height of the rectangle.
    :param angle: The angle of inclination of the rectangle in degrees.
    :param color: The color of the rectangle (BGR).
    :param thickness:
        The thickness of the rectangle border. Use -1 for filled rectangle.
    """
    # Define the rectangle's corner points before rotation
    rect = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )

    # Define the rotation matrix
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Rotate the rectangle's corner points and translate to the center
    rotated_rect = np.dot(rect, R) + np.array(center)

    # Convert points to integer
    rotated_rect = rotated_rect.astype(np.int32)

    # Draw the rectangle
    cv2.polylines(
        image, [rotated_rect], isClosed=True, color=color, thickness=thickness
    )


def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


def get_cropped_img(
    depth_image, mask_image, show=False, crop_size=300, is_cropping=False
):
    # Find the minimum depth value in the depth image
    max_depth = np.max(depth_image)

    # Create a mask for the toy region and for the background
    img_mask = cv2.bitwise_and(depth_image, depth_image, mask=mask_image)
    background_mask = cv2.bitwise_not(mask_image)

    # Fill the background with the minimum depth value
    background = np.full_like(depth_image, max_depth)
    background_filled = cv2.bitwise_and(
                background,
                background,
                mask=background_mask)

    # Combine the region of interest and the filled background
    depth_filled = cv2.add(img_mask, background_filled)

    # Find contours to get bounding box
    contours, _ = cv2.findContours(
        mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x, y, w, h = cv2.boundingRect(contours[0])

    if is_cropping:
        img_cropped = depth_filled[
            y: y + h,
            x: x + w
            ]
        img_cropped_resized = cv2.resize(img_cropped, (crop_size, crop_size))
    else:
        img_cropped_resized = depth_filled

    if show:
        show_img(
            normalize_array(img_cropped_resized),
            title="Cropped img with Filled Background",
        )

    return img_cropped_resized


def generate_oriented_rectangle(mask):
    # Ensure the mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours from the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None  # No contours found

    # Assume the largest contour is the object of interest
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the minimum area rectangle around the largest contour
    rect = cv2.minAreaRect(largest_contour)

    # Get the box points and convert them to integer
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    return rect, box


def draw_rectangle(
                image,
                center,
                width,
                height,
                color=(0, 255, 0),
                thickness=2
                ):
    """
    Draws a rectangle on an image using the center point, width, and height.

    Parameters:
        image (numpy.ndarray): The input image on which to draw the rectangle.
        center (tuple): The (x, y) coordinates of the center of the rectangle.
        width (int): The width of the rectangle.
        height (int): The height of the rectangle.
        color (tuple):
            The color of the rectangle in BGR format (default is green).
        thickness (int): The thickness of the rectangle border (default is 2).

    Returns:
        numpy.ndarray: The image with the rectangle drawn on it.
    """
    width = int(width)
    height = int(height)

    # Calculate the top-left and bottom-right coordinates of the rectangle
    top_left = (int(center[0] - width / 2), int(center[1] - height / 2))
    bottom_right = (int(center[0] + width / 2), int(center[1] + height / 2))

    # Draw the rectangle on the image
    cv2.rectangle(image, top_left, bottom_right, color, thickness)


def get_min_depth_coordinates(depth_image, mask):
    # Apply the mask to the depth image
    masked_depth = np.where(mask, depth_image, np.inf)
    # Find the minimum depth value
    min_depth = np.min(masked_depth)
    # Find the coordinates of the minimum depth value
    min_depth_coordinates = np.argwhere(masked_depth == min_depth)

    return min_depth, min_depth_coordinates


def calculate_midpoint(point1, point2):
    # Extract coordinates of each point
    x1, y1 = point1
    x2, y2 = point2

    # Calculate midpoint coordinates
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2

    return (int(midpoint_x), int(midpoint_y))
