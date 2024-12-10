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
import trimesh
import os
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from urdf_parser_py.urdf import URDF
from transforms3d.euler import euler2mat
from ament_index_python.packages import get_package_share_directory


class OMPCollisionChecker:
    def __init__(self):
        # Checker variables
        share_directory = get_package_share_directory(
            "botanic_grasping_gui"
            )
        root_path = os.path.join(
            share_directory,
            "resource",
            "total_description"
            )
        urdf_file = root_path + "/open_manipulator_pro.urdf"
        self.robot = self.parse_urdf(urdf_file)
        self.list_of_links = ["link1", "link2", "link3", "link4", "link6"]
        self.list_of_meshes = self.load_meshes(root_path)

    def load_meshes(self, root_path):
        list_of_meshes = {}
        for link in self.list_of_links:
            mesh = trimesh.load_mesh(root_path + "/" + link + ".stl")
            list_of_meshes[link] = mesh
        return list_of_meshes

    def parse_urdf(self, urdf_file):
        return URDF.from_xml_file(urdf_file)

    def compute_transformations(self, robot, joint_angles):
        # Dictionary to store the transformation matrices for each link
        transformations = {}
        # Identity matrix for the base link
        T = np.eye(4)
        # Dictionary to store the current joint angles
        joint_angle_dict = joint_angles

        for joint in robot.joints:
            # Get the parent and child links
            child_link = joint.child

            # Get the joint origin (position and orientation)
            origin_xyz = joint.origin.xyz if joint.origin else [0, 0, 0]
            origin_rpy = joint.origin.rpy if joint.origin else [0, 0, 0]

            # Compute the transformation from
            # the parent link to the joint origin
            T_joint_origin = np.eye(4)
            T_joint_origin[:3, :3] = euler2mat(*origin_rpy)
            T_joint_origin[:3, 3] = origin_xyz

            # Compute the transformation due to the joint angle
            if joint.type == "revolute" or joint.type == "continuous":
                angle = joint_angle_dict[joint.name]
                axis = joint.axis
                R_joint = np.eye(4)
                axis_angle = [x * angle for x in axis]
                R_joint[:3, :3] = euler2mat(*(axis_angle))
            elif joint.type == "prismatic":
                displacement = joint_angle_dict[joint.name]
                axis = joint.axis
                R_joint = np.eye(4)
                R_joint[:3, 3] = axis * displacement
            else:
                R_joint = np.eye(4)

            # Compute the total transformation for the child link
            T = np.dot(T, np.dot(T_joint_origin, R_joint))
            # Store the transformation matrix for the child link
            transformations[child_link] = T

        return transformations

    def plot_parallelepiped(self, edge_points, transformation_matrix):
        # Convert numpy array to a list of lists
        list_of_lists = edge_points.tolist()

        # Add the value to each list inside the list of lists
        for sublist in list_of_lists:
            sublist.append(1)

        unit_cube_vertices = np.array(list_of_lists)

        # Apply the transformation matrix to the vertices
        transformed_vertices = np.dot(
            unit_cube_vertices, transformation_matrix.T
            )

        # Remove the homogeneous coordinate
        transformed_vertices = transformed_vertices[:, :3]

        # Define the 12 edges of the cube
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        # Plot the parallelepiped
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the edges
        for edge in edges:
            ax.plot3D(*zip(*transformed_vertices[edge]), color="b")

        # Plot the vertices
        ax.scatter3D(*transformed_vertices.T, color="r")

        # Set the limits and labels
        max_range = (
            np.array(
                [
                    transformed_vertices[:, 0].max()
                    - transformed_vertices[:, 0].min(),
                    transformed_vertices[:, 1].max()
                    - transformed_vertices[:, 1].min(),
                    transformed_vertices[:, 2].max()
                    - transformed_vertices[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (
            transformed_vertices[:, 0].max() + transformed_vertices[:, 0].min()
        ) * 0.5
        mid_y = (
            transformed_vertices[:, 1].max() + transformed_vertices[:, 1].min()
        ) * 0.5
        mid_z = (
            transformed_vertices[:, 2].max() + transformed_vertices[:, 2].min()
        ) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()

    def plot_parallelepipeds(
            self, list_edges, transformations, show_mesh=True
            ):
        # Plot the parallelepiped
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        list_of_colors = [
            "r",
            "b",
            "g",
            "#b103fc",
            "#03fc6b",
            "#fc9803",
            "#6942f5",
            "#39383d",
        ]

        for i, edge_points in enumerate(list_edges):
            # Define the 12 edges of the cube
            edges = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]

            # Plot the edges
            for edge in edges:
                ax.plot3D(*zip(*edge_points[edge]), color=list_of_colors[i])

            if i < len(self.list_of_links):
                if show_mesh:
                    current_mesh = self.list_of_meshes[self.list_of_links[i]]
                    mesh_vertices = (
                        current_mesh.vertices * 0.001) @ transformations[
                        self.list_of_links[i]
                    ][:3, :3].T + transformations[self.list_of_links[i]][:3, 3]
                    mesh_faces = current_mesh.faces
                    mesh_collection = Poly3DCollection(
                        mesh_vertices[mesh_faces],
                        alpha=0.1,
                        facecolor="cyan",
                        edgecolor="k",
                    )
                    ax.add_collection3d(mesh_collection)

            # Plot the vertices
            ax.scatter3D(*edge_points.T, color="r")

        ax.set_xlim(-0.2, 0.4)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(0.0, 0.6)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()

    def gen_cubes(self, size):
        height = 0.0
        return np.array(
            [
                [0, 0, 0 + height],
                [size, 0, 0 + height],
                [size, size, 0 + height],
                [0, size, 0 + height],
                [0, 0, size + height],
                [size, 0, size + height],
                [size, size, size + height],
                [0, size, size + height],
            ]
        )

    def get_parallelepiped_by_mesh(self, transformations):
        list_of_parallelepiped = []

        for link in self.list_of_links:
            if self.list_of_meshes[link]:
                # Read the mesh file
                mesh = self.list_of_meshes[link]

                # Set plot limits
                limits = mesh.bounding_box_oriented.extents
                scale = 0.001

                x, y, z = limits * scale
                print(f"x, y, z: -----> {x, y, z}")

                if link in ["link1", "link5"]:
                    list_of_parallelepiped.append(
                        np.array(
                            [
                                [-x / 2, -y / 2, 0],
                                [x / 2, -y / 2, 0],
                                [x / 2, y / 2, 0],
                                [-x / 2, y / 2, 0],
                                [-x / 2, -y / 2, z],
                                [x / 2, -y / 2, z],
                                [x / 2, y / 2, z],
                                [-x / 2, y / 2, z],
                            ]
                        )
                    )
                elif link in ["link2"]:
                    z_adjust = 0.01
                    list_of_parallelepiped.append(
                        np.array(
                            [
                                [-x / 2, -y / 2, 0 + z_adjust],
                                [x / 2, -y / 2, 0 + z_adjust],
                                [x / 2, y / 2, 0 + z_adjust],
                                [-x / 2, y / 2, 0 + z_adjust],
                                [-x / 2, -y / 2, z + z_adjust],
                                [x / 2, -y / 2, z + z_adjust],
                                [x / 2, y / 2, z + z_adjust],
                                [-x / 2, y / 2, z + z_adjust],
                            ]
                        )
                    )
                elif link in ["link3"]:
                    x_offset = 0.035
                    y_adjust = -0.069
                    y_offset = 0.018
                    z_offset = 0.07
                    list_of_parallelepiped.append(
                        np.array(
                            [
                                [
                                    -x / 2 + x_offset,
                                    -y / 2 + y_adjust + y_offset,
                                    0 + z_offset,
                                ],
                                [
                                    x / 2 - x_offset,
                                    -y / 2 + y_adjust + y_offset,
                                    0 + z_offset,
                                ],
                                [
                                    x / 2 - x_offset,
                                    y / 2 + y_adjust - y_offset,
                                    0 + z_offset,
                                ],
                                [
                                    -x / 2 + x_offset,
                                    y / 2 + y_adjust - y_offset,
                                    0 + z_offset,
                                ],
                                [
                                    -x / 2 + x_offset,
                                    -y / 2 + y_adjust + y_offset,
                                    z - z_offset - 0.01,
                                ],
                                [
                                    x / 2 - x_offset,
                                    -y / 2 + y_adjust + y_offset,
                                    z - z_offset - 0.01,
                                ],
                                [
                                    x / 2 - x_offset,
                                    y / 2 + y_adjust - y_offset,
                                    z - z_offset - 0.01,
                                ],
                                [
                                    -x / 2 + x_offset,
                                    y / 2 + y_adjust - y_offset,
                                    z - z_offset - 0.01,
                                ],
                            ]
                        )
                    )
                elif link in ["link4"]:
                    x_adjust = +0.13
                    x_offset = 0.03
                    y_adjust = -0.0575
                    y_offset = 0.009
                    z_adjust = -0.02
                    z_offset = 0.02
                    list_of_parallelepiped.append(
                        np.array(
                            [
                                [
                                    -x / 2 + x_adjust + x_offset,
                                    -y / 2 + y_adjust + y_offset,
                                    0 + z_adjust + z_offset,
                                ],
                                [
                                    x / 2 + x_adjust - x_offset,
                                    -y / 2 + y_adjust + y_offset,
                                    0 + z_adjust + z_offset,
                                ],
                                [
                                    x / 2 + x_adjust - x_offset,
                                    y / 2 + y_adjust - y_offset,
                                    0 + z_adjust + z_offset,
                                ],
                                [
                                    -x / 2 + x_adjust + x_offset,
                                    y / 2 + y_adjust - y_offset,
                                    0 + z_adjust + z_offset,
                                ],
                                [
                                    -x / 2 + x_adjust + x_offset,
                                    -y / 2 + y_adjust + y_offset,
                                    z + z_adjust - z_offset,
                                ],
                                [
                                    x / 2 + x_adjust - x_offset,
                                    -y / 2 + y_adjust + y_offset,
                                    z + z_adjust - z_offset,
                                ],
                                [
                                    x / 2 + x_adjust - x_offset,
                                    y / 2 + y_adjust - y_offset,
                                    z + z_adjust - z_offset,
                                ],
                                [
                                    -x / 2 + x_adjust + x_offset,
                                    y / 2 + y_adjust - y_offset,
                                    z + z_adjust - z_offset,
                                ],
                            ]
                        )
                    )
                elif link in ["link6"]:
                    x_adjust = +0.06
                    x_offset = +0.01
                    y_adjust = -0.045
                    z_adjust = -0.018
                    z_offset = 0.005

                    list_of_parallelepiped.append(
                        np.array(
                            [
                                [
                                    -x / 2 + x_adjust + x_offset,
                                    -y / 2 + y_adjust,
                                    0 + z_adjust - z_offset,
                                ],
                                [
                                    x / 2 + x_adjust - x_offset,
                                    -y / 2 + y_adjust,
                                    0 + z_adjust - z_offset,
                                ],
                                [
                                    x / 2 + x_adjust - x_offset,
                                    y / 2 + y_adjust,
                                    0 + z_adjust - z_offset,
                                ],
                                [
                                    -x / 2 + x_adjust + x_offset,
                                    y / 2 + y_adjust,
                                    0 + z_adjust - z_offset,
                                ],
                                [
                                    -x / 2 + x_adjust + x_offset,
                                    -y / 2 + y_adjust,
                                    z + z_adjust + z_offset,
                                ],
                                [
                                    x / 2 + x_adjust - x_offset,
                                    -y / 2 + y_adjust,
                                    z + z_adjust + z_offset,
                                ],
                                [
                                    x / 2 + x_adjust - x_offset,
                                    y / 2 + y_adjust,
                                    z + z_adjust + z_offset,
                                ],
                                [
                                    -x / 2 + x_adjust + x_offset,
                                    y / 2 + y_adjust,
                                    z + z_adjust + z_offset,
                                ],
                            ]
                        )
                    )

        # Additional parallelepipeds
        x_size, y_size, z_size = 0.06, 0.07, 0.03
        x_adjust = +0.1
        y_adjust = -0.0

        list_of_parallelepiped.append(
            np.array(
                [
                    [-x_size + x_adjust, -y_size + y_adjust, -z_size],
                    [x_size + x_adjust, -y_size + y_adjust, -z_size],
                    [x_size + x_adjust, y_size + y_adjust, -z_size],
                    [-x_size + x_adjust, y_size + y_adjust, -z_size],
                    [-x_size + x_adjust, -y_size + y_adjust, z_size],
                    [x_size + x_adjust, -y_size + y_adjust, z_size],
                    [x_size + x_adjust, y_size + y_adjust, z_size],
                    [-x_size + x_adjust, y_size + y_adjust, z_size],
                ]
            )
        )

        new_list_of_parallelepipeds = []

        for i, edge_points in enumerate(list_of_parallelepiped):
            # Convert numpy array to a list of lists
            list_of_lists = edge_points.tolist()

            # Add the value to each list inside the list of lists
            for sublist in list_of_lists:
                sublist.append(1)
            unit_cube_vertices = np.array(list_of_lists)

            # Apply the transformation matrix to the vertices
            if i >= len(self.list_of_links):
                # If We am dealing with parallelepipeds farther than the link6
                # We will use the transformation for the link6
                # and adjust the parallelepiped it self.
                transformed_vertices = np.dot(
                    unit_cube_vertices, transformations["end_link"].T
                )
            else:
                transformed_vertices = np.dot(
                    unit_cube_vertices,
                    transformations[self.list_of_links[i]].T
                )

            # Remove the homogeneous coordinate
            transformed_vertices = transformed_vertices[:, :3]

            new_list_of_parallelepipeds.append(transformed_vertices)

        self.parallelepipeds = new_list_of_parallelepipeds
        print("self.parallelepipeds:", self.parallelepipeds)
        return new_list_of_parallelepipeds

    def check_overlap(self, list_of_parallelepipeds):
        def vertices_to_edges(vertices):
            edges = []
            edges_idx = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]

            # Plot the edges
            for edge in edges_idx:
                edge = vertices[edge[0]] - vertices[edge[1]]
                edges.append(edge)

            return np.array(edges)

        def generate_double_combinations(items):
            return list(itertools.combinations(items, 2))

        def normalize(v):
            norm = np.linalg.norm(v)
            return v if norm == 0 else v / norm

        # Generate all possible double combinations
        double_combinations = generate_double_combinations(
            list_of_parallelepipeds
            )
        check_overlap_doubles = []

        # Cross products of edges
        for double in double_combinations:
            bool_double_overlay = True
            edges1 = vertices_to_edges(double[0])
            edges2 = vertices_to_edges(double[1])

            axes = []

            # Face normals
            for i in range(3):
                axes.append(normalize(edges1[i]))
                axes.append(normalize(edges2[i]))

            for edge1 in edges1:
                for edge2 in edges2:
                    axis = np.cross(edge1, edge2)
                    if np.linalg.norm(axis) > 1e-8:  # Avoid zero vectors
                        axes.append(normalize(axis))

            for axis in axes:
                proj1 = np.dot(double[0], axis)
                proj2 = np.dot(double[1], axis)

                if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                    bool_double_overlay = False
            check_overlap_doubles.append(bool_double_overlay)

        return any(check_overlap_doubles)

    def check_self_collision(self, joint_angles):
        transformations = self.compute_transformations(
            self.robot, joint_angles
            )
        list_of_parallelepiped = self.get_parallelepiped_by_mesh(
            transformations
            )

        is_overlapping = False

        if self.check_overlap(list_of_parallelepiped):
            print("Overlap.")
            is_overlapping = True
        else:
            print("Do not overlap.")
            is_overlapping = False

        return is_overlapping, self.parallelepipeds
