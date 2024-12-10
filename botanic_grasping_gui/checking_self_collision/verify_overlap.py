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
import itertools


def check_overlap(list_of_parallelepipeds):
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

    def separating_axis_test(vertices1, vertices2):
        edges1 = vertices_to_edges(vertices1)
        edges2 = vertices_to_edges(vertices2)

        axes = []

        # Face normals
        for i in range(3):
            axes.append(normalize(edges1[i]))
            axes.append(normalize(edges2[i]))

        # Cross products of edges
        for edge1 in edges1:
            for edge2 in edges2:
                axis = np.cross(edge1, edge2)
                if np.linalg.norm(axis) > 1e-8:  # Avoid zero vectors
                    axes.append(normalize(axis))

        for axis in axes:
            proj1 = np.dot(vertices1, axis)
            proj2 = np.dot(vertices2, axis)

            if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                return False  # Separating axis found

        return True

    # Generate all possible double combinations
    double_combinations = generate_double_combinations(list_of_parallelepipeds)
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
                # return False  # Separating axis found
        check_overlap_doubles.append(bool_double_overlay)

    return any(check_overlap_doubles)
