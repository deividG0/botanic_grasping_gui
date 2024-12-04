# **Botanic Grasping**  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## **Overview**
Repository to group trajectory algorithms configurations for the OpenManipulatorPro to identification and grasping of fruits.

### **License**

This project is licensed under the [MIT License](LICENSE).

## **Requirements**

The `botanig_grasping` package has been tested under:

- ROS2 [`Humble Hawksbill`](https://docs.ros.org/en/humble/Releases/Release-Humble-Hawksbill.html) and Ubuntu 22.04 LTS (Jammy Jellyfish).

Ahead its presented the packages and libraries used in the repository with its own installation instructions link:
- PIP installation `sudo apt install python3-pip -y`
- [librealsense2](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)
- [pytorch](https://pytorch.org/)
- [ultralytics](https://docs.ultralytics.com/quickstart/#install-ultralytics)
- [vcstool](https://github.com/dirk-thomas/vcstool)
- *[numpy](https://pypi.org/project/numpy/)
- *[opencv](https://pypi.org/project/opencv-python/)
- *[mediapipe](https://pypi.org/project/mediapipe/)
- *[pyrealsense2](https://pypi.org/project/pyrealsense2/)
- *[transformations](https://pypi.org/project/transformations/)
- *[urdf-parser-py](https://pypi.org/project/urdf-parser-py/)
- *[trimesh](https://pypi.org/project/trimesh/)
- *[scikit-image](https://pypi.org/project/scikit-image/)

For the packages marked with * just use `pip install -r requirements.txt` after cloning the repository.

## **Installation**
1. Clone this repository into your workspace:
    ```bash
    cd ~/ros2_ws/src
    git clone https://github.com/deividG0/botanic_grasping_gui.git
    ```
2. Install dependencies:
    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    ```
3. Build the workspace:
    ```bash
    cd ~/ros2_ws
    colcon build
    ```

## **Usage**

For launch project simulation, run the following command:

```
ros2 launch manipulator_trajectory_vision_gui bringup.launch.py type_launch:=simulation
```

To use in real scenario, run the following command:
```
ros2 launch manipulator_trajectory_vision_gui bringup.launch.py type_launch:=real
```

### **Configuration**

**[params.yaml](manual_controller/config/params.yaml):** Parameters for trajectory planning for the manipulator and default poses.

### **Contributing**

To contribute to this package, you can either [open an issue](https://github.com/deividG0/botanic-grasping/issues) describing the desired subject or develop the feature yourself and [submit a pull request](https://github.com/deividG0/botanic-grasping/pulls) to the main branch.

If you choose to develop the feature yourself, please adhere to the [ROS 2 Code style and language] guidelines to improve code readability and maintainability.

