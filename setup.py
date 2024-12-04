import os
from setuptools import setup

PACKAGE_NAME = 'manipulator_trajectory_vision_gui'
PACKAGES_LIST = [
    PACKAGE_NAME,
    'manipulator_trajectory_vision_gui.pykin_omp',
    'manipulator_trajectory_vision_gui.helper_functions',
    'manipulator_trajectory_vision_gui.checking_self_collision',
    'manipulator_trajectory_vision_gui',
]

DATA_FILES = [
        ('share/ament_index/resource_index/packages',
            ['resource/' + PACKAGE_NAME]),
        ('share/' + PACKAGE_NAME, ['resource/images/reference_image.jpg']),
        ('share/' + PACKAGE_NAME, ['package.xml'])
    ]


def package_files(data_files, directory_list):
    """
    Get all files in a directory and subdirectory and return a list of tuples.

    Parameters
    ----------
    data_files : list
        List of tuples containing the path to install the files and the files
        themselves.
    directory_list : list
        List of directories to get the files from.

    Returns
    -------
    data_files : list
        List of tuples containing the path to install the files and the files
        themselves.

    """
    paths_dict = {}
    for directory in directory_list:
        for (path, _, filenames) in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', PACKAGE_NAME, path)

                if install_path in paths_dict:
                    paths_dict[install_path].append(file_path)
                else:
                    paths_dict[install_path] = [file_path]

    for key, value in paths_dict.items():
        data_files.append((key, value))

    return data_files


setup(
    name=PACKAGE_NAME,
    version='0.0.0',
    packages=PACKAGES_LIST,
    data_files=package_files(DATA_FILES, ['config/', 'launch/', 'resource/']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cimatec',
    maintainer_email='cimatec@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Nodes
            "real_point_cloud_publisher = "
            "manipulator_trajectory_vision_gui.real_point_cloud_publisher:main",
            "realsense_image_reader = "
            " manipulator_trajectory_vision_gui.realsense_image_reader:main",
            "image_rgb_yolo = "
            " manipulator_trajectory_vision_gui.image_rgb_yolo:main",
            "image_rgb_yolo_real = "
            " manipulator_trajectory_vision_gui.image_rgb_yolo_real:main",
            "omp_controller = "
            " manipulator_trajectory_vision_gui.omp_controller:main",
            "omp_joint_trajectory = "
            " manipulator_trajectory_vision_gui.omp_joint_trajectory:main",
            "gripper_joint_trajectory = "
            " manipulator_trajectory_vision_gui.gripper_joint_trajectory:main",
            "node_debug_pykin_real = "
            " manipulator_trajectory_vision_gui.node_debug_pykin_real:main",
            "camera_calibration_node = "
            " manipulator_trajectory_vision_gui.camera_calibration_node:main",

            "omp_ros2 = manipulator_trajectory_vision_gui.omp_ros2:main"
        ],
    },
)
