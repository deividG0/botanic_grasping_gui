from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit


def generate_launch_description():
    # First node execution
    gazebo = ExecuteProcess(
            cmd=[
                    'gnome-terminal',
                    '--',
                    'ros2',
                    'launch',
                    'open_manipulator_pro_gazebo',
                    'simulation.launch.py'],
            output='screen'
        )

    omp = ExecuteProcess(
            cmd=[
                    'gnome-terminal',
                    '--',
                    'ros2',
                    'launch',
                    'manipulator_trajectory_vision_gui',
                    'omp_ros2.launch.py'],
            output='screen'
        )

    yolo_real = ExecuteProcess(
            cmd=[
                    'gnome-terminal',
                    '--',
                    'ros2',
                    'launch',
                    'manipulator_trajectory_vision_gui',
                    'detection_simulation.launch.py'],
            output='screen'
        )

    return LaunchDescription([
        gazebo,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gazebo,
                on_exit=[
                    omp
                ]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=omp,
                on_exit=[
                    yolo_real
                ]
            )
        )
    ])
