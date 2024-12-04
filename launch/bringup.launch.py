from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    # Declare a launch argument to choose the control launch option
    type_launch_arg = DeclareLaunchArgument(
        'type_launch',
        default_value='simulation',
        description='Specify the type of launch: "real" or "simulation".'
    )

    # Get the value of the control launch argument
    type_launch_option = LaunchConfiguration('type_launch')

    # Dynamixel control group
    real_group = GroupAction(actions=[
        ExecuteProcess(
            cmd=[
                'gnome-terminal',
                '--',
                'ros2',
                'launch',
                'open_manipulator_pro_control',
                'dynamixel_control.launch.py'],
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'gnome-terminal',
                '--',
                'ros2',
                'launch',
                'manipulator_trajectory_vision_gui',
                'detection_real.launch.py'],
            output='screen'
        ),
    ])

    # Alternative control group
    simulation_group = GroupAction(actions=[
        ExecuteProcess(
            cmd=[
                'gnome-terminal',
                '--',
                'ros2',
                'launch',
                'open_manipulator_pro_gazebo',
                'simulation.launch.py'],
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'gnome-terminal',
                '--',
                'ros2',
                'launch',
                'manipulator_trajectory_vision_gui',
                'detection_simulation.launch.py'],
            output='screen'
        ),
    ])

    # Common nodes
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

    return LaunchDescription([
        type_launch_arg,
        # Dynamically select control group based on argument
        GroupAction(
            actions=[
                real_group
            ],
            condition=IfCondition(
                PythonExpression(
                    ["'", type_launch_option, "' == 'real'"]
                    )
                )
        ),
        GroupAction(
            actions=[
                simulation_group
            ],
            condition=IfCondition(
                PythonExpression(
                    ["'", type_launch_option, "' == 'simulation'"]
                    )
                )
        ),
        omp
    ])
