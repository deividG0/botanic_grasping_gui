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

"""Tkinter GUI class for the OMP Simulation."""
import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
from ament_index_python.packages import get_package_share_directory

from manipulator_trajectory_vision_gui.helper_functions.load_ros_parameters \
    import load_yaml_file


def is_valid_number(input_str: str):
    """Check if the input is a valid number.

    Parameters
    ----------
    input_str : str
        The input string.

    Returns
    -------
    bool
        True if the input is a valid number, False otherwise.

    """
    try:
        float(input_str)
        return True
    except ValueError:
        return False


class TkinterGui():
    """Tkinter GUI class for the OMP Simulation."""

    def __init__(
                self,
                send_goal,
                send_goal_gripper,
                switch_auto_grasp,
                is_in_workspace,
                switch_debug_mode,
                switch_cam_calibration_mode,
                switch_mirroring_effect_mode
                ):
        self.root = tk.Tk()
        self.root.title("OMP Simulation")
        # First two values are size and
        # the next two values are position in screen
        self.root.geometry("400x600+1500+500")
        self.root.resizable(False, False)

        self.mode = tk.IntVar(value=1)
        self.entry, self.pose_entry, self.joint_angles = None, None, None
        self.mode_selected = ''

        # Get the parameters from the yaml file
        config_file = os.path.join(
            get_package_share_directory("manipulator_trajectory_vision_gui"),
            'config',
            'params.yaml'
        )
        self.config = load_yaml_file(config_file)['tkinter_gui']

        self.switch_auto_grasp = switch_auto_grasp
        self.send_goal = send_goal
        self.is_in_workspace = is_in_workspace
        self.switch_debug_mode = switch_debug_mode
        self.send_goal_gripper = send_goal_gripper
        self.switch_cam_calibration_mode = switch_cam_calibration_mode
        self.switch_mirroring_effect_mode = switch_mirroring_effect_mode
        self.list_of_values = []

    def create_action_buttons_tkinter(
        self,
        buttons_lists: list
    ):
        """
        The last parameter is the frame where the button is clicked.
        """
        for button in buttons_lists:
            tk.Button(
                button[2],
                text=button[0],
                bg="white",
                fg="black",
                command=button[1],
                width=30).pack()

    def change_frame_tkinter(self, previous_frame, next_frame):
        """Change the frame in the Tkinter GUI"""
        previous_frame.pack_forget()
        next_frame.pack()

    def generate_main_frame(self, main_frame, inverse_kinematics_frame):
        """Generate the main frame in the Tkinter GUI.

        Parameters
        ----------
        main_frame : tk.Frame
            The main frame in the Tkinter GUI.
        inverse_kinematics_frame : tk.Frame
            The inverse kinematics frame in the Tkinter GUI.

        """
        gripper_closed_position = self.config['gripper_closed_position']
        gripper_open_position = self.config['gripper_open_position']
        main_frame_buttons = [
            ["STANDING",
             lambda: [
                 self.send_goal(self.config['standing_pose'],
                                movement="slow")],
             main_frame],
            ["HOME",
             lambda: [
                 self.send_goal(self.config['home_pose'],
                                movement="slow")],
             main_frame],
            ["OPEN GRIPPER",
             lambda: [
                 self.send_goal_gripper(
                    position=[gripper_open_position]
                 )],
             main_frame],
            ["CLOSE GRIPPER",
             lambda: [
                 self.send_goal_gripper(
                    position=[gripper_closed_position]
                 )],
             main_frame],
            ["SLEEP POSE",
             lambda: [
                self.send_goal(
                        self.config['sleep_pose'],
                        movement="slow",
                        check_self_collision=0.0)
                        ],
             main_frame],
            ["FLOOR GRASP POSE",
             lambda: [
                self.send_goal(
                    self.config['floor_grasp_pose'],
                    movement="slow")
                    ],
             main_frame],
            ["CAM CALIBRATION POSE",
             lambda: [
                self.send_goal(
                    self.config['cam_calibration_pose'],
                    movement="slow")
                    ],
             main_frame],
            ["AUTO GRASP POSE",
             lambda: [
                self.send_goal(
                    self.config['auto_grasp_pose'],
                    movement="slow")
                    ],
             main_frame],
        ]
        self.create_action_buttons_tkinter(main_frame_buttons)

        # Additional button
        auto_grasp_button = tk.Button(
                main_frame,
                text="AUTO GRASP",
                bg='#d4d4d4',
                fg="black",
                command=lambda: [
                    self.switch_func(
                        self.switch_auto_grasp, auto_grasp_button,
                        'AUTO GRASP'
                        ),
                    self.change_buttons(main_frame, auto_grasp_button),
                    ],
                width=30)
        auto_grasp_button.pack()

        # Additional button
        randomize_button = tk.Button(
                main_frame,
                text="RANDOMIZE",
                bg='#d4d4d4',
                fg="black",
                command=lambda: [
                    self.switch_func(
                        self.switch_debug_mode,
                        randomize_button,
                        'RANDOMIZE'
                        ),
                    self.change_buttons(main_frame, randomize_button),
                    ],
                width=30)
        randomize_button.pack()

        # Additional button
        cam_calibrate_button = tk.Button(
                main_frame,
                text="CAM CALIBRATION",
                bg='#d4d4d4',
                fg="black",
                command=lambda: [
                    self.switch_func(
                        self.switch_cam_calibration_mode,
                        cam_calibrate_button,
                        'CAM CALIBRATION'),
                    self.change_buttons(main_frame, cam_calibrate_button),
                    ],
                width=30)
        cam_calibrate_button.pack()

        # Additional button
        mirroring_effect_button = tk.Button(
                main_frame,
                text="MIRRORING EFFECT",
                bg='#d4d4d4',
                fg="black",
                command=lambda: [
                    self.switch_func(
                        self.switch_mirroring_effect_mode,
                        mirroring_effect_button,
                        'MIRRORING EFFECT'),
                    self.change_buttons(main_frame, mirroring_effect_button),
                    ],
                width=30)
        mirroring_effect_button.pack()

        # Add a Radiobutton (switch button) to the frame
        switch_button = tk.Radiobutton(
            main_frame,
            text="INVERSE KINEMATICS",
            variable=self.mode,
            value=1,
            command=self.switch)
        switch_button.pack(pady=10)

        label = tk.Label(
            main_frame,
            text="Enter goal position and orientation of end-effector"
            )
        label.pack()

        # Getting custom pose
        self.pose_entry = tk.Entry(main_frame)
        self.pose_entry.insert(0, "0.0 0.0 0.42 1.0 0.0 0.0 0.0")
        self.pose_entry.pack()

        # Add a Radiobutton (switch button) to the frame
        switch_button = tk.Radiobutton(
            main_frame,
            text="DIRECT KINEMATICS",
            variable=self.mode,
            value=2,
            command=self.switch
            )
        switch_button.pack(pady=10)

        label = tk.Label(main_frame, text="Enter joint angles for the OMP")
        label.pack()

        # Getting custom pose
        self.joint_angles = tk.Entry(main_frame)
        self.joint_angles.insert(0, "0.0 -1.4 0.5 0.0 0.9 0.0")
        self.joint_angles.pack()

        self.entry = self.pose_entry

        # Additional button
        tk.Button(
                main_frame,
                text="ENVIAR",
                bg="white",
                fg="black",
                command=lambda: [self.send_custom_goal(self.entry)],
                width=30).pack()

    def switch_func(self, func, clicked_button, mode_name=''):
        if clicked_button.cget("text") == 'CANCEL':
            func('off')
        else:
            messagebox.showinfo(
                "Mensagem",
                f'Mode {mode_name} activated')
            func('on')

    def change_buttons(self, frame, clicked_button):
        if self.mode_selected == '':
            self.mode_selected = clicked_button.cget("text")
            clicked_button.config(text='CANCEL')
            for widget in frame.winfo_children():
                if isinstance(widget, tk.Button) and widget != clicked_button:
                    widget.config(state='disabled')
        else:
            clicked_button.config(text=self.mode_selected)
            self.mode_selected = ''
            for widget in frame.winfo_children():
                if isinstance(widget, tk.Button) and widget != clicked_button:
                    widget.config(state='normal')

    def switch(self,):
        if self.mode.get() == 1:
            self.entry = self.pose_entry
            messagebox.showinfo("Mensagem",
                                'Modo de cinemática inversa')
        else:
            self.entry = self.joint_angles
            messagebox.showinfo("Mensagem",
                                'Modo de cinemática direta')

    def send_custom_goal(self, entry):
        if self.mode.get() == 1:
            entry = entry.get().split(" ")
            if len(entry) == 7:
                entry = [float(x) for x in entry]
                messagebox.showinfo("Mensagem",
                                    f"Pose recebida: {entry}.")
                self.send_goal(
                    entry,
                    movement="slow",
                    inv_kin=True
                    )
            else:
                messagebox.showinfo(
                    "Mensagem",
                    "Envie posição (x, y, z) e quaternion (qw qx qy qz). "
                    "Exemplo: 0.0 0.0 0.42 1.0 0.0 0.0 0.0."
                    )
        else:
            entry = entry.get().split(" ")
            if len(entry) == 6:
                entry = [float(x) for x in entry]
                messagebox.showinfo("Mensagem",
                                    f"Valores recebidos: {entry}.")
                self.send_goal(
                    entry,
                    movement="slow",
                    inv_kin=False
                    )
            else:
                messagebox.showinfo(
                        "Mensagem",
                        "Envie seis valores de ângulos."
                        )

    def generate_inverse_kinematics_frame(
        self,
        main_frame,
        inverse_kinematics_frame
    ):
        """Generate the inverse kinematics frame in the Tkinter GUI.

        Parameters
        ----------
        main_frame : tk.Frame
            The main frame in the Tkinter GUI.
        inverse_kinematics_frame : tk.Frame
            The inverse kinematics frame in the Tkinter GUI.

        """
        labels = [
            "X or J1:",
            "Y or J2:",
            "Z or J3:",
            "Rx or J4:",
            "Ry or J5:",
            "Rz or J6:",
            "Solution index:"
        ]
        default_values = self.config['robot_home_joint_angles']
        entries = []
        tk.Label(inverse_kinematics_frame,
                 text="Values in deg and meters",
                 height=1).pack()
        for i, label in enumerate(labels):
            tk.Label(inverse_kinematics_frame,
                     text=label).pack()
            default_value = default_values[i]
            df_tk = tk.DoubleVar()
            df_tk.set(default_value)
            entry = tk.Entry(inverse_kinematics_frame,
                             textvariable=df_tk,
                             width=10)
            entries.append(entry)
            entry.pack()

        self.list_of_values = []
        inv_kin_option = tk.BooleanVar()
        checkbox = tk.Checkbutton(
            inverse_kinematics_frame,
            text="Inv Kin",
            onvalue=True,
            offvalue=False,
            variable=inv_kin_option,
            state="active")
        checkbox.pack()
        inv_kin_debug_option = tk.BooleanVar()
        checkbox2 = tk.Checkbutton(
            inverse_kinematics_frame,
            text="Debug IKin",
            onvalue=True,
            offvalue=False,
            variable=inv_kin_debug_option,
            state="active")
        checkbox2.pack()

        def get_values():
            inv_kin_option_var = inv_kin_option.get()
            inv_kin_debug_option_var = inv_kin_debug_option.get()
            self.list_of_values = []
            if not inv_kin_option_var:
                for entry in entries:
                    input_value = entry.get()
                    if not is_valid_number(input_value):
                        messagebox.showinfo("Mensagem",
                                            "Only numbers accepted.")
                        return
                    value = float(input_value)
                    value = np.radians(value)
                    self.list_of_values.append(value)
            else:
                for idx, entry in enumerate(entries):
                    value = float(entry.get())
                    if idx < 3 and value > 0.85:
                        messagebox.showinfo("Mensagem",
                                            "The maxium OMP reach is 0.85m")
                        return
                    self.list_of_values.append(value)

            print(self.list_of_values)
            self.send_goal(self.list_of_values[:-1],
                           inv_kin=inv_kin_option_var,
                           movement='slow',
                           solution_index=int(self.list_of_values[-1]),
                           debug_inv_kin=inv_kin_debug_option_var)

        tk.Label(inverse_kinematics_frame, text="", height=1).pack()

        tk.Button(inverse_kinematics_frame,
                  text="Send goal",
                  command=get_values
                  ).pack(side="left")
        tk.Button(inverse_kinematics_frame,
                  text="Main Menu",
                  command=lambda: self.change_frame_tkinter(
                    inverse_kinematics_frame,
                    main_frame
                    )
                  ).pack(side="left")

    def build_frames(self):
        """Build the frames in the Tkinter GUI"""
        # Frame do botao atual
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP)
        inverse_kinematics_frame = tk.Frame()

        self.generate_main_frame(main_frame,
                                 inverse_kinematics_frame)
        self.generate_inverse_kinematics_frame(main_frame,
                                               inverse_kinematics_frame)

        return self.root
