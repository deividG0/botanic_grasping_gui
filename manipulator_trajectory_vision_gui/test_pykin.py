import numpy as np
from pykin_kinematics import IKCalculatorOMP

joint_limits = {'joint1': (-np.pi/2, np.pi/2),
                'joint2': (-np.pi/2, np.pi/2),
                'joint3': (-np.pi/2, 3*np.pi/4),
                'joint4': (-np.pi/2, np.pi/2),
                'joint5': (-np.pi/2, np.pi/2),
                'joint6': (-np.pi/2, np.pi/2)
                }

c = IKCalculatorOMP()
print(c.calculate_ik(
            pose=[0.5]*6,
            current_thetas=[0.5]*6,
            max_iter=100,
            joint_limits=list(joint_limits.values()),
            method="LM_modified"
            ))
