from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from mpl_toolkits.mplot3d import Axes3D
import ikpy
import numpy as np
from ikpy import plot_utils
import matplotlib.pyplot as plt
from ikpy import plot_utils
import matplotlib.pyplot

robot_arm_chain = Chain(name='left_arm', links=[
    OriginLink(),
    URDFLink(
      name="shoulder",
      translation_vector=[-10, 0, 5],
      orientation=[0, 1.57, 0],
      rotation=[0, 1, 0],
    ),
    URDFLink(
      name="elbow",
      translation_vector=[25, 0, 0],
      orientation=[0, 0, 0],
      rotation=[0, 1, 0],
    ),
    URDFLink(
      name="wrist",
      translation_vector=[22, 0, 0],
      orientation=[0, 0, 0],
      rotation=[0, 1, 0],
    )
])

pos = robot_arm_chain.forward_kinematics([0] * 4)
print(pos)


target_vector = [ 20, 20, 20]
target_frame = np.eye(4)
target_frame[:3, 3] = target_vector

print("The angles of each joints are : ", robot_arm_chain.inverse_kinematics(target_frame))

ax = plot_utils.init_3d_figure()
robot_arm_chain.plot(robot_arm_chain.inverse_kinematics(target_frame), ax, target=target_vector)
plt.xlim(-0.1, 0.1)
plt.ylim(-0.1, 0.1)
"""ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')

robot_arm_chain.plot(robot_arm_chain.inverse_kinematics([
    [2, 0, 0, 2],
    [0, 0, 0, 1],
    [0, 1, 1, 2],
    [1, 0, 2, 1]
    ]), ax)"""
matplotlib.pyplot.show()
