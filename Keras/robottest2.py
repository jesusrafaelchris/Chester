import ikpy
import numpy as np
from ikpy import plot_utils
import matplotlib.pyplot as plt
from ikpy import plot_utils
import matplotlib.pyplot

my_chain = ikpy.chain.Chain.from_urdf_file("/Users/christiangrinling/Desktop/Keras/poppy_ergo.URDF")

target_vector = [ 0.05, 0.02, 0.4]
target_frame = np.eye(4)
target_frame[:3, 3] = target_vector


print("The angles of each joints are : ", my_chain.inverse_kinematics(target_frame))


real_frame = my_chain.forward_kinematics(my_chain.inverse_kinematics(target_frame))
print("Computed position vector : %s, original position vector : %s" % (real_frame[:3, 3], target_frame[:3, 3]))



ax = plot_utils.init_3d_figure()
my_chain.plot(my_chain.inverse_kinematics(target_frame), ax, target=target_vector)
plt.xlim(-0.1, 0.1)
plt.ylim(-0.1, 0.1)
matplotlib.pyplot.show()
