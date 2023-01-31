import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

trajectory = pd.read_csv("C:\\Users\\posorio\\Downloads\\5_dbpendulum_fsm\\5_dbpendulum_fsm\\pendulum_twist.csv")
trajectory = trajectory.to_numpy()
aux_zeros = np.zeros((17,6))
trajectory = np.concatenate((trajectory,aux_zeros),axis=0)
timeline = np.linspace(0,len(trajectory),len(trajectory))
print(np.shape(trajectory))

fig5, axs5 = plt.subplots(2, 3)
for n,ax in enumerate(axs5.flatten()):
    ax.plot(timeline,trajectory[:,n],linewidth=2.0)
    if n < 3:
        ax.set(ylabel='Linear velocity (m/s)', xlabel='Time (s)')
    else:
        ax.set(ylabel='Angular velocity (rad/s)', xlabel='Time (s)')
        
fig5.suptitle('Kinematic Twist Double Pendulum', fontsize=20)
plt.show()

# seq_human = seq_human.clone().cpu().squeeze(0).numpy()
# axs5[0, 0].plot(sample, self.traj_plan_poses_gan[0,:],linewidth=2.0)
# axs5[0, 1].plot(sample, self.traj_plan_poses_real[0,:],linewidth=2.0)
# axs5[0, 2].plot(sample, seq_human[0,:],linewidth=2.0)

# axs5[1, 0].plot(sample, self.traj_plan_poses_gan[1,:],linewidth=2.0)
# axs5[1, 1].plot(sample, self.traj_plan_poses_real[1,:],linewidth=2.0)
# axs5[1, 2].plot(sample, seq_human[1,:],linewidth=2.0)

# axs5[2, 0].plot(sample, self.traj_plan_poses_gan[2,:],linewidth=2.0)
# axs5[2, 1].plot(sample, self.traj_plan_poses_real[2,:],linewidth=2.0)
# axs5[2, 2].plot(sample, seq_human[2,:],linewidth=2.0)

# axs5[3, 0].plot(sample, self.traj_plan_poses_gan[3,:],linewidth=2.0)
# axs5[3, 1].plot(sample, self.traj_plan_poses_real[3,:],linewidth=2.0)
# axs5[3, 2].plot(sample, seq_human[3,:],linewidth=2.0)

# axs5[4, 0].plot(sample, self.traj_plan_poses_gan[4,:],linewidth=2.0)
# axs5[4, 1].plot(sample, self.traj_plan_poses_real[4,:],linewidth=2.0)
# axs5[4, 2].plot(sample, seq_human[4,:],linewidth=2.0)

# axs5[5, 0].plot(sample, self.traj_plan_poses_gan[5,:],linewidth=2.0)
# axs5[5, 1].plot(sample, self.traj_plan_poses_real[5,:],linewidth=2.0)
# axs5[5, 2].plot(sample, seq_human[5,:],linewidth=2.0)
