LOAD_TRAJECTORY = True
LOAD_FROM_ARCHIVE = True
SAVE_TRAJECTORY = False

FSM_HOLD = 0
FSM_SWING1 = 1
FSM_SWING2 = 2
FSM_STOP = 3

t_hold = 0.5
t_swing1 = 1.0
t_swing2 = 1.0

flag_cst = 2
# 2: same network output
# 3: random network output
# 4: same human input
# 5: random human input

pendulum_trajectory_filepath = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Pendulum_Twist_Sim\\Data\\Pendulum_Twist_Swing\\pendulum_twist.csv"

trajectory_human_same_filepath = "C:\\Users\\posorio\\Documents\\Expressive movement\Modeling\\Pendulum_Twist_Sim\\Data\\Human_Input\\human_same_inut.npy"
trajectory_human_same_filepath_vel = "C:\\Users\\posorio\\Documents\\Expressive movement\Modeling\\Pendulum_Twist_Sim\\Data\\Human_Input\\human_same_input_vel.npy"
trajectory_network_same_filepath = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Pendulum_Twist_Sim\\Data\\Pendulum_Twist_Network_Output\\network_output_double_pendulum_same_human_input.npy"
trajectory_human_random_filepath = "C:\\Users\\posorio\\Documents\\Expressive movement\Modeling\\Pendulum_Twist_Sim\\Data\\Human_Input\\human_random_input.npy"
trajectory_human_random_filepath_vel = "C:\\Users\\posorio\\Documents\\Expressive movement\Modeling\\Pendulum_Twist_Sim\\Data\\Human_Input\\human_random_input_vel.npy"
trajectory_network_random_filepath = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Pendulum_Twist_Sim\\Data\\Pendulum_Twist_Network_Output\\network_output_double_pendulum_random_human_input.npy"

trajectories_human_save_filepath = "C:\\Users\\posorio\\Documents\\Expressive movement\Modeling\\Pendulum_Twist_Sim\\Data\\Human_Input\\"

path_to_compare = ["C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Pendulum_Twist_Sim\\Data\\Pendulum_Twist_Network_Output\\Archive\\Output_seed_10\\network_output_double_pendulum_same_human_input.npy",
                    "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Pendulum_Twist_Sim\\Data\\Pendulum_Twist_Network_Output\\Archive\\Output_seed_20\\network_output_double_pendulum_same_human_input.npy"]