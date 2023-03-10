import os
import numpy as np
import matplotlib.pyplot as plt

def get_files_from_dirs(org_dirpath,emotion="",actor="",lambda_val=1):
    lambda_val = str(lambda_val)
    network_dirpath = os.path.join(org_dirpath,"Network output")
    for root, dirs, files in os.walk(network_dirpath):
        for file in files:
            string_arr = file.split("_")
            if set(["twist", actor, emotion, lambda_val]).issubset(set(string_arr)):
                input_arr = np.load(os.path.join(root,file))
    return input_arr

def plot_twist(input_motion_arr,axs, input_label=""):
    shape = np.shape(input_motion_arr)
    samples = np.linspace(0, shape[0], shape[0])
    for n,ax in enumerate(axs.flatten()):
        ax.plot(samples,input_motion_arr[:,n], label=input_label, linewidth=3.0)
        ax.legend()

def main():
    
    data_dirpath = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Translation_Process_Simulation\\Data"
    robot_motion_twist = np.load(os.path.join(data_dirpath,"Original robot motion\\robot_motion_twist_EMLA_COE_1_seed_30.npy"))
    
    map_emotion = {"COE":"anger","JOE":"happy","TRE":"sad"}
    map_actor = {"SALE":0,"PAIB":1,"NABA":2,"EMLA":3}
    emotion_flag = False
    if not emotion_flag:
        list_oi = ["EMLA","NABA","PAIB","SALE"]
        emotion = "JOE"
    else:
        actor = "SALE"
        list_oi = ["COE","JOE","TRE"]

    fig, axs = plt.subplots(2,3)
    plot_twist(robot_motion_twist,axs, input_label="original motion")
         
    participant_arr = []
    for n,tag in enumerate(list_oi):
        if not emotion_flag:
            participant_arr.append(get_files_from_dirs(data_dirpath,emotion=emotion,actor=tag,lambda_val=1))
            plot_twist(participant_arr[n],axs, input_label="Actor :" + str(map_actor[tag]))
        else:
            participant_arr.append(get_files_from_dirs(data_dirpath,emotion=tag,actor=actor,lambda_val=1))
            plot_twist(participant_arr[n],axs, input_label="Emotion :" + map_emotion[tag])

    for n,ax in enumerate(axs.flat):
        if n < 3:
            ax.set(ylabel='Velocity (m/s)', xlabel='Sample')
        else:
            ax.set(ylabel='Angular velocity (rad/s)', xlabel='Sample')
    fig.suptitle('End Effector Twist - Different Human Inputs')

    for ax, col in zip(axs[0], ["X", "Y", "Z"]):
                ax.set_title(col, fontsize=18)
    plt.show()

if __name__=="__main__":
    main()