import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualize_network_outputs(network_output_random, trajectory):
    diff = abs(np.shape(network_output_random)[0] - np.shape(trajectory)[0])
    trajectory = np.concatenate((trajectory,np.zeros((diff,np.shape(trajectory)[1]))),axis=0)
    timeline = np.linspace(0,np.shape(trajectory)[0],np.shape(trajectory)[0])
    fig5, axs5 = plt.subplots(2, 3)
    for n,ax in enumerate(axs5.flatten()):
        ax.plot(timeline,network_output_random[:,n],linewidth=2.0, color='r', label="Network Output")
        ax.plot(timeline,trajectory[:,n],linewidth=1.0, color="b", label="Original Motion")
        if n < 3:
            ax.set(ylabel='Position (m)', xlabel='Sample')
        else:
            ax.set(ylabel='Angle (degree)', xlabel='Sample')
        ax.legend()     
    fig5.suptitle("Pose Meca500 End Effector - Network's Output", fontsize=20)

    for ax, col in zip(axs5[0], ["X", "Y", "Z"]):
        ax.set_title(col, fontsize=18)

    plt.show()

def get_files_from_dirs(org_dirpath,emotion="",actor="",lambda_val=1):
    lambda_val = str(lambda_val)
    network_dirpath = os.path.join(org_dirpath,"Network_Output\\Before_scaling")
    output_dirpath = os.path.join(org_dirpath,"Network_Output\\After_scaling")
    output_files = []
    for root, dirs, files in os.walk(network_dirpath):
        for file in files:
            string_arr = re.split(r"[_.]", file)
            if set(["position", actor, emotion, lambda_val]).issubset(set(string_arr)):
                input_arr = np.load(os.path.join(root,file))
                processed_arr = process_trajectory(input_arr)
                # np.save(os.path.join(output_dirpath,actor,file),processed_arr)
                output_files.append(file)
    return input_arr, output_files

def process_trajectory(input_arr):
    input_arr[:,:3] = input_arr[:,:3] * 1000 
    input_arr[:,0] = 200
    input_arr[:,2] = input_arr[:,2] + 250
    input_arr[:,2] = np.where(input_arr[:,2] > 320, 320, input_arr[:,2])
    input_arr[:,3:] = np.rad2deg(input_arr[:,3:])
    input_arr[:,4] = 90
    return input_arr


def main():
    data_dirpath = "C:\\Users\\posorio\\OneDrive - 国立研究開発法人産業技術総合研究所\\Documents\\Expressive movement\\Modeling\\Mecademic_Twist_Sim\\Data"
    traj_arr = []
    traj_arr_filenames = []
    for actor in ["EMLA","NABA","PAIB","SALE"]:
        for emotion in ["COE","JOE","TRE"]:
            for lambda_val in [1,100]:
                if actor=="PAIB" and emotion=="JOE" and lambda_val==100:
                    continue
                trajectory,filenames = get_files_from_dirs(data_dirpath,emotion=emotion,actor=actor,lambda_val=lambda_val)
                traj_arr.append(trajectory)
                traj_arr_filenames.append(filenames)

    dirpath = "C:\\Users\\posorio\\OneDrive - 国立研究開発法人産業技術総合研究所\\Documents\\Expressive movement\\Modeling\\Mecademic_Twist_Sim\\Data"
    trajectory_org = pd.read_csv(os.path.join(dirpath,"Robot\\logger_traj_file_output_df.csv"))
    print(trajectory_org.columns)
    trajectory_org = trajectory_org[['CartPos_X', 'CartPos_Y', 'CartPos_Z',
                            'CartPos_Alpha', 'CartPos_Beta', 'CartPos_Gamma']]
    trajectory_org['CartPos_Alpha'] = 0
    trajectory_org['CartPos_Beta'] = 90
    trajectory_org['CartPos_Gamma'] = 0
    trajectory_org = trajectory_org.to_numpy()

    for n,trajectory in enumerate(traj_arr):
        for elem in traj_arr_filenames[n]:
            print(elem)
            visualize_network_outputs(trajectory,trajectory_org)


if __name__=="__main__":
    main()