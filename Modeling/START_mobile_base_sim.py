import os
import sys
import pathlib
from operator import itemgetter

import random
import numpy as np
import pandas as pd
import torch

from Mobile_Base_Twist_Sim import ProcessTrajectory

from GAN_Translation.General_Functions.START_gan_proc import ModelingGAN
from Robot_Feature_Extraction.Preprocessing.preprocess_robot_datasets import PreprocessRobotData
from Robot_Feature_Extraction.Preprocessing.process_human_data import PreprocessHumanData

import utilities

def load_files_preproc_dataset(lambda_val=None):
    seed = 10
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    dir_path = pathlib.Path(__file__).parent / "GAN_Translation\\Config Files"
    dir_path_simulation = pathlib.Path(__file__).parent / "Translation_Process_Simulation\\Config Files"
    filepath_dataset_hum_cst = os.path.join(dir_path, "human_dataset_config.json")
    filepath_dataset_rob_cst = os.path.join(dir_path, "robot_dataset_config.json")
    filepath_vae_hum_cst = os.path.join(dir_path, "VAE_human_model_system_config.json")
    filepath_vae_rob_cst = os.path.join(dir_path, "VAE_robot_model_system_config.json")
    filepath_model_cst = os.path.join(dir_path, "model_system_config.json")
    filepath_sim_cst = os.path.join(dir_path_simulation, "simulation_config.json")

    dataset_constants_robot = utilities.load_raw_constants(filepath_dataset_rob_cst)
    dataset_constants_human = utilities.load_raw_constants(filepath_dataset_hum_cst)
    vae_constants_human = utilities.load_raw_constants(filepath_vae_hum_cst)
    vae_constants_robot = utilities.load_raw_constants(filepath_vae_rob_cst)
    model_constants = utilities.load_raw_constants(filepath_model_cst)
    model_constants["model_config"]["generator"]["human_vae"] = vae_constants_human
    model_constants["model_config"]["generator"]["robot_vae"] = vae_constants_robot
    model_constants["task_name"] = "GAN"

    prepros_obj_rob = PreprocessRobotData(dataset_constants_robot)
    prepros_obj_rob.start_preprocessing(tag='train')
    prepros_obj_hum = PreprocessHumanData(dataset_constants_human)
    prepros_obj_hum.load_recursive_pt(dataset_constants_human["dataset_paths"])

    dataset_rob = prepros_obj_rob.return_data_()
    dataset_hum = prepros_obj_hum.return_data_()
    dataset_tags_hum = prepros_obj_hum.return_data_tags()

    dataset = (dataset_rob, dataset_hum)
    if dataset_constants_human["expressive_data"]:
        dataset = (dataset_rob, dataset_hum,dataset_tags_hum)

        if model_constants["neutral_style"]:
            print("here")
            aux_general_dataset = {}
            for key, value in dataset_tags_hum.items():
                if key in ["emotion", "actor"]:
                    continue
                aux_pd = pd.DataFrame(value)
                # aux_pd.to_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Translation_Dataset_Analysis\\Data\\emotion_dataset_"+ key+ ".csv")
                aux_pd_nee = aux_pd[aux_pd["emo"]=="NEE"]
                aux_pd = aux_pd[aux_pd["emo"]!="NEE"]
                # print(aux_pd_nee.index.values)
                # print(len(aux_pd_nee.index.values))
                dataset_tags_hum[key] = aux_pd.reset_index(drop=True).to_dict()
                aux_dataset = dataset[1][key][0][aux_pd.index.values] 
                aux_dataset_laban_qual = dataset[1][key][1][aux_pd.index.values] 
                aux_general_dataset[key] = (aux_dataset,aux_dataset_laban_qual)
            
                print(len(dataset_tags_hum[key]["emo"]))
                print(len(dataset_tags_hum[key]["act"]))
                print(np.shape(aux_dataset))
                print(np.shape(aux_dataset_laban_qual))
            dataset = (dataset_rob,aux_general_dataset,dataset_tags_hum)
    
    if lambda_val is not None:
        model_constants["model_config"]["generator"]["twist_generation"]["lambda_gain"] = lambda_val
    
    modeling_proc_obj = ModelingGAN(model_constants)
    modeling_proc_obj.set_input_data(dataset, expressive=dataset_constants_human["expressive_data"])
    # modeling_proc_obj.set_lambda_gain(lambda_val)
    epoch, trainLoss, valLoss, model = modeling_proc_obj.load_model(evalFlag=True)
    model.eval()

    if model_constants["neutral_style"]:
        neutral_data = modeling_proc_obj.return_neutral_data()
        dataset = (dataset_rob,aux_general_dataset,dataset_tags_hum, neutral_data)

    return dataset, model, dataset_constants_human,dataset_constants_robot, model_constants

def save_dataset_tags(dataset_tags, movement_indices, key, tag="random", dirpath=""):

    trajectory_index = {0:"circle",1:"s_shape"}
    emotion_tags = [dataset_tags["emo"][x] for x in movement_indices]
    actor_tags = [dataset_tags["act"][x] for x in movement_indices]

    with open(os.path.join(dirpath,'emotion_tags_'+trajectory_index[key]+"_"+tag+'.txt'), 'w') as f:
        for line in emotion_tags:
            f.write(f"{line}\n")

    with open(os.path.join(dirpath,'actor_tags_'+trajectory_index[key]+"_"+tag+'.txt'), 'w') as f:
        for line in actor_tags:
            f.write(f"{line}\n")
    
def main():
    if len(sys.argv) > 1:
        elem = int(sys.argv[1])

    # ["EMLA","NABA","PAIB","SALE"]
    # ["COE","JOE","NEE","TRE"]
    trajectories_paths = ["C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Mobile_Base_Twist_Sim\\Data\\Robot\\trajectory_file_circle.csv",
                            "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Mobile_Base_Twist_Sim\\Data\\Robot\\trajectory_file_s_shape_short.csv"]
    human_data_save_path  = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Mobile_Base_Twist_Sim\\Data\\Human_Data"
    for actor in ["EMLA","NABA","PAIB","SALE"]:
        for emotion in ["COE","JOE","TRE"]:
            for lambda_val in [1,100]:
                if actor=="PAIB" and emotion=="JOE" and lambda_val==100:
                    continue
                dataset, model, dataset_constants_human, dataset_constants_robot, model_constants = load_files_preproc_dataset(lambda_val=lambda_val)
                mobile_data_obj = ProcessTrajectory([dataset_constants_human, dataset_constants_robot, model_constants],
                                                                    model,dataset_constants_robot["scalers_path"],
                                                                    emotion_tag=emotion, participant_tag=actor,lambda_val=lambda_val)
                mobile_data_obj.load_data(dataset)

                filter_values = [(0.2,60),(0.8,60)]
                random_human_indices, same_human_indices = mobile_data_obj.start_generation(trajectories_paths[elem],elem,save=True, 
                                                                                            cutoff_freq=filter_values[elem][0], fs=filter_values[elem][1])
                save_dataset_tags(dataset[2]["train"], same_human_indices, elem, tag="same", dirpath=human_data_save_path)
                save_dataset_tags(dataset[2]["train"], random_human_indices, elem, tag="random", dirpath=human_data_save_path)
    

if __name__=="__main__":
    main()