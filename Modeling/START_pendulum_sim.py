import os
import sys
import pathlib

import random
import numpy as np
import pandas as pd
import torch

from Pendulum_Twist_Sim.doublependulum_fsm_class import FSM_Sim
from Pendulum_Twist_Sim.data_processing import DataProcessing
from Pendulum_Twist_Sim.pendulum_twsit_sim_data_generation import PendulumDataGeneration
import Pendulum_Twist_Sim.constanst as cst

from GAN_Translation.General_Functions.START_gan_proc import ModelingGAN
from Robot_Feature_Extraction.Preprocessing.preprocess_robot_datasets import PreprocessRobotData
from Robot_Feature_Extraction.Preprocessing.process_human_data import PreprocessHumanData

import utilities

def load_files_preproc_dataset():
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
    
    
    modeling_proc_obj = ModelingGAN(model_constants)
    modeling_proc_obj.set_input_data(dataset, expressive=dataset_constants_human["expressive_data"])
    epoch, trainLoss, valLoss, model = modeling_proc_obj.load_model(evalFlag=True)
    model.eval()

    if model_constants["neutral_style"]:
        neutral_data = modeling_proc_obj.return_neutral_data()
        dataset = (dataset_rob,aux_general_dataset,dataset_tags_hum, neutral_data)

    return dataset, model, dataset_constants_human,dataset_constants_robot, model_constants
    

def main():
    sim = "no_sim"
    laban_check = "no_laban"
    generate_files = "no_generate"
    if len(sys.argv) > 1:
        sim = sys.argv[1]
        laban_check = sys.argv[2]
        generate_files = sys.argv[3]
    
    sim = False if sim=="no_sim" else True
    laban_check = False if laban_check=="no_laban" else True
    generate_files = False if generate_files=="no_generate" else True

    if not sim:
        if generate_files:
            dataset, model, dataset_constants_human, dataset_constants_robot, model_constants = load_files_preproc_dataset()
            pendulum_data_obj = PendulumDataGeneration([dataset_constants_human, dataset_constants_robot, model_constants],
                                                        model)
            pendulum_data_obj.load_data(dataset)
            pendulum_data_obj.start_generation(save=True)
        else:
            data_proc_obj = DataProcessing()
            data_proc_obj.load_files_from_paths([cst.trajectory_human_same_filepath, cst.trajectory_network_same_filepath],
                                                integrate=True, save_tag="human_same_input_vel")
            if laban_check:
                data_proc_obj.start_processing()
            
            data_proc_obj.load_files_from_paths([cst.trajectory_human_random_filepath, cst.trajectory_network_random_filepath],
                                                integrate=True, save_tag="human_random_input_vel")
            if laban_check:
                data_proc_obj.start_processing()
    else:
        fsm_obj = FSM_Sim()
        fsm_obj.start_simulation()

if __name__=="__main__":
    main()