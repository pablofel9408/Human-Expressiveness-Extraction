import os 
import sys
import json 
import pathlib

import torch 
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utilities
from GAN_Translation.General_Functions.START_gan_proc import ModelingGAN
from Robot_Feature_Extraction.Preprocessing.preprocess_robot_datasets import PreprocessRobotData
from Robot_Feature_Extraction.Preprocessing.process_human_data import PreprocessHumanData
from Translation_Process_Simulation.Simulation_Scripts import START_simulation
from Data_Exploration import DataExploration

def main():
    seed = 10
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    keyword = "no sim"
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
        single_sample = sys.argv[2]
        laban_analysis = sys.argv[3]

    dir_path = pathlib.Path(__file__).parent / "GAN_Translation\\Config Files"
    filepath_dataset_hum_cst = os.path.join(dir_path, "human_dataset_config.json")
    filepath_vae_hum_cst = os.path.join(dir_path, "VAE_human_model_system_config.json")
    filepath_model_cst = os.path.join(dir_path, "model_system_config.json")

    dataset_constants_human = utilities.load_raw_constants(filepath_dataset_hum_cst)
    vae_constants_human = utilities.load_raw_constants(filepath_vae_hum_cst)
    model_constants = utilities.load_raw_constants(filepath_model_cst)
    model_constants["model_config"]["generator"]["human_vae"] = vae_constants_human


    prepros_obj_hum = PreprocessHumanData(dataset_constants_human)
    prepros_obj_hum.load_recursive_pt(dataset_constants_human["dataset_paths"])

    dataset_hum = prepros_obj_hum.return_data_()
    dataset_tags_hum = prepros_obj_hum.return_data_tags()

    if dataset_constants_human["expressive_data"]:
        dataset_hum = (dataset_hum,dataset_tags_hum)

    data_exploration = DataExploration(model_constants, save=True)
    data_exploration.load_data(dataset_hum[0],dataset_tags_hum)
    dataset_nee = data_exploration.generate_latent_neutral()

    

if __name__=="__main__":
    main()