from operator import mod
from unicodedata import name
import os
import sys

import pathlib
import numpy as np 

import utilities
from Preprocessing.preprocess_robot_datasets import PreprocessRobotData
from Modeling.START_modeling_process import ModelingProc

def main():

    dir_path = pathlib.Path(__file__).parent / "Config Files"

    filepath_dataset_cst = os.path.join(dir_path,"dataset_config.json")
    filepath_model_cst = os.path.join(dir_path,"model_system_config.json")
    
    dataset_constants = utilities.load_raw_constants(filepath_dataset_cst)
    prepros_obj = PreprocessRobotData(dataset_constants)
    prepros_obj.start_preprocessing(tag='train')
    dataset = prepros_obj.return_data_()

    model_constants = utilities.load_raw_constants(filepath_model_cst)
    modeling_proc_obj = ModelingProc(model_constants)
    modeling_proc_obj.set_input_data(dataset)
    history, best_model_wts, model = modeling_proc_obj.training_loop()
    modeling_proc_obj.visualize_output(model)


if __name__=="__main__":
    main()