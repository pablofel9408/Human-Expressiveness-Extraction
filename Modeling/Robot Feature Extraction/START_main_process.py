from operator import mod
from unicodedata import name
import os
import sys

import pathlib
import numpy as np 
import torch 
import random

import utilities
from Preprocessing.preprocess_robot_datasets import PreprocessRobotData
from Preprocessing.process_human_data import PreprocessHumanData
from Modeling.START_modeling_process import ModelingProc

def main():

    seed = 0
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if len(sys.argv) > 1:
        task_name = sys.argv[1]
    else:
        print("Need selection from human or robot")
        sys.exit(1)

    dir_path = pathlib.Path(__file__).parent / "Config Files"

    task_name = task_name.lower()
    filepath_dataset_cst = os.path.join(dir_path, task_name + "_dataset_config.json")
    filepath_model_cst = os.path.join(dir_path, task_name + "_model_system_config.json")
    
    dataset_constants = utilities.load_raw_constants(filepath_dataset_cst)

    if task_name == "robot":
        prepros_obj = PreprocessRobotData(dataset_constants)
        prepros_obj.start_preprocessing(tag='train')
    else:
        prepros_obj = PreprocessHumanData(dataset_constants)
        prepros_obj.load_recursive_pt(dataset_constants["dataset_paths"])
    dataset = prepros_obj.return_data_()

    model_constants = utilities.load_raw_constants(filepath_model_cst)
    model_constants["task_name"] = task_name
    modeling_proc_obj = ModelingProc(model_constants)
    modeling_proc_obj.set_input_data(dataset)
    
    if not model_constants["pretrained_model"]:
        history, best_model_wts, model = modeling_proc_obj.training_loop()
    else:
        epoch, trainLoss, valLoss, model = modeling_proc_obj.load_model(evalFlag=True)

    for p in model.parameters():
            p.requires_grad = False

    modeling_proc_obj.visualize_output(model, train=False)


if __name__=="__main__":
    main()