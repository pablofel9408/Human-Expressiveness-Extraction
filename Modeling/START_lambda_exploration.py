import os
import sys
import pathlib

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm

from Pendulum_Twist_Sim.doublependulum_fsm_class import FSM_Sim
from Pendulum_Twist_Sim.data_processing import DataProcessing
from Pendulum_Twist_Sim.pendulum_twsit_sim_data_generation import PendulumDataGeneration
import Pendulum_Twist_Sim.constanst as cst

from GAN_Translation.General_Functions.START_gan_proc import ModelingGAN
from Robot_Feature_Extraction.Preprocessing.preprocess_robot_datasets import PreprocessRobotData
from Robot_Feature_Extraction.Preprocessing.process_human_data import PreprocessHumanData
from Translation_Process_Simulation.Simulation_Scripts import START_simulation

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
    simulation_constants = utilities.load_raw_constants(filepath_sim_cst)
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

    if model_constants["neutral_style"]:
        neutral_data = modeling_proc_obj.return_neutral_data()
        dataset = (dataset_rob,aux_general_dataset,dataset_tags_hum, neutral_data)

    return dataset,dataset_constants_human,dataset_constants_robot, model_constants, simulation_constants
    
def main():

    max_gain = 500 
    steps = 1

    dataset,dataset_constants_human,dataset_constants_robot,\
        model_constants, simulation_constants = load_files_preproc_dataset()

    modeling_proc_obj = ModelingGAN(model_constants)
    modeling_proc_obj.set_input_data(dataset, expressive=dataset_constants_human["expressive_data"])
    simulation_obj = START_simulation.Simulation_Methods(simulation_constants, None, dataset_constants_robot["scalers_path"])
    neutral_data = modeling_proc_obj.return_neutral_data() if model_constants["neutral_style"] else None 
    for dataset_tag in ["train", "val", "test"]:
        df_laban_human = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\laban_qualities_"+ dataset_tag + "_human.csv")
        df_laban_robot = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\laban_qualities_"+ dataset_tag + "_robot.csv")

        print("----Evaluating Dataset " + dataset_tag)
        simulation_obj.set_input_data(dataset, neutral_data=neutral_data,
                                        tag=dataset_tag, neutral=model_constants["neutral_style"])
        similarity_arr = []
        mse_arr = []
        qualities_names = list(df_laban_human.columns)[1:-1]
        kl_div_arr = {"robot":{i:[] for i in qualities_names},"human":{i:[] for i in qualities_names}}
        shannon_jenssen_dist_arr = {"robot":{i:[] for i in qualities_names},"human":{i:[] for i in qualities_names}}
        prev_best_mean = [0,0,1000000,0]
        for lambda_val in tqdm.tqdm(range(0,max_gain,steps)):
            modeling_proc_obj.set_lambda_gain(lambda_val)
            epoch, trainLoss, valLoss, model = modeling_proc_obj.load_model(evalFlag=True)
            model.eval()
            simulation_obj.set_model(model)
            cosine_similarity_mean, mse_twist, expressive_qualities = simulation_obj.generate_output_analysis_single_ssample(tag=dataset_tag, 
                                                                                            neutral=model_constants["neutral_style"], 
                                                                                            not_save=True, not_expressive=False)
            similarity_arr.append(cosine_similarity_mean)
            mse_arr.append(mse_twist)

            for n,qualitie in enumerate(expressive_qualities.columns):
                if qualitie in qualities_names:
                    xs, p1, p2 = utilities._get_point_estimates(df_laban_human[qualitie].to_numpy(), expressive_qualities[qualitie].to_numpy(), None)
                    kl_human = utilities._kl_divergence(xs, p1, p2)
                    kl_div_arr["human"][qualitie].append(kl_human)
                    # print(f"KL Divergence Laban Qualities {qualitie} Human - Network Output: {kl_human}")
                    xs, p1, p2 = utilities._get_point_estimates(df_laban_robot[qualitie].to_numpy(), expressive_qualities[qualitie].to_numpy(), None)
                    kl_robot = utilities._kl_divergence(xs, p1, p2)
                    kl_div_arr["robot"][qualitie].append(kl_robot)
                    # print(f"KL Divergence Laban Qualities {qualitie} Robot - Network Output: {kl_robot}")
                    js_human = utilities.js_metric(df_laban_human[qualitie].to_numpy(), expressive_qualities[qualitie].to_numpy(), None)
                    shannon_jenssen_dist_arr["human"][qualitie].append(js_human)
                    # print(f"Jensen-Shannon Laban Qualities {qualitie} Human - Network Output: {js_human}")
                    js_robot = utilities.js_metric(df_laban_robot[qualitie].to_numpy(), expressive_qualities[qualitie].to_numpy(), None)
                    shannon_jenssen_dist_arr["robot"][qualitie].append(js_robot)
                    # print(f"Jensen-Shannon Laban Qualities {qualitie} Robot - Network Output: {js_robot}")               
    

            if (np.mean(cosine_similarity_mean) > prev_best_mean[0]) and (lambda_val > 0):
                prev_best_mean[0] = np.mean(cosine_similarity_mean)
                prev_best_mean[1] = lambda_val

            if (mse_twist < prev_best_mean[2]) and (lambda_val> 0):
                prev_best_mean[2] = mse_twist
                prev_best_mean[3] = lambda_val
 
        similarity_arr = np.asarray(similarity_arr)
        concat_outputs = []
        for qualitie in qualities_names:
            concat_outputs.append(kl_div_arr["robot"][qualitie])
            concat_outputs.append(shannon_jenssen_dist_arr["robot"][qualitie])
            concat_outputs.append(kl_div_arr["human"][qualitie])
            concat_outputs.append(shannon_jenssen_dist_arr["human"][qualitie])
        
        for i in range(np.shape(similarity_arr)[1]):
            concat_outputs.append([similarity_arr[0,i],similarity_arr[1,i]])
        concat_outputs.append(mse_arr)
        df_outputs = pd.DataFrame(concat_outputs).T
        df_outputs.columns = [qualitie+"_"+tag+"_"+metric  for qualitie in qualities_names for tag in ["robot","human"] for metric in ["kl_div","js_dist"]] +\
                                 ["cosine_similarity_"+feature+"_"+axis for feature in ["v","av"] for axis in ["x","y","z"]] +\
                                     ["mse"]
        df_outputs.to_csv("distribution_and_similarity_"+dataset_tag+".csv")
        # concat_outputs = np.concatenate()

        # file.write(f"\n Best Cosine Similarity {prev_best_mean[0]} for Dataset {dataset_tag} - Lambda Cosine: {prev_best_mean[1]}\n")
        # file.write(f"\n Best MSE {prev_best_mean[2]} for Dataset {dataset_tag} - Lambda MSE: {prev_best_mean[3]} \n")
        labels = ["VX","VY","VZ","AVX","AVY","AVZ"]
        samples = np.linspace(0,max_gain,max_gain//steps)
        fig, axs = plt.subplots(1,2)
        for n, ax in enumerate(axs.flat):
            if n < 1:
                for coord in range(np.shape(similarity_arr)[1]):
                    ax.plot(samples,similarity_arr[:,coord], linewidth=2.0, label=labels[coord])
                ax.set_ylabel('Cosine Similarity', fontsize=14)
                ax.set_title('Gain Value Lambda vs Cosine Similarity - Dataset: ' + dataset_tag, fontsize=16)
                ax.legend()
            else:
                ax.plot(samples,mse_arr, linewidth=2.0)
                ax.set_ylabel('Mean Squared Error', fontsize=14)
                ax.set_title('Gain Value Lambda vs Mean Squared Error - Dataset: ' + dataset_tag, fontsize=16)
            ax.set_xlabel('Value of Lambda', fontsize=14)
        fig.suptitle("Effect of Lambda Gain On Robot Task Resemblance", fontsize=20)
        plt.show()
        
        fig, axs = plt.subplots(1,2)
        for n, ax in enumerate(axs.flat):
            if n < 1:
                for key, val in kl_div_arr["human"].items():
                    ax.plot(samples,val, linewidth=2.0, label=key)
                ax.set_title('Human vs Generated Output - Dataset: ' + dataset_tag, fontsize=16)
            else:
                for key, val in kl_div_arr["robot"].items():
                    ax.plot(samples,val, linewidth=2.0, label=key)
                ax.set_title('Robot vs Generated Output - Dataset: ' + dataset_tag, fontsize=16)
            ax.legend()
            ax.set_ylabel('Kullback Leibler Divergence', fontsize=14)
            ax.set_xlabel('Value of Lambda', fontsize=14)
        fig.suptitle("Effect of Lambda Gain On KL Divergence", fontsize=20)
        plt.show()

        fig, axs = plt.subplots(1,2)
        for n, ax in enumerate(axs.flat):
            if n < 1:
                for key, val in shannon_jenssen_dist_arr["human"].items():
                    ax.plot(samples,val, linewidth=2.0, label=key)
                ax.set_title('Human vs Generated Output - Dataset: ' + dataset_tag, fontsize=16)
            else:
                for key, val in shannon_jenssen_dist_arr["robot"].items():
                    ax.plot(samples,val, linewidth=2.0, label=key)
                ax.set_title('Robot vs Generated Output - Dataset: ' + dataset_tag, fontsize=16)
            ax.legend()
            ax.set_ylabel('Jensen-Shannon Distance', fontsize=14)
            ax.set_xlabel('Value of Lambda', fontsize=14)
        fig.suptitle("Effect of Lambda Gain On JS Distance", fontsize=20)
        plt.show()

    # simulation_obj.set_input_data(dataset, neutral_data=neutral_data, 
    #                                 tag="val", neutral=model_constants["neutral_style"])
    # simulation_obj.generate_output_analysis_single_ssample(tag="val", neutral=model_constants["neutral_style"], not_save=True)
    # simulation_obj.set_input_data(dataset, neutral_data=neutral_data, 
    #                                 tag="test", neutral=model_constants["neutral_style"])
    # simulation_obj.generate_output_analysis_single_ssample(tag="test", neutral=model_constants["neutral_style"], not_save=True)

if __name__=="__main__":
    main()