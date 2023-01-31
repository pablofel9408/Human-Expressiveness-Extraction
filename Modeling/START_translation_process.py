import os 
import sys
import json 
import pathlib

import torch 
import random 
import numpy as np
import pandas as pd

import utilities
from GAN_Translation.General_Functions.START_gan_proc import ModelingGAN
from Robot_Feature_Extraction.Preprocessing.preprocess_robot_datasets import PreprocessRobotData
from Robot_Feature_Extraction.Preprocessing.process_human_data import PreprocessHumanData
from Translation_Process_Simulation.Simulation_Scripts import START_simulation

from Additional_Scripts.construct_laban_qualities import Laban_Dict
from Additional_Scripts.latent_visualization import Latent_Viz
import seaborn as sns
import matplotlib.pyplot as plt

########
# Remember: 
# model_GAN_4.681271743774414_0.pth
# model_GAN_4.616969495206265_0.pth
########

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

    if (keyword=="sim") and not model_constants["pretrained"]:
        print("To tooggle simulation a pretrained model is required")
        # sys.exit(1)
        sim=True
    elif (keyword!="sim"):
        sim=False
    else:
        sim=True

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

        if model_constants["neutral_style"] and laban_analysis!="latent_rep" and keyword!="dataset_analysis_lat":
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

    if not keyword == "dataset_analysis":
        modeling_proc_obj = ModelingGAN(model_constants)
        modeling_proc_obj.set_input_data(dataset, expressive=dataset_constants_human["expressive_data"])

        if not model_constants["pretrained"]:
            if not model_constants["neutral_style"]:
                history, best_model_wts, model = modeling_proc_obj.training_loop_gan()
            else:
                history, best_model_wts, model = modeling_proc_obj.training_loop_gan_neutral_style()
        else:
            epoch, trainLoss, valLoss, model = modeling_proc_obj.load_model(evalFlag=True)

        model.eval()
        print(sim)
        if sim:
            simulation_obj = START_simulation.Simulation_Methods(simulation_constants, model, dataset_constants_robot["scalers_path"])
            neutral_data = modeling_proc_obj.return_neutral_data() if model_constants["neutral_style"] else None
            simulation_obj.set_input_data(dataset,neutral_data=neutral_data, neutral=model_constants["neutral_style"])
            simulation_obj.start_similation(neutral=model_constants["neutral_style"])
    
    else:
        print("here")
        if single_sample=="single_sample":
            print("here")
            modeling_proc_obj = ModelingGAN(model_constants)
            modeling_proc_obj.set_input_data(dataset,expressive=dataset_constants_human["expressive_data"])
            epoch, trainLoss, valLoss, model = modeling_proc_obj.load_model(evalFlag=True)
            simulation_obj = START_simulation.Simulation_Methods(simulation_constants, model, dataset_constants_robot["scalers_path"])
            if model_constants["neutral_style"]:
                neutral_data = modeling_proc_obj.return_neutral_data()
                simulation_obj.set_input_data(dataset, neutral_data=neutral_data,
                                                tag="train", neutral=model_constants["neutral_style"])
                simulation_obj.generate_output_analysis_single_ssample(tag="train", neutral=model_constants["neutral_style"])
                simulation_obj.set_input_data(dataset, neutral_data=neutral_data, 
                                                tag="val", neutral=model_constants["neutral_style"])
                simulation_obj.generate_output_analysis_single_ssample(tag="val", neutral=model_constants["neutral_style"])
                simulation_obj.set_input_data(dataset, neutral_data=neutral_data, 
                                                tag="test", neutral=model_constants["neutral_style"])
                simulation_obj.generate_output_analysis_single_ssample(tag="test", neutral=model_constants["neutral_style"])
            else: 
                modeling_proc_obj.set_input_data(dataset, expressive=dataset_constants_human["expressive_data"])
                simulation_obj.set_input_data(dataset, tag="train")
                simulation_obj.generate_output_analysis_single_ssample(tag="train")
                simulation_obj.set_input_data(dataset, tag="val")
                simulation_obj.generate_output_analysis_single_ssample(tag="val")
                simulation_obj.set_input_data(dataset, tag="test")
                simulation_obj.generate_output_analysis_single_ssample(tag="test")
        
        
        elif laban_analysis=="laban": 
            print(type(dataset_rob["train"]))
            for key,value in dataset_rob.items():
                laban_study = Laban_Dict(value)
                df = laban_study.start_process(name="qualities_robot_dataset_"+key,human=False)
                cols = df.columns
                fig, axes = plt.subplots(nrows=4, ncols=1,sharex=False, sharey=False)
                for n,ax in enumerate(axes):
                    sns.histplot(df, x=cols[n],ax=ax, bins=1000)
                fig.suptitle('Expressive Qualities Robot Dataset '+key)
            plt.show()
        
    if laban_analysis=="latent_rep":
        lat_viz = Latent_Viz()
        # lat_viz.load_emotional_data()

        if keyword == "dataset_analysis_lat":
            # new_raw_data = np.load("C:\\Users\\posorio\\Documents\\Expressive movement\\Data Engineering\\Main Scripts\\Data acquisition\\Human Expressive Data\\Data  Engineering Emotion Dataset\\Data  Engineering Emotion Dataset\\new_dataset_raw_data.npy")
            df = lat_viz.preprocess_data(dataset_hum["train"][0],dataset_tags_hum["train"])
            # sys.exit()
            lat_viz.visualization(df)
            plt.show()

            latent_dataset = lat_viz.generate_latent(model_constants["pretrained_model_path_Human_VAE"],
                                                    model_constants["model_config"]["generator"]["human_vae"]["model_config"], 
                                                    dataset_hum["train"][0], vae=True)
            df = lat_viz.preprocess_data(latent_dataset,dataset_tags_hum["train"])
            lat_viz.visualization(df)
            plt.show()
            
        else:
            if not model_constants["neutral_style"]:
                generated_dataset = lat_viz.generate_translation_latent(model,dataset_hum["train"],dataset_rob["train"])
                latent_dataset = lat_viz.generate_latent(model_constants["pretrained_model_path_Human_VAE"],
                                                        model_constants["model_config"]["generator"]["human_vae"]["model_config"], 
                                                        generated_dataset)
                df = lat_viz.preprocess_data(latent_dataset)
                lat_viz.visualization(df)
                plt.show()
            else:
                dataset_tag = "train"
                neutral_data = modeling_proc_obj.return_neutral_data()
                generated_dataset = lat_viz.generate_translation_latent_neutral_style(model,dataset_hum[dataset_tag][0],
                                                                                        dataset_rob[dataset_tag],
                                                                                        neutral_data[dataset_tag],
                                                                                        dataset_tags_hum[dataset_tag]["act"])
                latent_dataset = lat_viz.generate_latent(model_constants["pretrained_model_path_Human_VAE"],
                                                        model_constants["model_config"]["generator"]["human_vae"]["model_config"], 
                                                        generated_dataset)
                df = lat_viz.preprocess_data(latent_dataset,dataset_tags_hum["train"])
                lat_viz.visualization(df)
                plt.show()
                

        # simulation_obj = START_simulation.Simulation_Methods(simulation_constants, None, dataset_constants_robot["scalers_path"])
        # for n,data in enumerate(dataset):
        #     # if n==1:
        #     #     data_emotion = data[1]
        #     #     data = data[0]
        #     for key, value in data.items():
        #         if n==0:
        #             simulation_obj.analyze_dataset_laban_qualities(value,use_acc=False,save=True, tag=key+"_robot")
        #         else:
        #             simulation_obj.analyze_dataset_laban_qualities(value,use_acc=True,save=True, tag=key+"_human")
        #             # data_aux = data_emotion[key]
        #             # df = pd.DataFrame(data_aux)
        #             # df.to_csv("emotion_dataset_" + key + ".csv")

if __name__=="__main__":
    main()