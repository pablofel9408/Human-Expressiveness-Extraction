import os
import sys

import pathlib
import numpy as np 
import torch 
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import utilities
from Robot_Feature_Extraction.Preprocessing.preprocess_robot_datasets import PreprocessRobotData
from Robot_Feature_Extraction.Preprocessing.process_human_data import PreprocessHumanData
from Robot_Feature_Extraction.Modeling.START_modeling_process import ModelingProc

def pca_visualization(dataset,dataset_tags):
    
    shape = np.shape(dataset)
    # input_data = np.array([features_2d.flatten() for features_2d in input_data])
    dataset = dataset.reshape(shape[0],shape[1]*shape[2])
    print(np.shape(dataset))
    input_df = pd.DataFrame(dataset)
    # input_df = pd.concat([input_df, self.emotion_data], ignore_index=True, axis=1) 
    emotion_data = pd.DataFrame(dataset_tags)
    print(emotion_data.shape)
    print(emotion_data.head())
    human_train_dataset = pd.concat([input_df, emotion_data], ignore_index=True, axis=1)
    human_train_dataset.columns = [i for i in range(0,shape[1]*shape[2])] + ["emo", "user"]

    human_train_dataset = human_train_dataset[human_train_dataset["emo"]!="NEE"]
    columns = [i for i in range(0,shape[1]*(shape[2]))]
    pca_50 = PCA(n_components=2)
    tsne_results = pca_50.fit_transform(human_train_dataset[[i for i in range(0,len(human_train_dataset.columns)-2)]])
    # tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=300,
    #             init="random")
    # tsne_results = tsne.fit_transform(tsne_results)

    human_train_dataset['pca-2d-one'] = tsne_results[:,0]
    human_train_dataset['pca-2d-two'] = tsne_results[:,1]
    # df_subset['tsne-2d-three'] = tsne_results[:,2]

    sns.scatterplot(
            x="pca-2d-one", y="pca-2d-two",
            hue=human_train_dataset['emo'],
            data=human_train_dataset,
            legend="full",
            alpha=0.3,
            marker='o',
            linewidths=2
    )
    plt.legend(loc='upper right')
    plt.suptitle('PCA 2D Results', fontsize=20)
    plt.show()

    pca_50 = PCA(n_components=50)
    tsne_results = pca_50.fit_transform(human_train_dataset[[i for i in range(0,len(human_train_dataset.columns)-4)]])
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000,
                init="random")
    tsne_results = tsne.fit_transform(tsne_results)

    human_train_dataset['tsne-2d-one'] = tsne_results[:,0]
    human_train_dataset['tsne-2d-two'] = tsne_results[:,1]

    sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=human_train_dataset['emo'],
            data=human_train_dataset,
            legend="full",
            alpha=0.3,
            marker='o',
            linewidths=2
        )

    plt.show()

    for user in human_train_dataset["user"].unique():
        sns.scatterplot(
            x="pca-2d-one", y="pca-2d-two",
            hue=human_train_dataset[human_train_dataset["user"]==user]['emo'],
            data=human_train_dataset[human_train_dataset["user"]==user],
            legend="full",
            alpha=0.3,
            marker='o',
            linewidths=2
        )
        plt.legend(loc='upper right')
        plt.suptitle('PCA 2D Results - Subject: ' + user, fontsize=20)
        plt.show()

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

    # dir_path = pathlib.Path(__file__).parent / "Config Files"
    dir_path = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Robot_Feature_Extraction\\Config Files"

    task_name = task_name.lower()
    filepath_dataset_cst = os.path.join(dir_path, task_name + "_dataset_config.json")
    filepath_model_cst = os.path.join(dir_path, task_name + "_model_system_config.json")
    
    dataset_constants = utilities.load_raw_constants(filepath_dataset_cst)

    if task_name == "robot":
        prepros_obj = PreprocessRobotData(dataset_constants)
        prepros_obj.start_preprocessing(tag='train')
    elif task_name == "human":
        prepros_obj = PreprocessHumanData(dataset_constants)
        prepros_obj.load_recursive_pt(dataset_constants["dataset_paths"])
        # prepros_obj.plot_data()

    dataset = prepros_obj.return_data_()
    # for key,value in dataset.items():
    #     if len(value)>1:
    #         print(np.shape(value))
    #         print(f"Dataset {key}, Mean value: {np.mean(value[1],axis=0)}, Standard deviation{np.std(value[1],axis=0)}")
    #     else:
    #         print(f"Dataset {key}, Mean value: {np.mean(value,axis=0)}, Standard deviation{np.std(value,axis=0)}")

    model_constants = utilities.load_raw_constants(filepath_model_cst)
    model_constants["task_name"] = task_name

    modeling_proc_obj = ModelingProc(model_constants)

    if task_name=="human":
        if dataset_constants["data_labels"]:
            dataset_tags = prepros_obj.return_data_tags()

        if dataset_constants["expressive_data"]:
            dataset = (dataset,dataset_tags)
    
    # pca_visualization(dataset[0]["train"][0],dataset_tags["train"])
    aux_general_dataset = {}
    for key, value in dataset_tags.items():
        if key in ["emotion", "actor"]:
            continue
        
        aux_pd = pd.DataFrame(value)
        aux_pd_nee = aux_pd[aux_pd["emo"]=="NEE"]
        aux_pd = aux_pd[aux_pd["emo"]!="NEE"]
        # print(aux_pd_nee.index.values)
        # print(len(aux_pd_nee.index.values))
        dataset_tags[key] = aux_pd.to_dict()
        aux_dataset = dataset[0][key][0][aux_pd.index.values] 
        aux_dataset_laban_qual = dataset[0][key][1][aux_pd.index.values] 
        aux_general_dataset[key] = (aux_dataset,aux_dataset_laban_qual)
    
        print(len(dataset_tags[key]["emo"]))
        print(len(dataset_tags[key]["act"]))
        print(np.shape(aux_dataset))
        print(np.shape(aux_dataset_laban_qual))
    dataset = (aux_general_dataset,dataset_tags)

    modeling_proc_obj.set_input_data(dataset, expressive=dataset_constants["expressive_data"])
    
    if not model_constants["pretrained_model"]:
        if dataset_constants["expressive_data"]:
            history, best_model_wts, model = modeling_proc_obj.training_loop_laban_qualities()
        else:
            history, best_model_wts, model = modeling_proc_obj.training_loop()
    else:
        epoch, trainLoss, valLoss, model = modeling_proc_obj.load_model(evalFlag=True)

    for p in model.parameters():
            p.requires_grad = False

    modeling_proc_obj.visualize_output(model, train=False)


if __name__=="__main__":
    main()