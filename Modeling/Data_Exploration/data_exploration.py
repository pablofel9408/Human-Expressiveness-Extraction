import pandas as pd
import numpy as np

from .latent_generation import LatentGeneration

import os

class DataExploration():
    def __init__(self, config, save=False) -> None:
        self.save_flag = save

        self.dataset_tags = None
        self.latent_obj = LatentGeneration(config)

        self.dirpath_latent = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Data_Exploration\\Output Data\\Latent Encodings\\"
        self.dirpath_neutral = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Data_Exploration\\Output Data\\Neutral Dataset\\"
        self.dirpath_mean_latent = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Data_Exploration\\Output Data\\Mean Latent Encodings\\"

    def load_data(self,data, data_tags):
        self.dataset_tags = data_tags
        self.dataset_signals = data

    def process_neutral_emotion(self):
        dataset_nee = {i: {j:{} for j in ['PAIB', 'EMLA', 'NABA', 'SALE']} for i in self.dataset_tags.keys() if i not in ["emotion","actor"]}
        for tag, value in self.dataset_tags.items():
            if tag in ["emotion","actor"]:
                continue
            tags_train = pd.DataFrame(value)
            for act in tags_train["act"].unique():
                indices_nee_actor = np.where((tags_train["emo"]=="NEE") & (tags_train["act"]==act))
                dataset_nee_actor = self.dataset_signals[tag][0][indices_nee_actor]
                dataset_nee[tag][act] = {"indices":indices_nee_actor, "data":dataset_nee_actor}

                if self.save_flag:
                    np.save(os.path.join(self.dirpath_neutral, tag + "_neutral_"+ act + ".npy"),
                                dataset_nee_actor)
                    np.save(os.path.join(self.dirpath_neutral, tag + "_neutral_indices_"+ act + ".npy"),
                                indices_nee_actor)
            
        return dataset_nee

    def generate_latent_neutral(self):
        dataset_nee = self.process_neutral_emotion()
        for tag, value in dataset_nee.items():
            for act, data in value.items():
                dataset_nee[tag][act]["latent"] = self.latent_obj.latent_encoding(data["data"])

                if self.save_flag:
                    np.save(os.path.join(self.dirpath_latent, tag + "_latent_"+ act + ".npy"),
                                dataset_nee[tag][act]["latent"])
                    np.save(os.path.join(self.dirpath_mean_latent, tag + "_latent_mean_"+ act + ".npy"),
                                np.mean(dataset_nee[tag][act]["latent"],axis=0))

        return dataset_nee
