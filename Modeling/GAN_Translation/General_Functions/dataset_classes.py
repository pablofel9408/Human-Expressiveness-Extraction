import random
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, input_data):
        self.data = input_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx]
        return label

class CustomDataset_Hum(Dataset):
    def __init__(self, input_data, len_rob):
        self.data = input_data
        self.target_len = len_rob
        self.past_indices = [None,None]
        
    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):
        indices = random.sample(range(0,self.data.size(0)-1),2)
        label1 = self.data[indices[0]]
        label2 = self.data[indices[1]]
        return label1, label2

class CustomDataset_Hum_Style_Neutral(Dataset):
    def __init__(self, input_data, neutral, participants_tags, len_rob):
        self.data = input_data
        self.neutral_data = neutral
        self.participants_tags = participants_tags
        self.target_len = len_rob
        self.past_indices = [None,None]
        
    def __len__(self):
        return self.target_len

    def __getitem__(self, idx):

        indices = random.sample(range(0,self.data.size(0)-1),2)
        label1 = self.data[indices[0]]
        participant = self.participants_tags[indices[0]]
        neutral_label1 = self.neutral_data[participant]

        label2 = self.data[indices[1]]
        participant2 = self.participants_tags[indices[1]]
        neutral_label2 = self.neutral_data[participant2]

        return label1, neutral_label1, label2, neutral_label2

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)