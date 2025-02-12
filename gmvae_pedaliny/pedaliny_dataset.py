'''import pandas as pd
import os
import numpy as np
import librosa
from torch.utils.data import Dataset


class PedalinyDataset(Dataset):

    def __init__(self, dataframe_path, mode):

        self.df = pd.read_csv(dataframe_path)
        self.mode = mode


    def __len__(self):
        return len(self.df)
    

    def get_unique_labels(self):
        return self.df['label'].unique()


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["audio_path"] if self.mode == "audio" else row["sweep_path"]

        audio = librosa.load(filepath, sr=44100)[0]
        return {
            "audio": audio,
            "name": row["name"],
            "label": row["label"],
            "g": row["g_value"],
            "t": row["t_value"],
            "audio_path": row['audio_path'],  # Assuming 'audio_path' is in the item
            "sweep_path": row['sweep_path']
        }


if __name__ == "__main__":
    dataframe_path = "/home/ardan/ARDAN/PEDALINY/pedaliny_dataset/combined_dataframe.csv"
    dataset = PedalinyDataset(dataframe_path, mode="sweep")
    print(f"Number of samples in dataset: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample name: {sample['name']}")
    print(f"Sample label: {sample['label']}")
    print(f"Sample g: {sample['g']}")
    print(f"Sample t: {sample['t']}")
    print(f"Sample audio shape: {sample['audio'].shape}")'''





import pandas as pd
import librosa
from torch.utils.data import Dataset


DATAFRAME_PATH_PROVA = "/home/ardan/ARDAN/PEDALINY/pedaliny_dataframe_ultimate.csv"


class PedalinyDataset(Dataset):

    def __init__(self, dataframe_path, mode):

        self.df = pd.read_csv(dataframe_path)
        self.mode = mode


    def __len__(self):
        return len(self.df)
    
    def get_unique_labels(self):
        return self.df['pedal_name'].unique() 

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.mode == "audio":
            filepath = row["audio_path"]
        elif self.mode == "sweep":
            filepath = row["sweep_path"]
        elif self.mode == "noise":
            filepath = row["noise_path"]

        audio, _ = librosa.load(filepath, sr=48000)

        return {
            "audio": audio,
            "label": row["pedal_name"],  
            "g": row["g_value"],          
            "t": row["t_value"]
        }


if __name__ == "__main__":
    dataset_audio = PedalinyDataset(DATAFRAME_PATH_PROVA, mode="audio")
    print(f"Number of samples in dataset: {len(dataset_audio)}")
    sample_audio = dataset_audio[0]
    print(f"Sample audio shape: {sample_audio['audio'].shape}")
    print(f"Sample label: {sample_audio['label']}")
    print(f"Sample g: {sample_audio['g']}")
    print(f"Sample t: {sample_audio['t']}")
    