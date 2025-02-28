import pandas as pd
import librosa
from torch.utils.data import Dataset
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory


DATASET_DIR = '../dataset/'
DATAFRAME_PATH = os.path.join(DATASET_DIR,'pedals_dataframe_ultimate.csv')


class PedalsDataset(Dataset):

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
        else:
            raise ValueError("Mode must be either 'audio', 'sweep' or 'noise'")

        assert os.path.exists(filepath), f"File {filepath} not found"
        audio, _ = librosa.load(filepath, sr=48000)

        return {
            "audio": audio,
            "label": row["pedal_name"],  
            "g": row["g_value"],          
            "t": row["t_value"]
        }


if __name__ == "__main__":
    dataset_audio = PedalsDataset(DATAFRAME_PATH, mode="audio")
    print(f"Number of samples in dataset: {len(dataset_audio)}")
    sample_audio = dataset_audio[0]
    print(f"Sample audio shape: {sample_audio['audio'].shape}")
    print(f"Sample label: {sample_audio['label']}")
    print(f"Sample g: {sample_audio['g']}")
    print(f"Sample t: {sample_audio['t']}")
    