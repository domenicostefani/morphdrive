import pandas as pd
import librosa
from torch.utils.data import Dataset
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory

DATASET_DIR = '../dataset_32000Hz/'
DATAFRAME_PATH = os.path.join(DATASET_DIR,'pedals_dataframe.csv')

class PedalsDataset_VAE(Dataset):
    def __init__(self, dataframe, sr, mode, offset=0, length=-1):
        self.df = dataframe
        self.mode = mode
        self.sr = sr
        self.offset = offset
        self.length = length

    def __len__(self):
        return len(self.df)
    
    def get_unique_labels(self):
        return self.df['pedal_name'].unique() 
    
    def get_labels_list(self):
        return self.df['pedal_name'].tolist()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.mode == "audio":
            filepath = row["audio_path"]
        elif self.mode == "sweep1":
            filepath = row["sweep_path1"]
        elif self.mode == "sweep2":
            filepath = row["sweep_path2"]
        elif self.mode == "sweep3":
            filepath = row["sweep_path3"]
        elif self.mode == "noise":
            filepath = row["noise_path"]
        else:
            raise ValueError("Mode must be either 'audio', 'sweep[1-3]' or 'noise'")

        assert os.path.exists(filepath), f"File {filepath} not found"
        audio, _ = librosa.load(filepath, sr=self.sr)
        if self.offset == 0 and self.length == -1:
            pass
        else:
            audio = audio[self.offset:self.length+self.offset] 

        return {
            "audio": audio,
            "label": row["pedal_name"],  
            "g": row["g_value"],          
            "t": row["t_value"]
        }

if __name__ == "__main__":
    dataframe = pd.read_csv(DATAFRAME_PATH)
    dataset_audio = PedalsDataset_VAE(dataframe, sr=32000, mode="sweep1")
    print(f"Number of samples in dataset: {len(dataset_audio)}")
    sample_audio = dataset_audio[0]
    print(f"Sample audio shape: {sample_audio['audio'].shape}")
    print(f"Sample label: {sample_audio['label']}")
    print(f"Sample g: {sample_audio['g']}")
    print(f"Sample t: {sample_audio['t']}")
    