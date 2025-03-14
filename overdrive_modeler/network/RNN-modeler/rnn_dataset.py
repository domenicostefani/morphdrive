import soundfile as sf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Pedals_Dataset_RNN(Dataset):
    def __init__(self, full_dataframe, reduction_dataframe, unprocessed_file_path, win_len, sr=48000, pre_room=0):
        super().__init__()

        self.win_len = win_len
        self.sr = sr
        self.pre_room = pre_room

        original_wav, _ = sf.read(unprocessed_file_path)
        '''assert sr_x == sr, f"Sampling rate mismatch for original signal {original_path}"
        if len(original_wav.shape) == 1:
            original_wav = original_wav[..., None]  # Ensure channel dimension
        original_wav = original_wav[..., 0:1]  # Support mono only'''
        #self.wav_x = torch.from_numpy(original_wav.transpose(1, 0))  
        self.wav_x = np.expand_dims(original_wav, axis=0)

        #print(f"Original signal length: {self.wav_x.shape[-1]}")

        self.wav_x_chunks = []
        last_chunk_start_frame = self.wav_x.shape[-1] - win_len - pre_room + 1
        for offset in range(pre_room, last_chunk_start_frame, win_len):
            chunk = self.wav_x[:, offset - pre_room : offset + win_len]
            self.wav_x_chunks.append(torch.tensor(chunk, dtype=torch.float32))  

        #print(f"Number of wav_x_chunks: {len(self.wav_x_chunks)}")

        # TUTTA STA ROBA VA SISTEMATA 
        df_full = pd.read_csv(full_dataframe)
        df_tsne = pd.read_csv(reduction_dataframe)
        df_merged = df_full.merge(df_tsne, left_on=["pedal_name", "g_value", "t_value"], right_on=["label_name", "gain", "tone"])
        df_merged = df_merged[["label_name", "gain", "tone", "audio_path", "latents"]]
        print(f"Total samples in merged dataframe: {len(df_merged)}")

        self.df = df_merged

        self.conds = torch.tensor(
            [np.array(eval(latent), dtype=np.float32) for latent in self.df["latents"]],
            dtype=torch.float32
        )

        self.wav_y_chunks = []
        self.conds_chunks = []
        self.wav_x_chunk_indices = []

        for idx, row in self.df.iterrows():
            audio_path = row["audio_path"]
            cond = self.conds[idx]

            processed_wav, sr_y = sf.read(audio_path)

            if processed_wav.ndim > 1:
                processed_wav = processed_wav[:, 0] 
            processed_wav = torch.tensor(processed_wav, dtype=torch.float32).unsqueeze(0)  

            last_chunk_start_frame = processed_wav.shape[-1] - win_len - pre_room + 1
            for offset in range(pre_room, last_chunk_start_frame, win_len):
                self.wav_y_chunks.append(processed_wav[:, offset - pre_room : offset + win_len])
                self.conds_chunks.append(cond)

                corresponding_wav_x_index = (offset - pre_room) // win_len
                self.wav_x_chunk_indices.append(corresponding_wav_x_index)

        print(f"Total wav_y_chunks: {len(self.wav_y_chunks)}")
        print(f"Total conditions: {len(self.conds_chunks)}")
        print(f"Total wav_x_chunk_indices: {len(self.wav_x_chunk_indices)}")

        assert len(self.wav_y_chunks) == len(self.conds_chunks), "Mismatch in chunk count between wav_y and conditions"
        assert len(self.wav_y_chunks) == len(self.wav_x_chunk_indices), "Mismatch in chunk count between wav_y and wav_x indices"


    def __getitem__(self, idx):
        wav_y = self.wav_y_chunks[idx]
        cond = self.conds_chunks[idx]
        wav_x = self.wav_x_chunks[self.wav_x_chunk_indices[idx]]
        return wav_x, wav_y, cond

    def __len__(self):
        return len(self.wav_y_chunks)
    

if __name__ == "__main__":

    DATASET_DIR = os.path.join('..', 'dataset')
    FULL_DATAFRAME = os.path.join(DATASET_DIR, 'pedals_dataframe.csv')
    UNPROCESSED_FILE_PATH = os.path.join(DATASET_DIR, 'input','a_0-unprocessed_input.wav')

    REDUCTION_DATAFRAME = "../VAE/1-2025-03-11_16-50_tsne_latents.csv" # Choose the reduction dataframe to use

    dataset = Pedals_Dataset_RNN(FULL_DATAFRAME, REDUCTION_DATAFRAME, UNPROCESSED_FILE_PATH, win_len=4096)
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)

    # def save_to_audio_file(wav, path, sr=48000):
    #     wav = wav.squeeze(0).numpy()
    #     sf.write(path, wav, sr)

    #save_to_audio_file(dataset[0][0], "wav_x.wav")
    #save_to_audio_file(dataset[0][1], "wav_y.wav")

    