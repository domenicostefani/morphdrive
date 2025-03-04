import pandas as pd
from torch.utils.data import Dataset

DATAFRAME_PATH_PROVA = "/home/ardan/ARDAN/PEDALINY/tsne_latents_dataframe.csv"


class PedalsDataset_MLP(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        latents = row["latents"]
        coords = row["coords"]

        return {
            "latents": latents,
            "coords": coords
        }


if __name__ == "__main__":
    dataframe = pd.read_csv(DATAFRAME_PATH_PROVA)
    dataset_audio = PedalsDataset_MLP(dataframe)
    print(f"Number of samples in dataset: {len(dataset_audio)}")
    item = dataset_audio[0]
    print(item["latents"])
    print(item["coords"])