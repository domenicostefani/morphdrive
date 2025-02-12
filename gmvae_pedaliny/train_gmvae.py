import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from gmvae import Pedaliny_GMVAE
from pedaliny_dataset import PedalinyDataset


BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 3000
DATAFRAME_PATH = "/home/ardan/ARDAN/PEDALINY/pedaliny_dataframe_ultimate.csv"


def train():

    dataset = PedalinyDataset(DATAFRAME_PATH, mode="sweeps")
    unique_labels = len(dataset.get_unique_labels())

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = Pedaliny_GMVAE(input_dim=1, latent_dim=8, n_pedals=unique_labels).to('cuda')

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    loss_function = nn.MSELoss()


    for epoch in range(EPOCHS):
        for batch in dataloader:
            audio = batch["audio"].float().to('cuda').unsqueeze(1)
    
            output, _, _, _, _, _, _ = model(audio)
            loss = loss_function(output, audio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "/home/ardan/ARDAN/PEDALINY/pedaliny_gmvae_8.pth")
            

if __name__ == "__main__":
    train()

