import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from gmvae import Pedals_GMVAE
from pedals_dataset import PedalsDataset
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory

DATASET_DIR = '../dataset/'
OUTPUT_DIR = '../output/'
DATAFRAME_PATH = os.path.join(DATASET_DIR,'pedals_dataframe_ultimate.csv')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 3000

def train():

    dataset = PedalsDataset(DATAFRAME_PATH, mode="sweep")
    unique_labels = len(dataset.get_unique_labels())

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = Pedals_GMVAE(input_dim=1, latent_dim=8, n_pedals=unique_labels).to('cuda')

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

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "Pedals_GMVAE_8.pth"))
            
if __name__ == "__main__":
    train()

