import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory

from dataset import PedalsDataset_MLP
from model import Pedals_MLP
import ast

EPOCHS = 3
LEARNING_RATE = 1e-3
N_LATENTS = 8

VAE_DIR = '../VAE/'
DATAFRAME_PROVA = os.path.join(VAE_DIR,'tsne_latents_dataframe.csv')
assert os.path.exists(DATAFRAME_PROVA), f"File '{DATAFRAME_PROVA}' not found"

THIS_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
MLP_PATH = f'{THIS_FOLDER_PATH}/model_MLP_{N_LATENTS}.pth'


def model_loss(pred_coords, target_coords):
    return nn.MSELoss()(pred_coords, target_coords)


def train(dataloader, model, optimizer, EPOCHS):
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for idx, batch in enumerate(dataloader):
            latents = torch.tensor([ast.literal_eval(latent) for latent in batch["latents"]], dtype=torch.float32)
            coords = torch.tensor([ast.literal_eval(coord) for coord in batch["coords"]], dtype=torch.float32)

            optimizer.zero_grad()
            pred_coords = model(latents)
            loss = model_loss(pred_coords, coords)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), MLP_PATH)


if __name__ == '__main__':
    model = Pedals_MLP(input_dim=N_LATENTS, output_dim=2)
    dataframe = pd.read_csv(DATAFRAME_PROVA)
    dataset = PedalsDataset_MLP(dataframe)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    train(dataloader, model, optimizer, EPOCHS)
    