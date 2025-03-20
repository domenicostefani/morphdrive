import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory

from mlp_dataset import PedalsDataset_MLP
from mlp_model import Pedals_MLP
import ast

EPOCHS = 1500
LEARNING_RATE = 1e-3
N_LATENTS = 8
VALID = True

# REDUCTION_DATAFRAME = "../VAE/4-PAPER_2025-03-05_23-11_tsne_latents.csv"
REDUCTION_DATAFRAME = "../VAE/8-PAPER_2025-03-16_13-18_tsne_latents.csv"
assert os.path.exists(REDUCTION_DATAFRAME), f"File '{REDUCTION_DATAFRAME}' not found"

def model_loss(pred_coords, target_coords):
    return nn.MSELoss()(pred_coords, target_coords)


def train(dataloader, valid_loader, model, optimizer, EPOCHS):
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for idx, batch in enumerate(dataloader):
            latents = torch.tensor([ast.literal_eval(latent) for latent in batch["latents"]], dtype=torch.float32)
            coords = torch.tensor([ast.literal_eval(coord) for coord in batch["coords"]], dtype=torch.float32)

            optimizer.zero_grad()
            pred_latents = model(coords)
            loss = model_loss(pred_latents, latents)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            if valid_loader is None:
                print(f"Epoch: {epoch + 1},\tLoss: {total_loss / len(dataloader)}")
            else:
                model.eval()
                total_valid_loss = 0
                for idx, batch in enumerate(valid_loader):
                    latents = torch.tensor([ast.literal_eval(latent) for latent in batch["latents"]], dtype=torch.float32)
                    coords = torch.tensor([ast.literal_eval(coord) for coord in batch["coords"]], dtype=torch.float32)

                    pred_latents = model(coords)
                    loss = model_loss(pred_latents, latents)
                    total_valid_loss += loss.item()

                print(f"Epoch: {epoch + 1},\tLoss: {total_loss / len(dataloader)},\tValid Loss: {total_valid_loss / len(valid_loader)}")



        
        #print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    savename = os.path.basename(REDUCTION_DATAFRAME).split('_')[0]+"_mlp.pth"
    torch.save(model.state_dict(), savename)
    print(f"Model saved as {savename}") 

if __name__ == '__main__':
    model = Pedals_MLP(input_dim=2, output_dim=N_LATENTS)
    dataframe = pd.read_csv(REDUCTION_DATAFRAME)
    dataset = PedalsDataset_MLP(dataframe)
    if VALID:
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, drop_last=True)
    else:
        train_dataset = dataset
        valid_dataloader = None
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    train(train_dataloader, valid_dataloader, model, optimizer, EPOCHS)
    