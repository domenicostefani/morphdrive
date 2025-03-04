import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import Pedals_VAE
from dataset import PedalsDataset_VAE
import wandb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import seaborn as sns
import pandas as pd
from plotters import plot_spectrograms_to_wandb, load_audio_to_wandb, pca_on_latents, tsne_on_latents
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory

DATASET_DIR = '../dataset_32000Hz/'
DATAFRAME_PATH = os.path.join(DATASET_DIR,'pedals_dataframe.csv')

SR = 32000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 2
N_LATENTS = 8
VERSION = 3
THIS_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

SAVE_MODEL_PATH = os.path.abspath(f'model_VAE_{N_LATENTS}_V{VERSION}.pth')
SAVE_LATENTS_PATH = os.path.abspath(f'latents_VAE_{N_LATENTS}_V{VERSION}.csv')
LOGS = False

PEDALS = ['bigfella', 'chime', 'silkdrive', 'zendrive']

WANDB_PROJECT_NAME = "Pedals_VAE_marzo"
WANDB_ENTITY = 'francesco-dalri-2'

###################################################################################################

if LOGS:
    wandb.login()
    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY)

def model_loss(target_audio, x_predict, mu, logvar):
    recon_loss1 = nn.MSELoss(reduction='sum')(target_audio, x_predict)
    recon_loss2 = nn.HuberLoss(reduction='sum')(target_audio, x_predict)
    recon_loss = torch.mean(recon_loss1 + recon_loss2, dim=0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + kl_loss * 0.1
    return total_loss, recon_loss1, recon_loss2, kl_loss


def mse_loss(target_audio, x_predict):
    return nn.MSELoss(reduction='mean')(target_audio, x_predict)


def weights_init(model):
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(model.weight)


def filter_dataframe(dataframe, pedals):
    return dataframe[dataframe.pedal_name.isin(pedals)]


def lr_lambda(epoch):
        return 0.5 ** (epoch // 500)


def train(dataloader, model, optimizer, EPOCHS):
    model.train()
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(EPOCHS):

        total_loss = 0
        total_model_loss = 0
        total_recon_loss1 = 0
        total_recon_loss2 = 0
        total_kl_loss = 0

        for batch in dataloader:
            audio = batch["audio"].float().to('cuda').unsqueeze(1)    
            output, mu, logvar, _ = model(audio)

            loss, recon_loss1, recon_loss2, kl_loss = model_loss(audio, output, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_model_loss += loss.item()
            total_recon_loss1 += recon_loss1.item()
            total_recon_loss2 += recon_loss2.item()
            total_kl_loss += kl_loss.item()

        scheduler.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}")
        if LOGS:
            wandb.log({"train/total_loss": total_loss / len(dataloader),
                   "train/model_loss": total_model_loss / len(dataloader),
                   "train/recon_loss_mse": total_recon_loss1 / len(dataloader),
                   "train/recon_loss_spectral": total_recon_loss2 / len(dataloader),
                   "train/kl_loss": total_kl_loss / len(dataloader)
                   })
            if epoch % 10 == 0:
                plot_spectrograms_to_wandb(output[0], audio[0], sr=SR)
                load_audio_to_wandb(audio[0], output[0], sr=SR)

    torch.save(model.state_dict(), SAVE_MODEL_PATH)




def test(dataloader, model):
    model.eval()  

    total_loss = 0

    for batch in dataloader:
        audio = batch["audio"].float().to('cuda').unsqueeze(1)

        output, _, _, _ = model(audio)

        test_loss = mse_loss(audio, output)
        total_loss += test_loss.item()

    print(f"Test Loss: {total_loss / len(dataloader)}")

    if LOGS:
        wandb.log({
            "test/test_total_loss": total_loss / len(dataloader),
        })




def extract_latents(dataloader, model, label_to_index):
    model.eval()
    all_data = []

    for batch in dataloader:
        audio = batch["audio"].float().to("cuda").unsqueeze(1)
        target_class = torch.tensor([label_to_index[label] for label in batch["label"]], dtype=torch.long).to("cuda")
        target_gain = batch["g"].clone().detach().to(torch.long).to("cuda")
        target_tone = batch["t"].clone().detach().to(torch.long).to("cuda")

        _, _, _, z = model(audio)

        for i in range(z.shape[0]):
            all_data.append({
                "label": target_class[i].item(),
                "gain": target_gain[i].item(),
                "tone": target_tone[i].item(),
                "latents": z[i].cpu().detach().numpy().tolist()
            })

    df = pd.DataFrame(all_data)
    df.to_csv(SAVE_LATENTS_PATH, index=False)
    print("Latents saved")




if __name__ == "__main__":
    assert os.path.exists(DATAFRAME_PATH), f"Dataframe not found at {DATAFRAME_PATH}"
    dataframe = pd.read_csv(DATAFRAME_PATH)
    dataframe = filter_dataframe(dataframe, PEDALS)
    dataset = Pedaliny_Dataset_VAE(dataframe, sr=SR, mode="sweep1", offset=20000, length=88200)
    labels = dataset.get_unique_labels() 
    unique_labels = len(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    dataset_size = len(dataset)
    #train_size = int(0.9 * dataset_size)  
    #test_size = dataset_size - train_size  
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"Number of training samples: {len(dataset)}")
    #print(f"Number of testing samples: {len(test_dataset)}")
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


    model = Pedals_VAE(pedal_latent_dim=N_LATENTS).to('cuda')
    model.apply(weights_init)
    optimizer_model = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    train(train_dataloader, model, optimizer_model, EPOCHS)
    test(test_dataloader, model)
    extract_latents(train_dataloader, model, label_to_index)
    pca_on_latents(index_to_label, SAVE_LATENTS_PATH, THIS_FOLDER_PATH)
    tsne_on_latents(index_to_label, SAVE_LATENTS_PATH, THIS_FOLDER_PATH)


    if LOGS:
        wandb.finish()

