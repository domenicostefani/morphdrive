import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from vae_model import Pedals_VAE
from vae_dataset import PedalsDataset_VAE
import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory
from torch.optim.lr_scheduler import LambdaLR
from plotters import plot_spectrograms_to_wandb, load_audio_to_wandb, pca_on_latents, tsne_on_latents
import datetime

# Constants
DATASET_DIR = os.path.join('..', 'dataset_32')
DATAFRAME_PATH = os.path.join(DATASET_DIR, 'pedals_dataframe.csv')
SR = 32000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 2
N_LATENTS = 8
FOLDS = 5  # Number of folds for cross-validation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THIS_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
LOGS = False

if LOGS:
    import wandb

PEDALS = ['dumkudo', 'ss2', 'theelements', 'ocd', 'tubedreamer', 'chime', 'ktr', 'kingoftone']
PEDALS = ['chime', 'bigfella']
for pedal in PEDALS:
    assert os.path.exists(os.path.join(DATASET_DIR, pedal)), f"Pedal '{pedal}' not found in '{DATASET_DIR}'"

WANDB_PROJECT_NAME = "Pedals_VAE_marzo"
WANDB_ENTITY = 'francesco-dalri-2'

if LOGS:
    wandb.login()
    #wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 8, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(22056, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        x = self.model(x)
        return self.linear(x)




def complex_to_magnitude(stft_complex):
    """Utils functions for converting complex spectrogram to magnitude."""
    stft_mag = torch.sum(stft_complex ** 2) ** 0.5
    return stft_mag


class STFTLoss(nn.Module):
    def __init__(self, win_len: int = 2048, overlap: float = 0.75, is_complex: bool = True):
        super().__init__()

        '''print('> [Loss] --- STFT Complex Loss ---')
        print('> [Loss] is complex:', is_complex)
        print('> [Loss] win_len: {}, overlap: {}'.format(win_len, overlap))'''

        self.win_len = win_len
        self.is_complex = is_complex
        self.hop_len = int((1 - overlap) * win_len)

        self.window = nn.Parameter(
            torch.from_numpy(np.hanning(win_len)).float(), requires_grad=False
        )

    def forward(self, predict: torch.tensor, target: torch.tensor):

        predict = predict.reshape(-1, predict.shape[-1])
        target = target.reshape(-1, target.shape[-1])

        stft_predict = torch.stft(
            predict, n_fft=self.win_len, window=self.window,
            hop_length=self.hop_len, return_complex=True, center=False
        )
        stft_target = torch.stft(
            target, n_fft=self.win_len, window=self.window,
            hop_length=self.hop_len, return_complex=True, center=False
        )

        if self.is_complex:
            loss_final = F.l1_loss(stft_predict, stft_target)
        else:
            stft_predict_mag = complex_to_magnitude(stft_predict)
            stft_target_mag = complex_to_magnitude(stft_target)
            loss_final = F.l1_loss(stft_predict_mag, stft_target_mag)

        return loss_final

class MRSTFTLoss(nn.Module):
    def __init__(
        self,
        scales: list = [2048, 512, 32],#[2048, 512, 128, 32],
        overlap: float = 0.75
    ):
        super().__init__()
        #print('> [Loss] --- Multi-resolution STFT Loss ---')
    
        self.scales = scales
        self.overlap = overlap
        self.num_scales = len(self.scales)

        self.windows = nn.ParameterList(
            nn.Parameter(torch.from_numpy(np.hanning(scale)).float(), requires_grad=False) for scale in self.scales
        )
        self.windows = self.windows.to(DEVICE)

    def forward(
        self,
        predict: torch.tensor, 
        target: torch.tensor
    ):

        # (B, C, T)  to (B x C, T)
        x = predict.reshape(-1, predict.shape[-1])
        x_orig = target.reshape(-1, target.shape[-1])

        amp = lambda x: x[:,:,:,0]**2 + x[:,:,:,1]**2

        stfts = []
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(
                x, 
                n_fft=scale, 
                window=self.windows[i], 
                hop_length=int((1-self.overlap)*scale), 
                center=False, 
                return_complex=False
            )
            stfts.append(amp(cur_fft))

        stfts_orig = []
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(
                x_orig, 
                n_fft=scale, 
                window=self.windows[i], 
                hop_length=int((1-self.overlap)*scale), 
                center=False, 
                return_complex=False
            )
            stfts_orig.append(amp(cur_fft))

        # Compute loss scale x batch
        lin_loss_final = 0
        log_loss_final = 0
        for i in range(self.num_scales):
            lin_loss = torch.mean(abs(stfts_orig[i] - stfts[i]))
            log_loss = torch.mean(abs(torch.log(stfts_orig[i] + 1e-4) - torch.log(stfts[i] + 1e-4)))  

            lin_loss_final += lin_loss
            log_loss_final += log_loss
        
        lin_loss_final /= self.num_scales
        log_loss_final /= self.num_scales

        return lin_loss_final + log_loss_final



def model_loss(target_audio, x_predict, mu, logvar, discriminator):
    recon_loss1 = nn.MSELoss(reduction='sum')(target_audio, x_predict)
    recon_loss2 = nn.HuberLoss(reduction='sum')(target_audio, x_predict)
    stft_loss = MRSTFTLoss()(target_audio, x_predict) * 100
    recon_loss = torch.mean(recon_loss1 + recon_loss2, dim=0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    real_labels = torch.ones(x_predict.size(0), 1).to(DEVICE)
    adv_loss = nn.BCELoss()(discriminator(x_predict), real_labels)
    
    # total_loss = recon_loss + stft_loss + 0.2 * kl_loss - adv_loss
    total_loss = recon_loss + stft_loss + kl_loss - adv_loss * 2
    return total_loss, recon_loss, stft_loss, kl_loss, adv_loss

def valid_model_loss(target_audio, x_predict):
    recon_loss1 = nn.MSELoss(reduction='mean')(target_audio, x_predict)
    recon_loss2 = nn.HuberLoss(reduction='mean')(target_audio, x_predict)
    stft_loss = MRSTFTLoss()(target_audio, x_predict) 
    
    return recon_loss1, recon_loss2, stft_loss


def weights_init(model):
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(model.weight)

def lr_lambda(epoch):
        return 0.5 ** (epoch // 500)


def train_fold(train_loader, val_loader, model, discriminator, optimizer_model, optimizer_disc, save_model_path, fold, with_validation=True):
    model.train()
    discriminator.train()

    scheduler_model = LambdaLR(optimizer_model, lr_lambda=lr_lambda)
    scheduler_discriminator = LambdaLR(optimizer_disc, lr_lambda=lr_lambda)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_recon_loss = 0
        total_stft_loss = 0
        total_kl_loss = 0
        total_adv_loss = 0
        total_disc_loss = 0
        
        for batch in train_loader:
            audio = batch['audio'].float().to(DEVICE).unsqueeze(1)
            
            # Train VAE
            optimizer_model.zero_grad()
            output, mu, logvar, z = model(audio)
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("NaN or Inf detected!")
            loss, recon_loss, stft_loss, kl_loss, adv_loss = model_loss(audio, output, mu, logvar, discriminator)
            loss.backward()
            optimizer_model.step()
            
            # Train Discriminator
            optimizer_disc.zero_grad()
            
            real_labels = torch.ones(audio.size(0), 1).to(DEVICE)
            fake_labels = torch.zeros(audio.size(0), 1).to(DEVICE)
            
            real_loss = nn.BCELoss()(discriminator(audio), real_labels) * 10
            fake_loss = nn.BCELoss()(discriminator(output.detach()), fake_labels) * 10
            disc_loss = (real_loss + fake_loss) / 2
            
            disc_loss.backward()
            optimizer_disc.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_stft_loss += stft_loss.item()
            total_kl_loss += kl_loss.item()
            total_adv_loss += adv_loss.item()
            total_disc_loss += disc_loss.item()

            #print(f"Fold {fold}, Epoch {epoch}: Adv Loss={adv_loss.item():.4f}, Disc Loss={disc_loss.item():.4f}, Val Loss={v_loss_1 / len(val_loader):.4f}")

        scheduler_model.step()
        scheduler_discriminator.step()

        if with_validation:
            # Validation phase
            model.eval()
            val_recon_1_loss = 0
            val_recon_2_loss = 0
            val_stft_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    audio = batch['audio'].float().to(DEVICE).unsqueeze(1)
                    output, mu, logvar, _ = model(audio)
                    v_loss_1, v_loss2, v_stft_loss = valid_model_loss(audio, output)
                    val_recon_1_loss += v_loss_1.item()
                    val_recon_2_loss += v_loss2.item()
                    val_stft_loss += v_stft_loss.item()
                if LOGS:
                    wandb.log({
                        "val/v_loss_1": val_recon_1_loss / len(val_loader),
                        "val/v_loss2": val_recon_2_loss / len(val_loader),
                        "val/stft_loss": val_stft_loss / len(val_loader)
                    })
            model.train()
        
        if LOGS:
            wandb.log({
                "train/loss": total_loss / len(train_loader),
                "train/recon_loss": total_recon_loss / len(train_loader),
                "train/stft_loss": total_stft_loss / len(train_loader),
                "train/kl_loss": total_kl_loss / len(train_loader),
                "train/adversarial_loss": total_adv_loss / len(train_loader),
                "train/disc_loss": total_disc_loss / len(train_loader),
            })
            if epoch % 250 == 0:
                plot_spectrograms_to_wandb(output[0], audio[0], sr=SR)
                load_audio_to_wandb(audio[0], output[0], sr=SR)
        
    if save_model_path is not None:            
        torch.save(model.state_dict(), save_model_path)




def extract_latents(dataloader, model, label_to_index, index_to_label, save_path):
    model.eval()
    all_data = []

    for batch in dataloader:
        audio = batch["audio"].float().to(DEVICE).unsqueeze(1)
        target_class = torch.tensor([label_to_index[label] for label in batch["label"]], dtype=torch.long).to(DEVICE)
        target_gain = batch["g"].clone().detach().to(torch.long).to(DEVICE)
        target_tone = batch["t"].clone().detach().to(torch.long).to(DEVICE)

        _, _, _, z = model(audio)

        for i in range(z.shape[0]):
            all_data.append({
                "label": target_class[i].item(),
                "gain": target_gain[i].item(),
                "tone": target_tone[i].item(),
                "latents": z[i].cpu().detach().numpy().tolist()
            })

    df = pd.DataFrame(all_data)
    df['label'] = df['label'].map(index_to_label)
    df.to_csv(save_path, index=False)
    print("Latents saved")


def filter_dataframe(dataframe, pedals):
    return dataframe[dataframe.pedal_name.isin(pedals)]



if __name__ == "__main__":
    assert os.path.exists(DATAFRAME_PATH), f"Dataframe not found at {DATAFRAME_PATH}"
    dataframe = pd.read_csv(DATAFRAME_PATH)
    dataframe = filter_dataframe(dataframe, PEDALS)
    dataset = PedalsDataset_VAE(dataframe, sr=SR, mode="sweep1", offset=20000, length=88200)
    labels = dataset.get_unique_labels() 
    unique_labels = len(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    
    kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    full_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
   

    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, dataset.get_labels_list())):
        print(f"Training fold {fold+1}/{FOLDS}")
        if LOGS:
            wandb.init(project=WANDB_PROJECT_NAME, name=f"{start_time}_fold_{fold}", reinit=True, entity=WANDB_ENTITY) 
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


        '''
        save_latents_path = f'{THIS_FOLDER_PATH}/latents_VAE_fold{fold}.csv'
        save_model_path = f'{THIS_FOLDER_PATH}/model_VAE_fold{fold}.pth'
        pca_csv_path = f'{THIS_FOLDER_PATH}/pca_latents_fold{fold}.csv'
        tsne_csv_path = f'{THIS_FOLDER_PATH}/tsne_latents_fold{fold}.csv'
        pca_image_path = f'{THIS_FOLDER_PATH}/pca_latents_fold{fold}.png'
        tsne_image_path = f'{THIS_FOLDER_PATH}/tsne_latents_fold{fold}.png'
        '''
        
        model = Pedals_VAE(pedal_latent_dim=N_LATENTS).to(DEVICE)
        model.apply(weights_init)
        discriminator = Discriminator().to(DEVICE)
        
        optimizer_model = Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer_disc = Adam(discriminator.parameters(), lr=LEARNING_RATE * 0.5)
        
        train_fold(train_loader, val_loader, model, discriminator, optimizer_model, optimizer_disc, save_model_path=None, fold=fold)
        
        #extract_latents(full_dataloader, model, label_to_index, save_latents_path)
        #pca_on_latents(index_to_label, save_latents_path, pca_csv_path, pca_image_path)
        #tsne_on_latents(index_to_label, save_latents_path, tsne_csv_path, tsne_image_path)

    model = Pedals_VAE(pedal_latent_dim=N_LATENTS).to(DEVICE)
    model.apply(weights_init)
    discriminator = Discriminator().to(DEVICE)
    
    optimizer_model = Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_disc = Adam(discriminator.parameters(), lr=LEARNING_RATE * 0.5)

    save_latents_path = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_latents_VAE.csv'
    save_model_path = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_model_VAE.pth'
    pca_csv_path = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_pca_latents.csv'
    tsne_csv_path = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_tsne_latents.csv'
    pca_image_path = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_pca_latents.png'
    tsne_image_path = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_tsne_latents.png'
    pca_csv_path_3d = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_pca_latents_3d.csv'
    tsne_csv_path_3d = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_tsne_latents_3d.csv'
    pca_image_path_3d = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_3D_pca_latents.png'
    tsne_image_path_3d = f'{THIS_FOLDER_PATH}/{len(PEDALS)}-{start_time}_3D_tsne_latents.png'



    print("Training on full dataset")
    if LOGS:
        wandb.init(project=WANDB_PROJECT_NAME, name=f"{start_time}_FINAL", reinit=True, entity=WANDB_ENTITY) 

    train_fold(full_dataloader, None, model, discriminator, optimizer_model, optimizer_disc, save_model_path, fold=0, with_validation=False)
    extract_latents(full_dataloader, model, label_to_index, index_to_label, save_latents_path)
    
    pca_on_latents(save_latents_path, pca_csv_path, pca_image_path)
    tsne_on_latents(save_latents_path, tsne_csv_path, tsne_image_path)
    pca_on_latents(save_latents_path, pca_csv_path_3d, pca_image_path_3d, mode="3D")
    tsne_on_latents(save_latents_path, tsne_csv_path_3d, tsne_image_path_3d, mode="3D")
    
    if LOGS:
        wandb.finish()
