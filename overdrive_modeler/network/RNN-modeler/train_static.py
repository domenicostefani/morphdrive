import wandb
import time
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

import warnings
warnings.filterwarnings("ignore")


from dataset import Pedals_Dataset_RNN
from static_rnn import StaticHyperGRU



HIDDEN_INTIAL = None
SR = 48000
WINDOW_LENGTH = 2048
BS = 64
LR = 1e-3
EPOCHS = 2

FULL_DATAFRAME = "/home/ardan/ARDAN/PEDALINY/pedaliny_dataframe_marzo.csv"
REDUCTION_DATAFRAME = "/home/ardan/ARDAN/dafx25-ArdanDomPedaliny/overdrive_modeler/network/VAE/tsne_latents_dataframe.csv"
UNPROCESSED_FILE_PATH = "/home/ardan/ARDAN/PEDALINY/pedaliny_full_rec.wav"


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.init(project="pedaliny", entity="francesco-dalri-2")


def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)



def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def complex_to_magnitude(stft_complex):
    """Utils functions for converting complex spectrogram to magnitude."""
    stft_mag = torch.sum(stft_complex ** 2) ** 0.5
    return stft_mag


class DC_PreEmph(torch.nn.Module):
    """
    Pre-emphasis filter from GreyBoxDRC.
    """
    def __init__(self, R=0.995):
        super().__init__()

        t, ir = signal.dimpulse(signal.dlti([1, -1], [1, -R]), n=2000)
        ir = ir[0][:, 0]

        self.zPad = len(ir) - 1
        self.pars = torch.flipud(torch.tensor(ir, requires_grad=False, dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
        
    def forward(self, output: torch.tensor, target: torch.tensor):
        output = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), output), dim=2)
        target = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), target), dim=2)

        output = torch.nn.functional.conv1d(output, self.pars.type_as(output), bias=None)
        target = torch.nn.functional.conv1d(target, self.pars.type_as(output), bias=None)

        return output, target


class STFTLoss(nn.Module):
    def __init__(self, win_len: int = 2048, overlap: float = 0.75, is_complex: bool = True, pre_emp: bool = False):
        super().__init__()

        print('> [Loss] --- STFT Complex Loss ---')
        print('> [Loss] is complex:', is_complex)
        print('> [Loss] win_len: {}, overlap: {}'.format(win_len, overlap))

        self.win_len = win_len
        self.is_complex = is_complex
        self.hop_len = int((1 - overlap) * win_len)

        if pre_emp:
            self.pre_emp_filter = DC_PreEmph()
        else:
            self.pre_emp_filter = None

        self.window = nn.Parameter(
            torch.from_numpy(np.hanning(win_len)).float(), requires_grad=False
        )

    def forward(self, predict: torch.tensor, target: torch.tensor):
        if self.pre_emp_filter:
            predict, target = self.pre_emp_filter(predict, target)

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
        scales: list = [1024, 512, 128, 32],#[2048, 512, 128, 32],
        overlap: float = 0.75, 
        pre_emp: bool = False
    ):
        super().__init__()
        print('> [Loss] --- Multi-resolution STFT Loss ---')
    
        self.scales = scales
        self.overlap = overlap
        self.num_scales = len(self.scales)
        self.pre_emp = pre_emp

        self.windows = nn.ParameterList(
            nn.Parameter(torch.from_numpy(np.hanning(scale)).float(), requires_grad=False) for scale in self.scales
        )
        if self.pre_emp:
            self.pre_emp_filter = DC_PreEmph()
        else:
            self.pre_emp_filter = None

    def forward(
        self,
        predict: torch.tensor, 
        target: torch.tensor
    ):
        if self.pre_emp_filter:
            predict, target = self.pre_emp_filter(predict, target)

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
    
    

def MSE_Loss(predict: torch.tensor, target: torch.tensor):
    return F.mse_loss(predict, target)


def symmetric_mse_loss(pred, target):
    loss = (pred - target) ** 2
    return loss.mean() + ((pred.mean()) ** 2) 


class GlobalLoss(nn.Module):
    def __init__(self, stft_weight: float = 0.1, mse_weight: float = 100.0):
        """
        Global Loss combining MRSTFT and MSE.
        :param stft_weight: Weight for the MRSTFT loss.
        :param mse_weight: Weight for the MSE loss.
        """
        super().__init__()
        self.stft_weight = stft_weight
        self.mse_weight = mse_weight
        
        self.stft_loss = MRSTFTLoss(
            scales=[1024, 512, 128, 32],#[2048, 512, 128, 32],
            overlap=0.75,
            pre_emp=True
        ).to(DEVICE)

    def forward(self, predict: torch.tensor, target: torch.tensor):
        """
        Compute the combined loss.
        :param predict: Predicted waveform.
        :param target: Target waveform.
        :return: Combined loss value.
        """
        stft_loss_val = self.stft_loss(predict, target)
        mse_loss_val = F.mse_loss(predict, target)

        # Combine the losses with their respective weights
        stft_loss = self.stft_weight * stft_loss_val
        mse_loss = self.mse_weight * mse_loss_val
        
        return stft_loss, mse_loss






def train(model, loss_func, optimizer, train_dataloader, num_epochs=100):
    wandb.init(project="pedaliny_training", name="train_dynamic_hypergru")
    
    model.train()
    best_loss = float('inf')
    #max_grad_norm = 0.8

    print('{:=^40}'.format(' Start Training '))

    for epoch in range(num_epochs):
        h = HIDDEN_INTIAL
        total_loss = 0

        for i, batch in enumerate(train_dataloader):
            wav_x, wav_y, vec_c = batch
            wav_x, wav_y = wav_x.float().to(DEVICE), wav_y.float().to(DEVICE)
            vec_c = vec_c.float().to(DEVICE) if vec_c is not None else None

            h = None
            h = torch.zeros(1, wav_x.shape[0], model.rnn_size).to(wav_x.device)
            #hyper_h = torch.zeros(1, wav_x.shape[0], model.hyper_rnn_size).to(wav_x.device)

            start_time = time.time()
            wav_y_pred, h, _ = model(wav_x, vec_c, h)
            elapsed_time = time.time() - start_time
            print(f"Processing time: {elapsed_time * 1000:.2f} ms")

            stft_loss, mse_loss = loss_func(wav_y_pred, wav_y) 

            loss = stft_loss + mse_loss

            optimizer.zero_grad()
            loss.backward()
            # Uncomment if gradient clipping is needed
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()

            # Log loss and sample outputs to wandb
            if i % 5 == 0:  # Log every 10 batches
                wandb.log({
                    "batch_loss": loss.item(),
                    "stft_loss": stft_loss.item(),
                    "mse_loss": mse_loss.item(),
                    "wav_x": wandb.Audio(convert_tensor_to_numpy(wav_x[0]), sample_rate=SR),
                    "wav_y": wandb.Audio(convert_tensor_to_numpy(wav_y[0]), sample_rate=SR),
                    "wav_y_pred": wandb.Audio(convert_tensor_to_numpy(wav_y_pred[0]), sample_rate=SR),
            })
                
            torch.save(model.state_dict(), "/home/ardan/ARDAN/PEDALINY/pedaliny_static_rnn_mini.pth")

            print(f"Epoch {epoch} - Batch {i}, Loss: {loss.item():.5f}, STFT Loss: {stft_loss.item():.5f}, MSE Loss: {mse_loss.item():.5f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")

        # Log epoch loss to wandb
        wandb.log({"epoch_loss": avg_loss})
        if avg_loss < best_loss:
            torch.save(model.state_dict(), "/home/ardan/ARDAN/PEDALINY/pedaliny_static_rnn_mini.pth")
            best_loss = avg_loss

    # Save model
    #torch.save(model.state_dict(), "/home/ardan/ARDAN/PEDALINY/pedaliny_rnn_8.pth")
    


def test(model, loss_func, test_dataloader):
    print('{:=^40}'.format(' Start Testing '))
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            wav_x, wav_y, vec_c = batch
            wav_x, wav_y = wav_x.float().to(DEVICE), wav_y.float().to(DEVICE)
            vec_c = vec_c.float().to(DEVICE) if vec_c is not None else None

            h = torch.zeros(1, wav_x.shape[0], model.rnn_size).to(wav_x.device)
            #hyper_h = torch.zeros(1, wav_x.shape[0], model.hyper_rnn_size).to(wav_x.device)

            wav_y_pred, _, _ = model(wav_x, vec_c, h)
            stft_loss, mse_loss = loss_func(wav_y_pred, wav_y)

            loss = stft_loss + mse_loss
            test_loss += loss.item()

            if i % 20 == 0:  # Log every 10 batches
                wandb.log({
                    "test_batch_loss": loss.item(),
                    "test_stft_loss": stft_loss.item(),
                    "test_mse_loss": mse_loss.item(),
                })

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.5f}")
    wandb.log({"test_loss": avg_test_loss})



if __name__ == '__main__':
    dataset = Pedals_Dataset_RNN(FULL_DATAFRAME, REDUCTION_DATAFRAME, UNPROCESSED_FILE_PATH, win_len=WINDOW_LENGTH)

    # Split into train and test (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)

    model = StaticHyperGRU(inp_channel =  1,
                            out_channel = 1,
                            rnn_size =2,
                            sample_rate = SR,
                            n_mlp_blocks = 2,
                            mlp_size = 32,
                            num_conds = 8,
                            ).to(DEVICE)


    # Initialize weights
    def init_weights(model):
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

    init_weights(model)

    loss_func = GlobalLoss(stft_weight=1, mse_weight=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

    # Train the model
    '''train(
        model, loss_func, optimizer, train_dataloader=train_dataloader,
        num_epochs=EPOCHS,
    )'''

    # Test the model
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)
    #test(model, loss_func, test_dataloader)

    #wandb.finish()

