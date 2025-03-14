import threading
import numpy as np
import torch
import torch.nn as nn
import argparse
import soundfile as sf
import os
from . import rnn_static_model
from .rnn_static_model import StaticHyperGRU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNStatic_Inferencer:
    def __init__(self, rnn_model_path, samplerate=48000):
        self.rnn_model = StaticHyperGRU(
            inp_channel =  1,
            out_channel = 1,
            rnn_size = 32, # 16
            sample_rate = samplerate,
            n_mlp_blocks = 2, # 4
            mlp_size = 16, # 32
            num_conds = 8,
            ).to(device)
                
        self.rnn_model.load_state_dict(torch.load(rnn_model_path, map_location=device, weights_only=True))
        self.rnn_model.eval()

    def inference_memoryless(self, input_audio, conditioniong_vector):
        h = torch.zeros(1, 1, self.rnn_model.rnn_size).to(device)

        wav_x_tensor = torch.tensor(input_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        conditioning_tensor = torch.tensor(conditioniong_vector, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            wav_y_pred, _, _ = self.rnn_model(wav_x_tensor, conditioning_tensor, h)

        wav_y_pred = wav_y_pred.squeeze().cpu().numpy()
        return wav_y_pred


def gen_sine(freq, duration, sr):
    return np.sin(2 * np.pi * freq * np.linspace(0, duration, sr * duration)).astype(np.float32)

if __name__ == "__main__":

    RNN_MODEL_PATH = '4-2025-03-05_static_rnn.pth'
    SR = 48000

    os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory
    assert os.path.exists(RNN_MODEL_PATH), "RNN model not found at %s" % os.path.abspath(RNN_MODEL_PATH)

    # generate a 2 seconds sine wave at SINE_FREQ Hz, at -6 dB
    SINE_FREQ = 440
    SINE_AMPLITUDE_DB = -6
    SINE_DURATION_SEC = 2
    SINE_AMPLITUDE = 10**(SINE_AMPLITUDE_DB/20)
    sine_wave = gen_sine(SINE_FREQ, SINE_DURATION_SEC, SR) * SINE_AMPLITUDE

    
    random_conditioning = np.random.rand(8).astype(np.float32)

    inferencer = RNNStatic_Inferencer(RNN_MODEL_PATH, SR)

    wav_y_pred = inferencer.inference_memoryless(sine_wave, random_conditioning)

    # Save both
    sf.write('sine_input.wav', sine_wave, SR)
    sf.write('sine_output.wav', wav_y_pred, SR)
    print("Sine wave input and model output saved as ./sine_input.wav and ./sine_output.wav")