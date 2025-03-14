import threading
import numpy as np
import torch
import torch.nn as nn
import argparse
import soundfile as sf
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from .RNNprocessingNetwork.rnn_static_inference import RNNStatic_Inferencer
from .MLP.mlp_inference import MLP_Inferencer



class HyperdriveStaticRNNinference:
    def __init__(self, rnn_model_path, reverse_tsne_model_path, tsne_dataframe_path, samplerate=48000):
        self.rnn_inferencer = RNNStatic_Inferencer(rnn_model_path,samplerate)
        self.mlp_inferencer = MLP_Inferencer(reverse_tsne_model_path)
        self.tsne_dataframe_path = tsne_dataframe_path

    def inference_memoryless(self, input_audio, x, y):
        # Process coordinates
        input_tensor = torch.tensor([x, y], dtype=torch.float32).to(device)
        conditioning_vector = self.mlp_inferencer.inference(input_tensor)

        return self.rnn_inferencer.inference_memoryless(input_audio, conditioning_vector)

def gen_sine(freq, duration, sr):
    return np.sin(2 * np.pi * freq * np.linspace(0, duration, sr * duration)).astype(np.float32)

if __name__ == "__main__":
    MLP_MODEL_PATH = './MLP/4-2025-03-05_mlp.pth'
    RNN_MODEL_PATH = './RNNprocessingNetwork/4-2025-03-05_static_rnn.pth'
    SR = 48000

    assert os.path.exists(MLP_MODEL_PATH), "MLP model not found at %s" % os.path.abspath(MLP_MODEL_PATH)
    assert os.path.exists(RNN_MODEL_PATH), "RNN model not found at %s" % os.path.abspath(RNN_MODEL_PATH)

    # generate a 2 seconds sine wave at SINE_FREQ Hz, at -6 dB
    SINE_FREQ = 440
    SINE_AMPLITUDE_DB = -6
    SINE_DURATION_SEC = 2
    SINE_AMPLITUDE = 10**(SINE_AMPLITUDE_DB/20)
    sine_wave = gen_sine(SINE_FREQ, SINE_DURATION_SEC, SR) * SINE_AMPLITUDE

    random_xy = np.random.rand(2).astype(np.float32)

    inferencer = HyperdriveStaticRNNinference(RNN_MODEL_PATH, MLP_MODEL_PATH, 'tsne_dataframe.csv', SR)

    wav_y_pred = inferencer.inference_memoryless(sine_wave, random_xy[0], random_xy[1])

    # Save both
    sf.write('sine_input.wav', sine_wave, SR)
    sf.write('sine_output.wav', wav_y_pred, SR)
    print("Sine wave input and model output saved as ./sine_input.wav and ./sine_output.wav")
