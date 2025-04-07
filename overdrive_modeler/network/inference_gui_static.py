import threading
import numpy as np
import torch
import torch.nn as nn
import argparse
import soundfile as sf
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from RNNprocessingNetwork.rnn_static_inference import RNNStatic_Inferencer
from MLP.mlp_inference import MLP_Inferencer



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
    return np.sin(2 * np.pi * freq * np.linspace(0, duration, int(sr * duration))).astype(np.float32)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperdrive Static RNN Inference")
    parser.add_argument("--mlp_model_path", type=str, default='./MLP/4-2025-03-05_mlp.pth', help="Path to the MLP model")
    parser.add_argument("--rnn_model_path", type=str, default='./RNNprocessingNetwork/4-2025-03-05_static.pth', help="Path to the RNN model")
    parser.add_argument("--input_audio_path", '-i', type=str, default=None, help="Path to the input audio file")
    parser.add_argument("--output_audio_path", '-o', type=str, default=None, help="Path to the output audio file")
    parser.add_argument("--xy_coordinates", '-xy', type=float, nargs=2, default=None, help="X and Y coordinates for conditioning")

    args = parser.parse_args()

    MLP_MODEL_PATH = args.mlp_model_path
    RNN_MODEL_PATH = args.rnn_model_path
    INPUT_AUDIO_PATH = args.input_audio_path
    OUTPUT_AUDIO_PATH = args.output_audio_path
    XY_COORDINATES = args.xy_coordinates
    SR = 48000

    assert os.path.exists(MLP_MODEL_PATH), "MLP model not found at %s" % os.path.abspath(MLP_MODEL_PATH)
    assert os.path.exists(RNN_MODEL_PATH), "RNN model not found at %s" % os.path.abspath(RNN_MODEL_PATH)

    if INPUT_AUDIO_PATH is None:
        # generate a 2 seconds sine wave at SINE_FREQ Hz, at -6 dB
        SINE_FREQ = 440
        SINE_AMPLITUDE_DB = -6
        SINE_DURATION_SEC = 2
        SINE_AMPLITUDE = 10**(SINE_AMPLITUDE_DB/20)
        audio_in = gen_sine(SINE_FREQ, SINE_DURATION_SEC, SR) * SINE_AMPLITUDE
    else:
        assert os.path.exists(INPUT_AUDIO_PATH), "Input audio file not found at %s" % os.path.abspath(INPUT_AUDIO_PATH)
        audio_in, SR = sf.read(INPUT_AUDIO_PATH)
        assert SR == 48000, "Input audio sample rate must be 48000 Hz"
        assert audio_in.ndim == 1, "Input audio must be mono (1D array)"
        audio_in = audio_in.astype(np.float32)
    
    if XY_COORDINATES is None:
        # Generate random x and y coordinates
        xy_coordinates = np.random.rand(2).astype(np.float32)
    else:
        assert type(XY_COORDINATES) == tuple or list, "XY coordinates must be a tuple or list"
        assert len(XY_COORDINATES) == 2, "XY coordinates must be a tuple or list of length 2"
        assert all(isinstance(i, (int, float)) for i in XY_COORDINATES), "XY coordinates must be a tuple or list of numbers"
        xy_coordinates = np.array(XY_COORDINATES).astype(np.float32)


    inferencer = HyperdriveStaticRNNinference(RNN_MODEL_PATH, MLP_MODEL_PATH, 'tsne_dataframe.csv', SR)

    audio_out = inferencer.inference_memoryless(audio_in, xy_coordinates[0], xy_coordinates[1])

    # Save
    if INPUT_AUDIO_PATH is None:
        sf.write('sine_input.wav', audio_in, SR)
        print("Input sine wave saved to sine_input.wav")

    if OUTPUT_AUDIO_PATH is not None:
        outname = OUTPUT_AUDIO_PATH
    elif INPUT_AUDIO_PATH is None:
        outname = "static_inference_output.wav"
    else:
        outname = os.path.splitext(os.path.basename(INPUT_AUDIO_PATH))[0] + "_static_inference_output.wav"

    sf.write(outname, audio_out, SR)
    print("Inference ouput saved to %s" % outname)
