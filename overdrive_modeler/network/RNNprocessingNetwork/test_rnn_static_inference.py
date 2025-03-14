import threading
import numpy as np
import torch
import torch.nn as nn
import argparse
import soundfile as sf
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory
MLP_MODEL_PATH = '../MLP/mlp_model.py'
assert os.path.exists(MLP_MODEL_PATH), "MLP model not found at %s" % os.path.abspath(MLP_MODEL_PATH)
# symlink here
if not os.path.exists('mlp_model.py'):
    os.symlink(MLP_MODEL_PATH, 'mlp_model.py')
from mlp_model import Pedals_MLP
from rnn_static_model import StaticHyperGRU



argparser = argparse.ArgumentParser()
# Mandatory path to rnn model
argparser.add_argument("--rnn_model", type=str, required=True, help="Path to the traced RNN model")
argparser.add_argument("--reverse_tsne", type=str, required=True, help="Path to the reverse t-SNE model")
argparser.add_argument("--tsne_df", type=str, required=True, help="Path to the TSNE dataframe file")

args = argparser.parse_args()

SR = 48000
CHUNK_SIZE = 1024
TRACED_RNN = args.rnn_model
REVERSE_TSNE_MODELPATH = args.reverse_tsne
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATAFRAME_PATH = args.tsne_df

assert os.path.exists(TRACED_RNN),             "Traced RNN model not found at %s" % os.path.abspath(TRACED_RNN)
assert os.path.exists(REVERSE_TSNE_MODELPATH), "PCA to latent model not found at %s" % os.path.abspath(REVERSE_TSNE_MODELPATH)
assert os.path.exists(DATAFRAME_PATH),         "Datagram file not found at %s" % os.path.abspath(DATAFRAME_PATH)



reverseTsneModel = Pedals_MLP().to(device)
reverseTsneModel.load_state_dict(torch.load(REVERSE_TSNE_MODELPATH, map_location=device, weights_only=True))
reverseTsneModel.eval()

rnn_model = StaticHyperGRU(
    inp_channel =  1,
    out_channel = 1,
    rnn_size = 32, # 16
    sample_rate = SR,
    n_mlp_blocks = 2, # 4
    mlp_size = 16, # 32
    num_conds = 8,
    ).to(device)

# rnn_model_traced = torch.jit.load(TRACED_RNN)
rnn_model.load_state_dict(torch.load(TRACED_RNN, map_location=device, weights_only=True))
rnn_model.eval()

vec_c = torch.zeros(1, 8, dtype=torch.float32).to(device)
vec_lock = threading.Lock()
h = torch.zeros(1, 1, rnn_model.rnn_size).to(device)

overlap_buffer = np.zeros(CHUNK_SIZE)

# OSC Handler
def handle_coordinates(address, *args):
    global vec_c
    x, y = args
    input_tensor = torch.tensor([[x, y]], dtype=torch.float32).to(device)

    with torch.no_grad():
        output = reverseTsneModel(input_tensor)
    with vec_lock:
        vec_c.copy_(output)  
    #print(f"COORD: ({x}, {y}) -> COND: {output.cpu().numpy()}")

# Audio Callback
# def audio_callback(in_data):
#     global h, overlap_buffer, vec_c


#     wav_x = np.frombuffer(in_data, dtype=np.float32)


#     combined_input = np.concatenate((overlap_buffer, wav_x))
#     wav_x_windowed = combined_input[:CHUNK_SIZE]
#     overlap_buffer[:] = combined_input[CHUNK_SIZE:]

#     wav_x_tensor = torch.tensor(wav_x_windowed, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)


#     with vec_lock:
#         current_vec_c = vec_c.clone()

#     with torch.no_grad():
#         wav_y_pred, _, _ = rnn_model(wav_x_tensor, current_vec_c, h)


#     wav_y_pred = wav_y_pred.squeeze().cpu().numpy()
#     out_data = wav_y_pred.astype(np.float32).tobytes()

def gen_sine(freq, duration, sr):
    return np.sin(2 * np.pi * freq * np.linspace(0, duration, sr * duration)).astype(np.float32)

def main():
    
    # generate a 2 seconds sine wave at SINE_FREQ Hz, at -6 dB
    SINE_FREQ = 440
    SINE_AMPLITUDE_DB = -6
    SINE_DURATION_SEC = 2
    SINE_AMPLITUDE = 10**(SINE_AMPLITUDE_DB/20)
    sine_wave = gen_sine(SINE_FREQ, SINE_DURATION_SEC, SR) * SINE_AMPLITUDE

    X = 0.1
    Y = 0.1



    # Process coordinates
    x = X
    y = Y
    input_tensor = torch.tensor([[x, y]], dtype=torch.float32).to(device)

    with torch.no_grad():
        output = reverseTsneModel(input_tensor)
    with vec_lock:
        vec_c.copy_(output)  

    # run the sine wave through the model

    wav_x_tensor = torch.tensor(sine_wave, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
    with vec_lock:
        current_vec_c = vec_c.clone()

    with torch.no_grad():
        wav_y_pred, _, _ = rnn_model(wav_x_tensor, current_vec_c, h)

    wav_y_pred = wav_y_pred.squeeze().cpu().numpy()

    # Save both
    sf.write('sine_input.wav', sine_wave, SR)
    sf.write('sine_output.wav', wav_y_pred, SR)



if __name__ == "__main__":
    main()
