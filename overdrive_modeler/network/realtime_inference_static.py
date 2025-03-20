import threading
import numpy as np
import torch
import torch.nn as nn
import pyaudio
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from RNNprocessingNetwork.rnn_static_model import StaticHyperGRU
from MLP.mlp_inference import MLP_Inferencer
from sendDataframe2Processing import DFSender

import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change working directory to the script directory

SR = 48000
CHUNK_SIZE = 1024
TRACED_RNN = "./RNNprocessingNetwork/traced4-PAPER_2025-03-05_static_small.pth"
DATAFRAME_PATH = "./VAE/4-PAPER_2025-03-05_23-11_tsne_latents.csv" # Declare dataframe path
reverse_tsne_model_path = "./MLP/4-PAPER_mlp.pth"


assert os.path.exists(TRACED_RNN), "Traced RNN model not found at %s" % os.path.abspath(TRACED_RNN)
assert os.path.exists(DATAFRAME_PATH), "Datagram file not found at %s" % os.path.abspath(DATAFRAME_PATH)

## Read and send points on startup, later send again if receiving message "/gimmeDataframe"
dataframeSender = DFSender("127.0.0.1", 12000)
dataframeSender.readDataframe(DATAFRAME_PATH)
dataframeSender.send()


rnn_model = StaticHyperGRU(
    inp_channel=1,
    out_channel=1,
    rnn_size=4,
    sample_rate=SR,
    n_mlp_blocks=3,
    mlp_size=16,
    num_conds=8,
).to(device)

rnn_model_traced = torch.jit.load(TRACED_RNN)
rnn_model_traced.eval()


vec_c = torch.zeros(1, 8, dtype=torch.float32).to(device)
vec_lock = threading.Lock()
h = torch.zeros(1, 1, rnn_model.rnn_size).to(device)

overlap_buffer = np.zeros(CHUNK_SIZE)

mlp_inferencer = MLP_Inferencer(reverse_tsne_model_path)

# OSC Handler
def handle_coordinates(address, *args):
    global vec_c
    x, y = args
    input_tensor = torch.tensor([x, y], dtype=torch.float32).to(device)
    output = mlp_inferencer.inference(input_tensor)
    with vec_lock:
        vec_c.copy_(output)  
    #print(f"COORD: ({x}, {y}) -> COND: {output.cpu().numpy()}")

# Audio Callback
def audio_callback(in_data, frame_count, time_info, status):
    global h, overlap_buffer, vec_c


    wav_x = np.frombuffer(in_data, dtype=np.float32)


    combined_input = np.concatenate((overlap_buffer, wav_x))
    wav_x_windowed = combined_input[:CHUNK_SIZE]
    overlap_buffer[:] = combined_input[CHUNK_SIZE:]

    wav_x_tensor = torch.tensor(wav_x_windowed, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)


    with vec_lock:
        current_vec_c = vec_c.clone()

    with torch.no_grad():
        wav_y_pred, _, _ = rnn_model_traced(wav_x_tensor, current_vec_c, h)


    wav_y_pred = wav_y_pred.squeeze().cpu().numpy()
    out_data = wav_y_pred.astype(np.float32).tobytes()

    return (out_data, pyaudio.paContinue)


def main():
    print("Starting OSC server and audio processing...")

    # Start OSC server
    dispatcher = Dispatcher()
    dispatcher.map("/mouse/positionScaled", handle_coordinates)
    dispatcher.map("/gimmeDataframe", lambda address, *args : dataframeSender.send())

    osc_server = BlockingOSCUDPServer(("127.0.0.1", 12345), dispatcher)
    osc_thread = threading.Thread(target=osc_server.serve_forever, daemon=True)
    osc_thread.start()
    print("OSC server is running...")

    # Start PyAudio
    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paFloat32,
        input_device_index=0,
        channels=1,
        rate=SR,
        input=True,
        output=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=audio_callback,
    )

    stream.start_stream()
    print("Processing audio in real time. Press Ctrl+C to stop.")

    try:
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        print("\nStopping...")

    # Stop 
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio processing stopped.")

if __name__ == "__main__":
    main()
