import threading
import numpy as np
import torch
import torch.nn as nn
import pyaudio
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from rnn_static_mini import StaticHyperGRU


SR = 44100
CHUNK_SIZE = 1024
TRACED_RNN = "pedaliny_static_rnn_mini_traced.pth"
PCA_NN = "pca_to_latent_model_8.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PCAtoLatentModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=8):
        super(PCAtoLatentModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        return self.fc(x)

pca_model = PCAtoLatentModel().to(device)
pca_model.load_state_dict(torch.load(PCA_NN, map_location=device, weights_only=True))
pca_model.eval()



rnn_model = StaticHyperGRU(
    inp_channel=1,
    out_channel=1,
    rnn_size=2,
    sample_rate=SR,
    n_mlp_blocks=2,
    mlp_size=32,
    num_conds=8,
).to(device)



'''
x = torch.randn(1, 1, 1024) 
c = torch.randn(1, 8) 
h0 = torch.randn(1, 1, 2)  

model.load_state_dict(torch.load("/Users/ardan/Desktop/PedalinY/pedaliny_static_rnn_mini.pth", map_location=DEVICE))
model.eval()
model_traced = torch.jit.trace(model, (x, c, h0))
model_traced.save("pedaliny_static_rnn_mini_traced.pth")'''



rnn_model_traced = torch.jit.load(TRACED_RNN)
rnn_model_traced.eval()


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
        output = pca_model(input_tensor)
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
