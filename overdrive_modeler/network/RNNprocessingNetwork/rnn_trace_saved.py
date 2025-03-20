import torch
from rnn_static_model import StaticHyperGRU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

SR = 48000
CHUNK_SIZE = 64

rnn_model_path = '4-PAPER_2025-03-05_static_small.pth'
rnn_model = StaticHyperGRU(
    inp_channel=1,
    out_channel=1,
    rnn_size=4,
    sample_rate=SR,
    n_mlp_blocks=3,
    mlp_size=16,
    num_conds=8,
).to(device)




assert os.path.exists(rnn_model_path), f"Model file not found: {rnn_model_path}"
assert os.path.splitext(rnn_model_path)[1] == ".pth", "Model file must be a .pth file"
example_input = (
    torch.zeros(1, 1, CHUNK_SIZE, dtype=torch.float32).to(device),  # Input tensor
    torch.zeros(1, 8, dtype=torch.float32).to(device),              # Conditioning vector
    torch.zeros(1, 1, rnn_model.rnn_size, dtype=torch.float32).to(device),           # Hidden state
)
rnn_model.load_state_dict(torch.load(rnn_model_path, map_location=device))
rnn_model.eval()

rnn_model_traced = torch.jit.trace(rnn_model, example_input)
newname = 'traced'+os.path.basename(rnn_model_path)
torch.jit.save(rnn_model_traced, newname)