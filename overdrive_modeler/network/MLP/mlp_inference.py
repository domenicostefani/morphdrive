import numpy as np
import torch
import torch.nn as nn
import os
from .mlp_model import Pedals_MLP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class MLP_Inferencer:
    def __init__(self, mlp_model_path):
        assert os.path.exists(mlp_model_path), "MLP model not found at %s" % os.path.abspath(mlp_model_path)
        self.mlp_model = Pedals_MLP(2, 8).to(device)
        self.mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device, weights_only=True))
        self.mlp_model.eval()
                

    def inference(self, input_vector):
        assert len(input_vector) == 2, "Input vector must have 2 elements (x, y) but has %d" % len(input_vector)

        if type(input_vector) is torch.Tensor:
            input_tensor = input_vector.clone().detach().to(device)
        else:
            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = self.mlp_model(input_tensor)
        return output_tensor.squeeze().cpu().numpy()

if __name__ == '__main__':
    MLP_MODEL_PATH = '4-2025-03-05_mlp.pth'
    assert os.path.exists(MLP_MODEL_PATH), "MLP model not found at %s" % os.path.abspath(MLP_MODEL_PATH)

    random_in = np.random.rand(2).astype(np.float32)

    inferencer = MLP_Inferencer(MLP_MODEL_PATH)

    mlp_output = inferencer.inference(random_in)

    print("Input vector:", random_in,   '\tSize:',len(random_in))
    print("Output vector:", mlp_output, '\tSize:',len(mlp_output))