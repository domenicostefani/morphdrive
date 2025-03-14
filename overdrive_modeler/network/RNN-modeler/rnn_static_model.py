import torch
import torch.nn as nn
import math
from einops import rearrange


class NonlearnableCell(nn.Module):
    def __init__(self, gru_gates):
        super().__init__()

        self.norm_i = nn.Identity()
        self.norm_h = nn.Identity()
        self.gru_gates = gru_gates

    def forward_cell(self, x, h, w_ih, w_hh, b_ih=None, b_hh=None):
        hx = h

        ih = torch.bmm(x, w_ih)
        hh = torch.bmm(hx, w_hh)
        if b_ih is not None and b_hh is not None:
            ih = ih + b_ih
            hh = hh + b_hh

        ih = self.norm_i(ih)
        hh = self.norm_h(hh)

        return self.forward_gru_cell(hx, ih, hh)

    def forward_gru_cell(self, hx, ih, hh):
        i_r, i_i, i_n = ih.chunk(self.gru_gates, -1)
        h_r, h_i, h_n = hh.chunk(self.gru_gates, -1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hx - newgate)

        return hy, hy


    def forward(self, xs, state, w_ih, w_hh, b_ih, b_hh):
        outputs = []

        state = rearrange(state, '1 b h -> b 1 h')

        for i in range(xs.shape[1]):
            out, state = self.forward_cell(
                xs[:, i:i + 1, ...],
                state,
                w_ih,
                w_hh,
                b_ih,
                b_hh
            )
            outputs.append(out)
        return torch.cat(outputs, dim=1), state


class StaticHyperGRU(nn.Module):
    def __init__(
        self,
        inp_channel: int = 1,
        out_channel: int = 1,
        rnn_size: int = 4,
        sample_rate: int = 48000,
        n_mlp_blocks: int = 3,
        mlp_size: int = 8,
        num_conds: int = 8,
        rnn_bias: bool = True
    ):
        super().__init__()

        if num_conds <= 0:
            raise ValueError(f'Static hyper networks are used only when condition is provided')

        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.rnn_size = rnn_size
        self.sample_rate = sample_rate
        self.n_mlp_blocks = n_mlp_blocks
        self.mlp_size = mlp_size
        self.num_conds = num_conds
        self.rnn_bias = rnn_bias

        self.gru_gates = 3
        self.main_rnn = NonlearnableCell(self.gru_gates)

        self.linear_out = nn.Linear(self.rnn_size, self.out_channel, bias=self.rnn_bias)

        # MLP
        self.cond_mlp = nn.ModuleList()
        for n in range(self.n_mlp_blocks - 1):
            _input_features = self.num_conds if n == 0 else self.mlp_size
            self.cond_mlp.append(nn.Sequential(
                nn.Linear(_input_features, self.mlp_size, bias=True),
                nn.LeakyReLU(0.1)
            ))
        self.cond_mlp = nn.Sequential(*self.cond_mlp)

        for idx, w in enumerate(self.cond_mlp.parameters()):
            if idx == 0:
                bound = math.sqrt(self.gru_gates / (self.mlp_size * self.num_conds))
            else:
                bound = math.sqrt(self.gru_gates / (self.mlp_size * self.mlp_size))
            w.data.uniform_(-bound, bound)

        _proj_ih_out = self.gru_gates * self.inp_channel * self.rnn_size
        _proj_hh_out = self.gru_gates * self.rnn_size * self.rnn_size
        _bias_out = self.gru_gates * self.rnn_size

        self.proj_ih = nn.Linear(self.mlp_size, _proj_ih_out, bias=True)
        self.proj_hh = nn.Linear(self.mlp_size, _proj_hh_out, bias=True)

        if self.rnn_bias:
            self.proj_bih = nn.Linear(self.mlp_size, _bias_out, bias=True)
            self.proj_bhh = nn.Linear(self.mlp_size, _bias_out, bias=True)


    def compute_receptive_field(self): 
        return 1, (1/self.sample_rate) * 1000
    

    def compute_num_of_params(self):
        return (sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad))
    

    def forward(self, x, c, h0):
        w = self.cond_mlp(c)
        w_ih = self.proj_ih(w).reshape(-1, self.inp_channel, self.rnn_size * self.gru_gates)
        w_hh = self.proj_hh(w).reshape(-1, self.rnn_size, self.rnn_size * self.gru_gates)
        b_ih, b_hh = None, None

        if self.rnn_bias:
            b_ih = self.proj_bih(w).reshape(-1, 1, self.rnn_size * self.gru_gates)
            b_hh = self.proj_bhh(w).reshape(-1, 1, self.rnn_size * self.gru_gates)

        x = x.permute(0, 2, 1)

        x, h_out = self.main_rnn(x, h0, w_ih, w_hh, b_ih, b_hh)

        x = self.linear_out(x)

        x = x.permute(0, 2, 1)

        return x, h_out.detach(), (w_ih.detach(), w_hh.detach(), b_ih.detach(), b_hh.detach())



if __name__ == "__main__":

    model = StaticHyperGRU(
        inp_channel=1,
        out_channel=1,
        rnn_size=32,
        sample_rate=48000,
        n_mlp_blocks=3,
        mlp_size=16,
        num_conds=8,
    )

    _rec_in_samples, _ = model.compute_receptive_field()

    x = torch.randn(32, 1, 1024 + _rec_in_samples - 1)
    c = torch.randn(32, 8)
    h = torch.ones(1, 32, 32)

    out, _, _ = model(x, c, h)

    print(f'Out shape: {out.shape}')
    print(f'Number of parameters: {model.compute_num_of_params()}')
