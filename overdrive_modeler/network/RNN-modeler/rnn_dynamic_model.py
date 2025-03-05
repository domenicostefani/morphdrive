import torch
import torch.nn as nn
import math
from einops import rearrange


class LearnableCell(nn.Module):
    def __init__(self, input_size: int , hidden_size: int, num_layers: int, layer_norm: bool = False, bias: bool = True, filmed: bool = False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.bias = bias 
        self.filmed = filmed

        self.x2h = nn.Linear(input_size, num_layers * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, num_layers * hidden_size, bias=bias)
        
        self.norm_i, self.norm_h, self.norm_c = None, None, None
        self.norm_i = nn.LayerNorm(num_layers * hidden_size, elementwise_affine = False) if layer_norm else nn.Identity()
        self.norm_h = nn.LayerNorm(num_layers * hidden_size, elementwise_affine = False) if layer_norm else nn.Identity()
        
        self.initialize_weights()

    def forward(self, xs, state, alpha_i = None, beta_i = None, alpha_h = None, beta_h = None):
        
        outputs = []

        try:
            state  = rearrange(state, '1 b h -> b 1 h')
        except:
            state = state 
            
        for i in range(xs.shape[1]):
            out, state = self.forward_cell(
                xs[:, i:i+1, ...],
                state,
                alpha_i, beta_i, alpha_h, beta_h)
            outputs += [out]
        return torch.cat(outputs, dim=1), state
    

    def forward_cell(self, x, h, alpha_i = None, beta_i = None, alpha_h = None, beta_h = None):
        hx = h 
        return self.forward_gru_cell(x, hx, alpha_i, beta_i, alpha_h, beta_h)

        
    def forward_gru_cell(self, x, h, alpha_i = None, beta_i = None, alpha_h = None, beta_h = None):
        
        ih = self.norm_i(self.x2h(x))
        hh = self.norm_h(self.h2h(h))

        if self.filmed:
            ih = (ih * alpha_i.unsqueeze(1)) + beta_i.unsqueeze(1)
            hh = (hh * alpha_h.unsqueeze(1)) + beta_h.unsqueeze(1)  

        i_r, i_i, i_n = ih.chunk(self.num_layers, -1)
        h_r, h_i, h_n = hh.chunk(self.num_layers, -1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate   = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (h - newgate)
        
        return hy, hy


    
    def initialize_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)



class DynamicHyperCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, hyper_size: int, n_z: int, layer_norm: bool = False, bias: bool = True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size
        self.n_z = n_z
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.bias = bias 


        self.hyper_rnn = LearnableCell(
            input_size=input_size + hidden_size, 
            hidden_size=hyper_size, 
            num_layers=num_layers,
            layer_norm=layer_norm, 
            bias=bias)

        self.z_h = nn.Linear(hyper_size, num_layers * n_z)
        self.z_x = nn.Linear(hyper_size, num_layers * n_z)
        self.z_b = nn.Linear(hyper_size, num_layers * n_z, bias=False)

        d_h = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(num_layers)]
        self.d_h = nn.ModuleList(d_h)

        d_x = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(num_layers)]
        self.d_x = nn.ModuleList(d_x)

        d_b = [nn.Linear(n_z, hidden_size) for _ in range(num_layers)]
        self.d_b = nn.ModuleList(d_b)

        self.w_h = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(num_layers)])
        self.w_x = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, input_size)) for _ in range(num_layers)])


        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)


    def forward(self, xs, cond, h_c, h_c_hat):
        
        outputs = []
        
        for i in range(xs.shape[1]):
            out, h_c, h_c_hat = self.forward_cell(
                xs[:, i:i+1, ...],
                cond,
                h_c,
                h_c_hat)
            outputs += [out.permute(1, 0, 2)]
        return torch.cat(outputs, dim=1), (h_c, h_c_hat)
    
    
    def forward_cell(self, x, cond, h_c, h_c_hat):

        return self.forward_gru_cell(x, cond, h_c, h_c_hat)
        

    def forward_gru_cell(self, x, cond, h_c, h_c_hat):

        h = h_c
        h_hat = h_c_hat 
        
        
        x_hat = torch.cat((h.squeeze(0), cond), dim=-1).unsqueeze(1)

        
        _, h_hat = self.hyper_rnn(x_hat, h_hat)

        h_hat = h_hat.squeeze(0)
        h = h.squeeze(0)
        x = x.squeeze(1)

        z_h = self.z_h(h_hat).chunk(self.num_layers, dim=-1)
        z_x = self.z_x(h_hat).chunk(self.num_layers, dim=-1)
        z_b = self.z_b(h_hat).chunk(self.num_layers, dim=-1)

        
        rin = []

        for i in range(self.num_layers):
            d_h = self.d_h[i](z_h[i]).squeeze(1)
            d_x = self.d_x[i](z_x[i]).squeeze(1)
            if i == self.num_layers-1:
                y = torch.tanh(
                    rin[0] * d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + 
                    d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + 
                    self.d_b[i](z_b[i]).squeeze(1)
                )
            else:
                y = torch.sigmoid(
                    d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + self.d_b[i](z_b[i]).squeeze(1)
                )
            if self.layer_norm:
                rin.append(self.layer_norm[i](y))
            else:
                rin.append(y)

        r, i, n = rin # [b, n]
        h_next = n + i * (h - n)
        h_next = h_next.unsqueeze(0)
        
        return h_next, h_next, h_hat


    def initialize_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        
        for i in range(self.num_layers):
            nn.init.orthogonal_(self.w_h[i])
            nn.init.orthogonal_(self.w_x[i])
        
        for w in self.parameters():
            if w not in self.w_h and w not in self.w_x:
                w.data.uniform_(-std, std)



class DynamicHyperGRU(nn.Module):
    def __init__(self, inp_channel: int, out_channel: int, rnn_size: int, num_layers: int, sample_rate: int, hyper_rnn_size: int = 8, n_z_size: int = 24, num_conds: int = 0, layer_norm: bool = False, rnn_bias: bool = True):
        super().__init__()

        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.hyper_rnn_size = hyper_rnn_size
        self.n_z_size = n_z_size
        self.num_conds = num_conds
        self.layer_norm = layer_norm
        self.rnn_bias = rnn_bias
        self.sample_rate = sample_rate

        self.main_rnn = DynamicHyperCell(
            input_size = self.num_conds,
            hidden_size = self.rnn_size, 
            num_layers = self.num_layers,
            hyper_size = self.hyper_rnn_size, 
            n_z = self.n_z_size, 
            layer_norm = self.layer_norm, 
            bias = rnn_bias
        )

        self.linear_out = nn.Linear(
            self.rnn_size,
            self.out_channel,
            bias = self.rnn_bias
        )

    def compute_receptive_field(self): 
        return 1, (1/self.sample_rate) * 1000 
    

    def compute_num_of_params(self):
        return(f'Number of Trainable Params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
    

    def forward(
        self, x, c, h0, h_hat0):
        # B x C x T -> B x T x C
        x = x.permute(0, 2, 1)
        x, h = self.main_rnn(x, c, h0, h_hat0)
        x = self.linear_out(x)
        x = torch.tanh(x)
        x = x.permute(0, 2, 1)

        return x, h, None






if __name__ == "__main__":

    model = DynamicHyperGRU(
        inp_channel=1,
        out_channel=1,
        rnn_size=4,
        num_layers=3,
        hyper_rnn_size=4,
        sample_rate=48000,
        n_z_size=256,
        num_conds=8,
        layer_norm=False
    )

    receptive_field_samples, _ = model.compute_receptive_field()

    x = torch.randn(32, 1, 1024 + receptive_field_samples - 1)
    c = torch.randn(32, 8)
    h = torch.ones(1, 32, 4)
    h_hat = torch.ones(1, 32, 4)

    out, _, _ = model(x, c, h, h_hat)

    print(f'out shape: {out.shape}')
    print(f'Number of parameters: {model.compute_num_of_params()}')

