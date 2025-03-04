import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
    

class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class DeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn=True):
        super(DeConvLayer, self).__init__()
        self.bn = bn
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.deconv(x)
        x = self.batch_norm(x) if self.bn else x
        x = self.activation(x)
        return x


class MultiScaleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], stride=1, padding=1):
        super(MultiScaleConvLayer, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=ks, stride=stride, padding=ks//2)
            for ks in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        conv_results = [conv(x) for conv in self.convs]
        x = torch.sum(torch.stack(conv_results), dim=0)  # Sum over the multi-scale outputs
        x = self.bn(x)
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  
        return self.relu(out)


class SubPixelConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(SubPixelConv1d, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv1d(in_channels, out_channels * upscale_factor, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = rearrange(x, 'b (out_channels upscale) l -> b out_channels (l upscale)', upscale=self.upscale_factor)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, input_dim=1):
        super(ConvEncoder, self).__init__()
        self.multi_scale_conv = MultiScaleConvLayer(input_dim, 8)
        self.conv0 = ConvLayer(8, 8, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvLayer(8, 8, kernel_size=4, stride=2, padding=1)
        self.residual = ResidualBlock(8, 8, kernel_size=3, stride=1, padding=1)
        self.multi_scale_conv2 = MultiScaleConvLayer(8, 8)
        self.conv1bis = ConvLayer(8, 8, kernel_size=4, stride=2, padding=3, dilation=2)
        self.conv1ter = ConvLayer(8, 8, kernel_size=3, stride=2, padding=1)
        self.residual2 = ResidualBlock(16, 16, kernel_size=3, stride=1, padding=1)
        self.multi_scale_conv3 = MultiScaleConvLayer(16, 16)
        self.conv2 = ConvLayer(16, 32, kernel_size=6, stride=4, padding=1)
        self.residual3 = ResidualBlock(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvLayer(32, 32, kernel_size=6, stride=3, padding=1)
        self.conv4 = ConvLayer(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = ConvLayer(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv6 = ConvLayer(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv7 = ConvLayer(128, 256, kernel_size=4, stride=2, padding=0)
        self.conv8 = ConvLayer(256, 128, kernel_size=4, stride=2, padding=1)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.multi_scale_conv(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.residual(x)
        x = self.multi_scale_conv2(x)
        x1 = self.conv1bis(x)
        x2 = self.conv1ter(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.residual2(x)
        x = self.multi_scale_conv3(x)
        x = self.conv2(x)
        x = self.residual3(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        flat = self.flat(x) #7168
        return flat


class LatentSpace(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=16, weigth=1):
        super(LatentSpace, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.weigth = weigth
    
    def _reparametrization_trick(self, mu, logvar, weight):
        sigma = torch.sqrt(torch.exp(logvar))
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size()).to(mu.device) # perche' lo devo mandare a device?
        z = mu + weight * sigma * eps
        return z

    def forward(self, x):
        x = self.linear_block(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = torch.tanh(self._reparametrization_trick(mu, logvar, self.weigth))
        return mu, logvar, z


class Encoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=16):
        super(Encoder, self).__init__()
        self.convencoder = ConvEncoder(input_dim)
        self.latent_space = LatentSpace(7808, latent_dim) #7168

    def forward(self, x):
        features = self.convencoder(x)
        mu, logvar, z = self.latent_space(features)
        return mu, logvar, z


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim=16):
        super(Decoder, self).__init__()
        self.fully1 = FullyConnected(latent_dim, latent_dim * 16)
        self.fully2 = FullyConnected(latent_dim * 16, 11904) #11008
        self.deconv00 = DeConvLayer(128, 128, kernel_size=4, stride=2, padding=0)
        self.pixel1 = SubPixelConv1d(128, 128, 2)
        #self.deconv0 = DeConvLayer(128, 128, kernel_size=4, stride=2, padding=0)
        self.deconv1 = DeConvLayer(128, 128, kernel_size=4, stride=2, padding=1) 
        self.deconv2 = DeConvLayer(128, 64, kernel_size=3, stride=2, padding=2)  
        self.pixel2 = SubPixelConv1d(64, 64, 2) 
        self.deconv4 = DeConvLayer(64, 32, kernel_size=4, stride=2, padding=2)    
        self.deconv5 = DeConvLayer(32, 16, kernel_size=4, stride=2, padding=2)    
        #self.deconv6 = DeConvLayer(16, 16, kernel_size=5, stride=2, padding=1, bn=False)  
        self.deconv7 = DeConvLayer(16, 8, kernel_size=4, stride=2, padding=2, bn=False)
        self.pixel3 = SubPixelConv1d(8, 8, 2)
        self.deconv8 = DeConvLayer(8, 4, kernel_size=4, stride=2, padding=3, bn=False)
        self.deconv8bis = DeConvLayer(4, 4, kernel_size=2, stride=1, padding=2, bn=False)
        self.deconv9 = DeConvLayer(4, output_dim, kernel_size=4, stride=2, padding=2, bn=False)     
    
    def forward(self, z):
        x = self.fully1(z)
        x = self.fully2(x)
        #x = rearrange(x, 'b (c h) -> b c h', c=128, h=86)
        x = rearrange(x, 'b (c h) -> b c h', c=128, h=93)
        x = self.deconv00(x)
        x = self.pixel1(x)
        #x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        #x = self.deconv3(x)
        x= self.pixel2(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        #x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.pixel3(x)
        x = self.deconv8(x)
        x = self.deconv8bis(x)
        x = self.deconv9(x)
        return x


def log_gauss(q_z, mu, logvar):
    llh = - 0.5 * (torch.pow(q_z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
    return torch.sum(llh, dim=1)


class Pedals_GMVAE(nn.Module):
    def __init__(self, 
                 input_dim=1,
                 latent_dim=16, 
                 n_pedals=5
                 ):
        super(Pedals_GMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_pedals = n_pedals
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)
        self._build_mu_lookup()
        self._build_logvar_lookup()
 
    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_pedals, self.latent_dim)
        nn.init.xavier_uniform_(mu_lookup.weight)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        logvar_lookup = nn.Embedding(self.n_pedals, self.latent_dim).to(self.device)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_lookup = logvar_lookup

  
    def log_gauss(self, q_z, mu, logvar):
        llh = - 0.5 * (torch.pow(q_z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
        return torch.sum(llh, dim=1)


    def approx_q_y(self, q_z, mu_lookup, logvar_lookup, k=5): #taken from https://github.com/yjlolo/gmvae-synth
        q_z_shape = list(q_z.size()) 

        batch_size = q_z_shape[0]
        log_q_y_logit = torch.zeros(batch_size, k).type(q_z.type())

        for k_i in torch.arange(0, k):
            k_i = torch.tensor(k_i).to(q_z.device)
            mu_k, logvar_k = mu_lookup(k_i), logvar_lookup(k_i)
            log_q_y_logit[:, k_i] = self.log_gauss(q_z, mu_k, logvar_k) + np.log(1 / k)

        q_y = torch.nn.functional.softmax(log_q_y_logit, dim=1)
        return log_q_y_logit, q_y


    def _infer_class(self, q_z):
        log_q_y_logit, q_y = self.approx_q_y(q_z, self.mu_lookup, self.logvar_lookup, k=self.n_pedals)
        val, ind = torch.max(q_y, dim=1)
        return log_q_y_logit, q_y, ind


    def forward(self, x):

        mu, logvar, z = self.encoder(x)
        log_q_y_logit, q_y, ind = self._infer_class(z)
        output = self.decoder(z)

        return output, mu, logvar, z, log_q_y_logit, q_y, ind


if __name__ == '__main__':
    model = Pedals_GMVAE(input_dim=1, latent_dim=16, n_pedals=5).to('cuda')
    input = torch.randn(1, 1, 192000).to('cuda') # 88200
    output = model(input)
    
    with open('gmvae_summary.txt', 'w', encoding='utf-8') as f:
        printBoth = lambda x: print(x, file=f) or print(x)
        printBoth(f'OUTPUT SHAPE : {output[0].shape}')
        printBoth(f'MU SHAPE : {output[1].shape}')
        printBoth(f'LOGVAR SHAPE : {output[2].shape}')
        printBoth(f'Z SHAPE : {output[3].shape}')
        printBoth(f'Z MIXTURE SHAPE : {output[4].shape}')
        printBoth(f'Q_Y SHAPE : {output[5].shape}')
        printBoth(f'IND SHAPE : {output[6].shape}')
        printBoth(f'NUMBER OF PARAMETERS: {sum(p.numel() for p in model.parameters())}')

        print('', file=f)

        with torch.no_grad():
            from torchinfo import summary
            summ = summary(model.to('cuda'), (1,1,192000))
            print(summ)
            print(summ, file=f)



