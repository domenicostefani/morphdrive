import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Classifier(nn.Module):
    def __init__(self, input_dim=8, n_classes=4):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, activation='leaky', bn=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = nn.LeakyReLU() if activation == 'leaky' else nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    

class DeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='silu', bn=True):
        super(DeConvLayer, self).__init__()
        self.bn = bn
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU() if activation == 'silu' else nn.Tanh()

    def forward(self, x):
        x = self.deconv(x)
        x = self.batch_norm(x) if self.bn else x
        x = self.activation(x)
        return x


class LatentSpace(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=16, weigth=1):
        super(LatentSpace, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
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
    def __init__(self, input_dim, pedal_latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = ConvLayer(input_dim, 8, kernel_size=4, stride=2, padding=1)
        self.conv1bis = ConvLayer(8, 8, kernel_size=4, stride=2, padding=3, dilation=2)
        self.conv1ter = ConvLayer(8, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvLayer(16, 16, kernel_size=6, stride=4, padding=1)
        self.conv3 = ConvLayer(16, 32, kernel_size=6, stride=3, padding=1)
        self.conv4 = ConvLayer(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = ConvLayer(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv6 = ConvLayer(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv7 = ConvLayer(128, 256, kernel_size=4, stride=2, padding=0)
        self.conv8 = ConvLayer(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv9 = ConvLayer(128, 128, kernel_size=4, stride=2, padding=1)
        self.flat = nn.Flatten()
        self.fully = FullyConnected(3584, pedal_latent_dim*64) 

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv1bis(x)
        x2 = self.conv1ter(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.flat(x)
        x = self.fully(x)

        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fully1 = FullyConnected(latent_dim, latent_dim * 16)
        self.fully2 = FullyConnected(latent_dim * 16, 11008)
        self.deconv1 = DeConvLayer(128, 256, kernel_size=4, stride=2, padding=1, activation='silu')
        self.deconv2 = DeConvLayer(256, 512, kernel_size=5, stride=2, padding=1, activation='silu')
        self.deconv3 = DeConvLayer(512, 256, kernel_size=4, stride=2, padding=1, activation='silu') 
        self.deconv4 = DeConvLayer(256, 128, kernel_size=4, stride=2, padding=1, activation='silu')  
        self.deconv5 = DeConvLayer(128, 64, kernel_size=4, stride=2, padding=2, activation='silu')    
        self.deconv6 = DeConvLayer(64, 32, kernel_size=4, stride=2, padding=2, activation='silu')    
        self.deconv7 = DeConvLayer(32, 32, kernel_size=4, stride=2, padding=2, activation='silu', bn=False)    
        self.deconv8 = DeConvLayer(32, 16, kernel_size=4, stride=2, padding=1, activation='silu', bn=False)    
        self.deconv9 = DeConvLayer(16, 16, kernel_size=3, stride=2, padding=2, activation='silu', bn=False)
        self.deconv10 = DeConvLayer(16, 8, kernel_size=4, stride=2, padding=2, activation='tanh', bn=False)    
        self.final_conv = ConvLayer(8, output_dim, kernel_size=3, stride=1, padding=1, activation='tanh') 
        self.denoiser = nn.Sequential(
            nn.Conv1d(output_dim, 16, kernel_size=3, stride=1, padding=1, groups=1),
            nn.Tanh(),
            nn.Conv1d(16, output_dim, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
    

    def forward(self, z):
        x = self.fully1(z)
        x = self.fully2(x)
        x = rearrange(x, 'b (c h) -> b c h', c=128, h=86)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.deconv8(x)
        x = self.deconv9(x)
        x = self.deconv10(x)
        x = self.final_conv(x)
        x = self.denoiser(x)

        return x



class Pedaliny_VAE(nn.Module):
    def __init__(self, 
                 input_dim=1,
                 pedal_latent_dim=8, 
                 ):
        super(Pedaliny_VAE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(input_dim, pedal_latent_dim)
        self.decoder = Decoder(input_dim, pedal_latent_dim)
        self.pedal_latent_space = LatentSpace(pedal_latent_dim*64, pedal_latent_dim)

    def forward(self, x):
        features = self.encoder(x)
        mu, logvar, z = self.pedal_latent_space(features)
        output = self.decoder(z)
        return output, mu, logvar, z




if __name__ == '__main__':
    model = Pedaliny_VAE(input_dim=1, pedal_latent_dim=8).to('cuda')
    input = torch.randn(1, 1, 88200).to('cuda') 
    output, mu, logvar, z = model(input)
    print(f'OUTPUT SHAPE : {output[0].shape}')
    print(f'MU SHAPE : {mu.shape}')
    print(f'LOGVAR SHAPE : {logvar.shape}')
    print(f'Z SHAPE : {z.shape}')
    print(f'NUMBER OF PARAMETERS: {sum(p.numel() for p in model.parameters())}')

