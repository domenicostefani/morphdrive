import torch
import torch.nn as nn


class Pedals_MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=8):
        super(Pedals_MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        return self.fc(x)
    

if __name__ == '__main__':
    model = Pedals_MLP(input_dim=2, output_dim=8)
    input = torch.randn(1, 2)
    output = model(input)

    with open('mlp_summary.txt', 'w', encoding='utf-8') as f:
        printBoth = lambda *args, **kwargs: print(*args, **kwargs) or print(*args, **kwargs, file=f)

        printBoth('sum(p.numel() for p in model.parameters() if p.requires_grad):',str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        printBoth('output.shape:', output.shape)
        with torch.no_grad():
            from torchinfo import summary
            summ = summary(model.to('cuda'), (1,2))
            print(summ)
            print(summ, file=f)