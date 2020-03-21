from torch import nn
import torch

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.layer = module

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out_tensor = torch.zeros((x.shape[0], self.layer.out_features, x.shape[2]))
        for i in range(x.shape[2]):
            out_tensor[:,:,i] = self.layer(x[:,:,i])
        return out_tensor

