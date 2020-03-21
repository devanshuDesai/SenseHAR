import numpy as np
from torch import nn
from modules import *

class PSM(nn.Module):
    def __init__(self, n_classes, k, fr, num_feat_map=64, p=0.3, shar_channels=3):
        super(PSM, self).__init__()
        self.shar_channels = shar_channels
        self.num_feat_map = num_feat_map
        self.encoder = Encoder(k, fr, num_feat_map, p, shar_channels)
        self.decoder = Decoder(n_classes, p)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        encodes = []
        outputs = []
        for device in x:
            encode = self.encoder(device)
            outputs.append(self.decoder(encode.cuda()))
            encodes.append(encode)
        # Add shared channel
        shared_encode = torch.mean(torch.stack(encodes), 2).permute(1,0,2).cuda()
        outputs.append(self.decoder(shared_encode))
        return torch.mean(torch.stack(outputs), 0)

class Encoder(nn.Module):
    def __init__(self, k, fr, num_feat_map=64, p=0.3, shar_channels=3):
        super(Encoder, self).__init__()
        self.k          = k
        self.fr         = fr
        self.channels   = num_feat_map
        self.conv1      = nn.Conv2d(1, 64, kernel_size=(1,1), padding=0)
        self.bnd1       = nn.BatchNorm2d(64)
        self.mp_100     = nn.MaxPool2d(kernel_size=(2,1))
        self.drop       = nn.Dropout(p)
        self.conv2      = nn.Conv2d(192, 64, kernel_size=(3,1), padding=(1,0))
        self.bnd2       = nn.BatchNorm2d(64)
        self.mp_75      = nn.MaxPool2d(kernel_size=(1,3))
        self.relu       = nn.ReLU(inplace=True)
        self.time_dist  = TimeDistributed(nn.Linear(192, shar_channels))

    def __call__(self, x):
        '''Takes one device at a time'''
        return self.forward(x)

    def forward(self, x):
        # TODO: Handle this natively in DataLoader
        x = x.type(torch.cuda.FloatTensor)

        x = x.permute(0,3,2,1)
        x = self.bnd1(self.relu(self.conv1(x)))
        if (self.fr == 100):
            x = self.mp_100(x)
        x = self.drop(x)
        x = x.permute(0,1,3,2)
        x = x.reshape(*x.shape, 1)
        x = x.reshape(x.shape[0], self.channels, -1, self.k, 3)
        x = x.reshape(x.shape[0], self.k*self.channels, 3, -1)
        x = x.permute(0, 1, 3, 2)

        x = self.bnd2(self.relu(self.conv2(x)))
        if (self.fr in [50,100]):
            x = self.mp_100(x)
        if (self.fr == 75):
            x = self.mp_75(x)
        x = self.drop(x)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x.shape[0], 3*self.channels, -1)
        x = self.time_dist(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n_classes, p=0.3):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(3, 32, num_layers=2, batch_first=True)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(32, n_classes)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = x.permute(1,0,2)
        x = self.drop(self.tanh(x[-1]))
        x = self.fc(x)
        return x


