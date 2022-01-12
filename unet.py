import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu  = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        print(x.shape)
        x = self.conv(x)
        x = self.relu(x)
        #x = self.conv2(x)
        x = self.pool(x)

        return x

class DecoderBlock(nn.Module):
    def __init__

class UneT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = EncoderBlock(in_channels=3, out_channels=16)
        self.encoder2 = EncoderBlock(in_channels=16, out_channels=32)
        self.encoder3 = EncoderBlock(in_channels=32, out_channels=64)
        self.encoder4 = EncoderBlock(in_channels=64, out_channels=128)

    def forward(self, x):
        p1 = self.encoder1(x)
        p2 = self.encoder2(p1)
        p3 = self.encoder3(p2)
        p4 = self.encoder4(p3)


        return p4

x = torch.randn(1, 3, 128, 128)

u = UneT()
z = u(x)
print(z.shape)
