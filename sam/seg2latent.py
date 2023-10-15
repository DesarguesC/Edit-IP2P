from torch import nn
import torch
# from stable_diffusion.ldm.modules.attention import LinearAttention as la
from stable_diffusion.ldm.modules.diffusionmodules.openaimodel import Upsample, Downsample
"""
                       SAM
        R^3         --------->      segmentation space
         |                                  |
    D(·) | E(·)                        D(·) | E(·) 
         |       feature extractor          |
    latent space    <---------      latent segmentation


"""


class ProjectionModel(nn.Module):
    # projection via one-shot learning
    # to extract latent feature from segmentation gaining from SAM
    def __init__(self, dim=4, dropout=0.5):
        # seg 2 latent | judge in latent space
        # channel => dim
        super().__init__()
        # in_channels = out_channels = dim
        self.dim = dim
        self.dropout = dropout

        self.down = Downsample(channels=dim, use_conv=False)    # decline two times via pool, keep channels
        self.up = Upsample(channels=dim, use_conv=True)         # expand two times via conv, keep channels

        self.conv1 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: 64 * 64
        assert len(x.shape) == 4, f'forward input shape = {x.shape}'
        x0 = self.up(x)                                        # x0: 128 * 128
        x1 = self.conv2(x0) + x0                               # x1: 128 * 128
        x2 = self.conv1(self.down(x1))                        # x3: 64 * 64
        x3 = self.conv1(x2)                                     # x4: 64 * 64

        return x3


class 


