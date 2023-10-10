from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch import nn
import torch
from stable_diffusion.ldm.modules.attention import LinearAttention as la
from stable_diffusion.ldm.modules.attention import

"""
                       SAM
        R^3         --------->      segmentation space
         |                                  |
    D(·) | E(·)                        D(·) | E(·) 
         |       feature extractor          |
    latent space    <---------      latent segmentation



"""


class POS(nn.Module):
    # projection via one-shot learning
    # to extract latent feature from segmentation gaining from SAM
    def __init__(self, dim=3, dropout=0.5):
        # seg 2 latent | judge in latent spacec
        super.__init__()
        # in_channels = out_channels = dim
        self.dim = dim
        self.dropout = dropout

        self.conv1 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Conv2d(2 * self.dim, self.dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        assert len(x.shape) == 4
        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(torch.cat([x0, x1], dim=1) + x0)
        x3 = self.conv3(x2 + x)

        return x3





