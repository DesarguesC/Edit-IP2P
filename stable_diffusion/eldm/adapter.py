import torch
import torch.nn as nn
from collections import OrderedDict
from einops import (repeat, rearrange)
from stable_diffusion.ldm.modules.diffusionmodules.util import (conv_nd, avg_pool_nd, timestep_embedding)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x

@torch.no_grad()   # ???    
def fix_shape(t, x):
    assert len(t.shape)+2 == len(x.shape), f't.shape = {t.shape}, x.shape = {x.shape}'
    t = repeat(rearrange(t, 'b c -> b c 1'), 'b c 1 -> b c h', h=x.shape[2])
    t = repeat(rearrange(t, 'b c h -> b c h 1'), 'b c h 1 -> b c h w', w=x.shape[-1])
    return t
        
    
class TimeEmbed(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, res=False):
        super(TimeEmbed, self).__init__()
        self.res = res
        self.mid_channels = (in_channels + out_channels) // 2
        self.time_embedding = nn.Sequential(
            nn.Linear(in_channels, self.mid_channels),
            nn.SiLU(),
            nn.Linear(self.mid_channels, out_channels),
            QuickGELU()
        )

    def forward(self, t):
        # print(f'in TimeEmbed forward (before): t.device = {t.device}, t.shape = {t.shape}')
        return self.time_embedding(t) + ( 0. if not self.res else self.mid_channels * 1. * t )
    
        
        
def view_shape(u: list):
    assert isinstance(u, list), f'type: {type(u)}'
    for i in u:
        print(i.shape)

class Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, unshuffle=4, sk=False, use_conv=True, repeat_only=False, use_time=False):
        super(Adapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(unshuffle)
        self.cin = cin
        self.repeat_only = repeat_only
        self.use_time = use_time
        if use_time:
            self.first_time_emb = TimeEmbed(in_channels=self.cin, out_channels=channels[0])
            self.time_embeddings = [TimeEmbed(in_channels=channels[i-1], out_channels=channels[i], res=False) for i in range(1, len(channels))]
            self.time_embeddings.append(TimeEmbed(in_channels=channels[-1], out_channels=channels[-1], res=True))
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        

    def forward(self, x, param, t=None):
        # unshuffle
        # print(f'adapter forward: param = {param}')
        x = self.unshuffle(x)
        # extract features
        features = []
        t_ = []
        x = self.conv_in(x)
        t_emb =  self.first_time_emb(timestep_embedding(t, dim=self.cin, repeat_only=self.repeat_only))\
                                                                                        if t != None else None
        # batch * cin
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            # print(f'before: t_emb.device = {t_emb.device}')
            t_emb = self.time_embeddings[i](t_emb.to('cpu')).to(DEVICE) if t!= None else None
            # probably due to the selection appeared in forward step
            # print(f'after (no fix, input of next emb): t_emb.shape = {t_emb.shape}')
            t_emb_ = fix_shape(t_emb, x) if t != None else None
            features.append(x)
            t_.append(t_emb_ * (1. if param is None else param))
        return features, t_
            
            
            


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleAdapter(nn.Module):

    def __init__(self, width=1024, context_dim=768, num_head=8, n_layes=3, num_token=4):
        super().__init__()

        scale = width ** -0.5
        self.transformer_layes = nn.Sequential(*[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)])
        self.num_token = num_token
        self.style_embedding = nn.Parameter(torch.randn(1, num_token, width) * scale)
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, context_dim))

    def forward(self, x):
        # x shape [N, HW+1, C]
        style_embedding = self.style_embedding + torch.zeros(
            (x.shape[0], self.num_token, self.style_embedding.shape[-1]), device=x.device)
        x = torch.cat([x, style_embedding], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, -self.num_token:, :])
        x = x @ self.proj

        return x


class ResnetBlock_light(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.block1 = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

    def forward(self, x):
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        return h + x


class extractor(nn.Module):
    def __init__(self, in_c, inter_c, out_c, nums_rb, down=False):
        super().__init__()
        self.in_conv = nn.Conv2d(in_c, inter_c, 1, 1, 0)
        self.body = []
        for _ in range(nums_rb):
            self.body.append(ResnetBlock_light(inter_c))
        self.body = nn.Sequential(*self.body)
        self.out_conv = nn.Conv2d(inter_c, out_c, 1, 1, 0)
        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=False)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        x = self.in_conv(x)
        x = self.body(x)
        x = self.out_conv(x)

        return x


class Adapter_light(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64):
        super(Adapter_light, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            if i == 0:
                self.body.append(extractor(in_c=cin, inter_c=channels[i]//4, out_c=channels[i], nums_rb=nums_rb, down=False))
            else:
                self.body.append(extractor(in_c=channels[i-1], inter_c=channels[i]//4, out_c=channels[i], nums_rb=nums_rb, down=True))
        self.body = nn.ModuleList(self.body)

    def forward(self, x):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        for i in range(len(self.channels)):
            x = self.body[i](x)
            features.append(x)

        return features
