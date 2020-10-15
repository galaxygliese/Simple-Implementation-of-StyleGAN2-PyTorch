#-*- coding:utf-8 -*-
#
# --- StyleGan2 Networks ---
#  Generator, Discriminator
# --------------------------

from modules import *
from toolbox import *
import torch.nn as nn
import torch


class Generator(nn.Module):
    """
        config: {'Mapping': [{'Adjusted_FC':[512, 512, ...]}, {'Adjusted_FC':[512, ...]}, ...],
                 'Generator':[{'Constant_Input':[512, ...]}, ...],
                 'Discriminator':[{''}]
                }
    """
    def __init__(self, config:dict):
        super().__init__()
        self.style_size = config['style_size']
        self.image_channel = 3 if int(config['rgb']) else 1
        self.constant_input = Constant_Input(int(config['input_channel']), int(config['input_size']))
        self.mapping = nn.Sequential(*[self.load_module(row[:-1], config['Mapping'][row]) for row in config['Mapping']])
        self.truncation_trick = Truncation_Trick(float(config['psi']))
        self.conv_blocks = nn.ModuleList([self.load_module(row[:-1], config['Generator_Block'][row], rgb_channel=self.image_channel) for row in config['Generator_Block']])

    def load_module(self, module_name, params, rgb_channel=3):
        params = {k:int(v) for k, v in params.items()}
        if 'Conv' in module_name:
            params['rgb_channel'] = rgb_channel
        return eval(module_name)(**params)

    def init(self):
        print("Generator initialized")
        for m in self.modules():
            if 'Modulated_Conv2d' in type(m).__name__ or 'FC' in type(m).__name__:
                nn.init.kaiming_normal_(m.weight)

    def forward(self, z):# z: (B, L)
        w = self.truncation_trick(self.mapping(z))
        x = self.constant_input(w)
        for i, block in enumerate(self.conv_blocks):
            x, s = block(x, w)
            if i == 0:
                im = s
            else:
                im = s + F.interpolate(im, scale_factor=2, mode='bilinear', align_corners=False)
        return im, w

class Discriminator(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.input_channel = int(config['Discriminator']['input_channel'])
        self.output_channel = int(config['Discriminator']['output_channel'])
        self.image_channel = 3 if int(config['rgb']) else 1
        self.input_conv = Adjusted_Conv2d(self.image_channel, self.input_channel, 1, 1, 0)
        self.residual_blocks = nn.Sequential(*[self.load_module(row[:-1], config['Discriminator'][row]) for row in config['Discriminator'] if row != 'input_channel' and row != 'output_channel'])

        self.fc = nn.Sequential(*[
                Adjusted_FC(self.output_channel, self.output_channel, bias=True),
                Adjusted_FC(self.output_channel, 1, bias=False, activation=False)
        ])

    def load_module(self, module_name, params):
        params = {k:int(v) for k, v in params.items()}
        return eval(module_name)(**params)

    def init(self):
        print("Discriminator initialized")
        for m in self.modules():
            if 'Modulated_Conv2d' in type(m).__name__ or 'FC' in type(m).__name__:
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):# x:(B, C, H, W)
        x = self.input_conv(x)
        x = self.residual_blocks(x)
        x = x.mean(3).mean(2)
        x = self.fc(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    from pprint import pprint
    import matplotlib.pyplot as plt
    config = parse('configs/stylegan2.conf')
    G = Generator(config)
    G.init()
    D = Discriminator(config)
    z = torch.randn(1, int(config['style_size']))
    im, w = G(z)
    print(im.shape)
    o = D(im)
    print(o, o.shape)
