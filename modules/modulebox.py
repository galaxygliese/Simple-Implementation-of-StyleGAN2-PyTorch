#-*- coding:utf-8 -*-
#
# --- StyleGan2 Modules ---
# Constant_Input, Pixel_Wise_Normalization,
# GaussianBlur, Add_Noise, Truncation_Trick
# Adjusted_FC, Adjusted_Conv2d,
# Modulated_Conv2d,
# toRGB, Residual,
# ModConv2d_SBlock, ModConv2d_DBlock
# B-> Batch size  C-> channel(input) K-> channel(output), L-> size, H->height, W->Width
# -------------------------

import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np


class Constant_Input(nn.Module):
    def __init__(self, channel, size):
        super().__init__()
        self.constant_tensor = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, x): #x: (B, *)
        B = x.shape[0]
        return self.constant_tensor.repeat(B, 1, 1, 1)

class Pixel_Wise_Normalization(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z): # z: (B, L)
        return z / torch.sqrt((z**2 + self.eps).mean(dim=1, keepdim=True))


class Truncation_Trick(nn.Module):
    def __init__(self, psi):
        super().__init__()
        self.psi = psi

    def forward(self, w):# w:(B, L)
        m = w.mean(0, keepdim=True)
        return m - self.psi * (w - m)


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=4):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.kernel_size = kernel_size
        self.register_buffer('kernel', self.make_kernel(kernel_size))


    def make_kernel(self, kernel_size):
        row = []
        shift_size, res = kernel_size - 2, kernel_size % 2
        assert shift_size > 0
        row = [1+i*shift_size for i in range(kernel_size//2+res)]
        row += row[::-1][res:kernel_size//2+res] # [1, 3, 3, 1](k=4)

        kernel = torch.tensor(row, dtype=torch.float32)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum() # (kernel_size, kernel_size)
        return kernel

    def forward(self, x): #x: (B, C, H, W)
        B, C, _, _ = x.shape
        return F.conv2d(x, weight=self.kernel.expand(C, 1, self.kernel_size, self.kernel_size), bias=None, padding=self.padding, groups=C)


class Adjusted_FC(nn.Module):
    def __init__(self, input_size, output_size, bias=False, activation=True, alpha=0.2, bias_value=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.full((output_size, ), bias_value)) if bias else None
        self.weight_scale = 1. / np.sqrt(input_size)
        self.activation = nn.LeakyReLU(alpha, inplace=True) if activation else None

    def forward(self, x): #x: (B, L)
        x = F.linear(x, self.weight*self.weight_scale, bias=self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Adjusted_Conv2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, bias=False, bias_value=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_channel, input_channel, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.full((output_channel, ), bias_value)) if bias else None
        self.weight_scale = 1. / np.sqrt(input_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding

    def forward(self, x): #x: (B, C, H, W)
        x = F.conv2d(x, self.weight*self.weight_scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return x


class Add_Noise(nn.Module): # B-block
    def __init__(self):
        super().__init__()
        self.noise_scaler = nn.Parameter(torch.zeros(1))

    def forward(self, x): #x: (B, C, H, W)
        B, C, H, W = x.shape
        noise = x.new_empty(B, 1, H, W).uniform_()
        return x + (1 + self.noise_scaler) * noise

class Modulated_Conv2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, style_size, padding=1, demodulate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, output_channel, input_channel, kernel_size, kernel_size)) #1 for batchsize
        self.bias = nn.Parameter(torch.zeros(output_channel, ))
        self.weight_scale = 1. / np.sqrt(input_channel * kernel_size ** 2)

        ## Important!
        self.style_scale = nn.Parameter(torch.zeros(1))

        self.demodulate = demodulate
        self.modulate = Adjusted_FC(style_size, input_channel, bias=True, bias_value=1.)
        self.padding = padding
        self.eps = 1e-8

    def forward(self, x, w): #x: (B, C, H, W)-> feature map   w:(B, L)->style
        B, C, H, W = x.shape
        _, oC, _, k, k = self.weight.shape
        w = self.modulate(w).view(B, 1, C, 1, 1) * (self.style_scale + 1)
        weight = self.weight * self.weight_scale * w # modulated weight
        if self.demodulate:
            sigma_inv = 1. / torch.sqrt((weight**2).sum([2, 3, 4]) + self.eps) # index [2, 3, 4] -> output_channel, kernel_size, kernel_size
            weight = weight * sigma_inv.view(B, oC, 1, 1, 1) # demodulated weight
        x = F.conv2d(x.view(1, B*C, H, W), weight.view(B*oC, C, k, k), bias=None, stride=1, padding=self.padding, groups=B)
        return x.view(B, oC, H, W)


class Residual(nn.Module):
    def __init__(self, input_channel, output_channel, blur_size=4, alpha=0.2):
        super().__init__()
        self.conv1 = Adjusted_Conv2d(input_channel, input_channel, 3, 1, 1, bias=False)
        self.blur1 = GaussianBlur(blur_size)
        self.conv2 = Adjusted_Conv2d(input_channel, output_channel, 3, 2, 1, bias=False)
        self.skip_conv = Adjusted_Conv2d(input_channel, output_channel, 1, 2, 0, bias=False)
        self.blur2 = GaussianBlur(blur_size)

        self.activation1 = nn.LeakyReLU(alpha, inplace=True)
        self.activation2 = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x): #x: (B, C, H, W)
        o = self.activation1(self.conv1(x))
        o = self.blur1(o)
        o = self.activation2(self.conv2(o))

        s = self.blur2(x)
        s = self.skip_conv(s)

        o = (o + s) / np.sqrt(2)
        return o


class toRGB(nn.Module):
    def __init__(self, input_channel, style_size, rgb_channel=3):
        super().__init__()
        self.conv = Modulated_Conv2d(input_channel, rgb_channel, 1, style_size, padding=0, demodulate=False)

    def forward(self, x, w): #x: (B, C, H, W)-> feature map   w:(B, L)->style
        x = self.conv(x, w)
        return x


class ModConv2d_SBlock(nn.Module): #Single Block -> First conv layer of Generator
    def __init__(self, input_channel, output_channel, kernel_size, style_size, alpha=0.2, rgb_channel=3):
        super().__init__()
        self.modconv2d = Modulated_Conv2d(input_channel, output_channel, kernel_size, style_size, demodulate=True)
        self.add_noise = Add_Noise()
        self.activation = nn.LeakyReLU(alpha, inplace=True)
        self.torgb = toRGB(output_channel, style_size, rgb_channel=rgb_channel)

    def forward(self, x, w): #x: (B, C, H, W)
        x = self.modconv2d(x, w)
        x = self.add_noise(x)
        x = self.activation(x)
        im = self.torgb(x, w)
        return x, im

class ModConv2d_DBlock(nn.Module): # Double Block
    def __init__(self, input_channel, output_channel, kernel_size, style_size, alpha=0.2, rgb_channel=3):
        super().__init__()
        self.modconv2d1 = Modulated_Conv2d(input_channel, input_channel, kernel_size, style_size, demodulate=True)
        self.add_noise1 = Add_Noise()
        self.activation1 = nn.LeakyReLU(alpha, inplace=True)

        self.modconv2d2 = Modulated_Conv2d(input_channel, output_channel, kernel_size, style_size, demodulate=True)
        self.add_noise2 = Add_Noise()
        self.activation2 = nn.LeakyReLU(alpha, inplace=True)
        self.torgb = toRGB(output_channel, style_size, rgb_channel=rgb_channel)

    def forward(self, x, w):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.modconv2d1(x, w)
        x = self.add_noise1(x)
        x = self.activation1(x)

        x = self.modconv2d2(x, w)
        x = self.add_noise2(x)
        x = self.activation2(x)
        im = self.torgb(x, w)
        return x, im


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    module1 = ModConv2d_SBlock(512, 512, 3, 512)
    module2 = ModConv2d_DBlock(512, 512, 3, 512)
    x = torch.randn(2, 512, 4, 4)
    w = torch.randn(2, 512)
    o1, im1 = module1(x, w)
    o2, im2 = module2(o1, w)
    print(o1.shape, im1.shape)
    print(o2.shape, im2.shape)
