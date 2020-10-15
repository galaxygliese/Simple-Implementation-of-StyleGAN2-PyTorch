#-*- coding:utf-8 -*-

from network import Generator
from toolbox import parse
import matplotlib.pyplot as plt
import torchvision
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configfile', default='./configs/stylegan2_tiny.conf', type=str)
parser.add_argument('-d', '--device', default='cuda', type=str)
parser.add_argument('-n', '--num', default=64, type=int)
parser.add_argument('-w', '--weightfile', type=str)
args = parser.parse_args()

device = args.device
config = parse(args.configfile)

G = Generator(config).to(device)
G.load_state_dict(torch.load(args.weightfile, map_location=device))
G.eval()

with torch.no_grad():
    z = torch.randn(args.num, int(config['style_size']), device=device)
    I, w = G(z)

print(torch.mean(z[0]-z[1]), torch.mean(z[1]-z[2]))
print(torch.mean(w[0]-w[1]), torch.mean(w[1]-w[2]))
print(torch.mean(I[0]-I[1]), torch.mean(I[1]-I[2]))
img = torchvision.utils.make_grid(I).permute(1, 2, 0).cpu().detach().numpy()
plt.imshow(img)
plt.show()
