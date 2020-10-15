#-*- coding:utf-8 -*-

from torch.utils.tensorboard import SummaryWriter
from network import Generator, Discriminator
from toolbox import Train_Handler, parse
from dataset import Real_Data_Generator
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim as optim
import numpy as np
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=5, type=int)
parser.add_argument('-b', '--batchsize', default=16, type=int)
parser.add_argument('-c', '--configfile', default='./configs/stylegan2_tiny.conf', type=str)
parser.add_argument('-d', '--device', default='cuda', type=str)
parser.add_argument('-l', '--limit', default=10000, type=int)
parser.add_argument('--datafolder', type=str)
parser.add_argument('--dataset_name', default='celebA', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--ngpu', default=1, type=int)

parser.add_argument('--weight_folder', default='./weights', type=str)
parser.add_argument('--result_folder', default='./results', type=str)
parser.add_argument('--save_per_epoch', default=1, type=int)
parser.add_argument('--print_per_iteration', default=1, type=int)
parser.add_argument('--plot_per_iteration', default=1, type=int)
parser.add_argument('--seed', default=4546, type=int)
parser.add_argument('--size', default=256, type=int)
args = parser.parse_args()

config = parse(args.configfile)
device = args.device
seed = args.seed

def worker_init_fn(worker_id):
    np.random.seed(worker_id+seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if int(config['rgb']):
    composed = T.Compose([T.Resize(args.size), T.RandomHorizontalFlip(), T.RandomRotation(90), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
else:
    composed = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])

dataset = Real_Data_Generator(args.datafolder, transforms=composed, limit=args.limit)
dataloader = DataLoader(dataset, batch_size=args.batchsize, num_workers=4*args.ngpu, worker_init_fn=worker_init_fn)
print("Training Datas:", len(dataset))

if not os.path.exists(args.weight_folder):
    os.mkdir(args.weight_folder)

if not os.path.exists(args.result_folder):
    os.mkdir(args.result_folder)

G = Generator(config)
D = Discriminator(config)

G_optim = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
D_optim = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

writer = SummaryWriter()

print("Running On:", device)
train_handler = Train_Handler(args.dataset_name,
                args.epochs,
                G_optim,
                D_optim,
                dataloader,
                device = device,
                writer = writer,
                save_per_epoch = args.save_per_epoch,
                print_per_iteration = args.print_per_iteration,
                plot_per_iteration = args.plot_per_iteration,
                weight_folder=args.weight_folder,
                result_folder=args.result_folder
)

train_handler.run(G, D, config)
