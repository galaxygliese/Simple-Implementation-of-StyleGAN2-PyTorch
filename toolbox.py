#-*- coding-utf-8 -*-
#
# ----- Tool Box ------
#  Config Parser
#  Training Tools
# --------------------

from configobj import ConfigObj
from pprint import pprint
from PIL import Image
import torchvision
import numpy as np
import torch
import json
import os

def parse(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError("Config File Not Found")
    config = ConfigObj(filepath)
    output = {k:v for k, v in list(config['DEFAULT'].items())+[(k, config[k]) for k in config if k != 'DEFAULT']}
    return output


class Train_Handler(object): # Real->1 Fake->0
    def __init__(self, dataset_name,
                       epochs,
                       G_optim,
                       D_optim,
                       dataloader,
                       device='cuda',
                       writer=None,
                       G_reg_per_iteration = 4,
                       D_reg_per_iteration = 16,
                       save_per_epoch=1,
                       plot_per_iteration=1,
                       print_per_iteration=5,
                       weight_folder='./weight',
                       result_folder='./result'):
        self.G_optim = G_optim
        self.D_optim = D_optim
        self.dataloader = dataloader
        self.dataset_name = dataset_name

        self.epochs = epochs
        self.device = device

        self.writer = writer
        self.save_per_epoch = save_per_epoch
        self.print_per_iteration = print_per_iteration
        self.plot_per_iteration = plot_per_iteration
        self.G_reg_per_iteration = G_reg_per_iteration
        self.D_reg_per_iteration = D_reg_per_iteration

        self.weight_folder = weight_folder
        self.result_folder = result_folder

    def save(self, G, D, epoch):
        G_weight_path = os.path.join(self.weight_folder, 'Generator_{0}_epoch{1}.pth'.format(self.dataset_name, str(epoch)) )
        D_weight_path = os.path.join(self.weight_folder, 'Discriminator_{0}_epoch{1}.pth'.format(self.dataset_name, str(epoch)) )
        torch.save(G.state_dict(), G_weight_path)
        torch.save(D.state_dict(), D_weight_path)

    def logging(self, iter, G_loss, D_loss):
        if self.writer is not None:
           self.writer.add_scalar('Loss/G_Loss', G_loss, iter)
           self.writer.add_scalar('Loss/D_loss', D_loss, iter)

    def export_images(self, epoch, imgs, num=16):
        B, _, H, W = imgs.shape
        grid_imgs = torchvision.utils.make_grid(imgs[:num, ...])
        grid_imgs = grid_imgs.cpu().permute(1, 2, 0).numpy() * 255
        pilo_imgs = Image.fromarray(grid_imgs.clip(0, 255).astype(np.uint8))
        pilo_imgs.save(os.path.join(self.result_folder, self.dataset_name+'_epoch{}.jpg'.format(str(epoch))))


    def freeze(self, model, flag=False):
        for param in model.parameters():
            param.requires_grad = not flag

    def D_logistic_loss(self, real_predicts, fake_predicts):
        real_loss = -torch.log(real_predicts)
        fake_loss = -torch.log(1-fake_predicts)
        return real_loss.mean() + fake_loss.mean()

    def D_regularization(self, real_predicts, real_images):
        B = real_predicts.shape[0]
        grad = torch.autograd.grad(outputs=real_predicts.sum(), inputs=real_images, create_graph=True)[0]
        reg_loss = (grad**2).reshape(B, -1).sum(1).mean()
        return reg_loss

    def G_logistic_loss(self, fake_predicts):
        fake_loss = -torch.log(fake_predicts)
        return fake_loss.mean()


    def G_path_length_regularization(self, fake_predicts, latents, gamma=0.01): #fake_predicts:(B, 3, H, W)  latents:(B, L)
        B, _, H, W = fake_predicts.shape
        y = torch.randn_like(fake_predicts) / np.sqrt(H * W)
        Jy = torch.autograd.grad(outputs=(fake_predicts * y).sum(), inputs=latents, create_graph=True)[0]
        l = torch.sqrt( (Jy**2).mean(1) )
        a = gamma * torch.mean(l)
        R = (l - a).pow(2).mean()
        return R

    def process(self, epoch, G, D, style_size):
        for iter, real_tensors  in enumerate(self.dataloader): # real_tensors:(B, 3, H, W)
            B = real_tensors.shape[0]

            # Train D
            self.freeze(G, True)
            self.freeze(D, False)

            z = torch.randn(B, style_size, device=self.device)
            I_real = real_tensors.to(self.device)

            I_fake, _ = G(z)
            fake_predicts = D(I_fake)
            real_predicts = D(I_real)
            D_loss = self.D_logistic_loss(real_predicts, fake_predicts)

            if iter % self.D_reg_per_iteration == 0:
                I_real.requires_grad = True
                real_predicts_ = D(I_real)
                L = self.D_regularization(real_predicts_, I_real)
                D_loss = D_loss + L
                # self.D_optim.zero_grad()
                # L.backward()
                # self.D_optim.step()

            self.D_optim.zero_grad()
            D_loss.backward()
            self.D_optim.step()

            # Train G
            self.freeze(G, False)
            self.freeze(D, True)

            z_ = torch.randn(B, style_size, device=self.device)
            I_fake_, w = G(z_)
            fake_predicts_ = D(I_fake_)

            G_loss = self.G_logistic_loss(fake_predicts_)

            if iter % self.G_reg_per_iteration == 0:
                R = self.G_path_length_regularization(I_fake_, w)
                G_loss = G_loss + R
                # self.G_optim.zero_grad()
                # R.backward()
                # self.G_optim.step()

            self.G_optim.zero_grad()
            G_loss.backward()
            self.G_optim.step()

            self.logging(iter, G_loss.detach(), D_loss.detach())

            if iter % self.print_per_iteration == 0:
                print("Epoch:[%d/%d] | Iteration: %d | G_loss: %.5f | D_loss: %.5f" % (epoch+1, self.epochs, iter+1, G_loss.detach(), D_loss.detach()))

            if iter % self.plot_per_iteration == 0:
                G.eval()
                I_fake, _ = G(z_)
                self.export_images(epoch+1, I_fake.detach())
                G.train()
        return I_fake_.detach()

    def run(self, G, D, config):
        G = G.to(self.device)
        D = D.to(self.device)
        style_size = int(config['style_size'])
        for epoch in range(self.epochs):
            fake_images = self.process(epoch, G, D, style_size)
            if epoch % self.save_per_epoch == 0:
                self.export_images(epoch+1, fake_images)
                self.save(G, D, epoch+1)
        if self.writer is not None:
            self.writer.close()




if __name__ == '__main__':
    from network import Generator
    config = parse('configs/stylegan2_very_tiny.conf')
    pprint(config)
    G = Generator(config)
    #D = Discriminator(config)
    T = Train_Handler(G, torch.rand(1), 1, 1, None, None, device='cpu')
    z = torch.randn(2, 128)
    I, w = G(z)
    T.G_path_length_regularization(I, w)
