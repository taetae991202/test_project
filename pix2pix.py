from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt

import cv2
import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MakeDataset(Dataset):
    def __init__(self, path2orign_img, path2blur_img, transform=False):
        super().__init__()
        self.path2orign_img = path2orign_img
        self.path2blur_img = path2blur_img
        self.img_filenames = [x for x in listdir(self.path2orign_img)]
        self.transform = transform

    def __getitem__(self, index):
        a = Image.open(join(self.path2orign_img, self.img_filenames[index])).convert('RGB')
        b = Image.open(join(self.path2blur_img, self.img_filenames[index])).convert('RGB')

        if self.transform:
            a = self.transform(a)
            b = self.transform(b)

        return a, b

    def __len__(self):
        return len(self.img_filenames)

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64,128)
        self.down3 = UNetDown(128,256)
        self.down4 = UNetDown(256,512,dropout=0.5)
        self.down5 = UNetDown(512,512,dropout=0.5)
        self.down6 = UNetDown(512,512,dropout=0.5)
        self.down7 = UNetDown(512,512,dropout=0.5)
        self.down8 = UNetDown(512,512,normalize=False,dropout=0.5)

        self.up1 = UNetUp(512,512,dropout=0.5)
        self.up2 = UNetUp(1024,512,dropout=0.5)
        self.up3 = UNetUp(1024,512,dropout=0.5)
        self.up4 = UNetUp(1024,512,dropout=0.5)
        self.up5 = UNetUp(1024,256)
        self.up6 = UNetUp(512,128)
        self.up7 = UNetUp(256,64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8,d7)
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)

        return u8


class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2,64,normalize=False)
        self.stage_2 = Dis_block(64,128)
        self.stage_3 = Dis_block(128,256)
        self.stage_4 = Dis_block(256,512)

        self.patch = nn.Conv2d(512,1,3,padding=1) # 16x16 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x



class PIX2PIX():
    def __init__(self,
                 path2orign_img,
                 path2blur_img,
                 path2models,
                 gen_img_dir,
                 input_channels=3,
                 loss_func_gan=nn.BCELoss(),
                 loss_func_pix=nn.L1Loss(),
                 lambda_pixel=100,
                 lr=2e-5,
                 beta1=0.5,
                 beta2=0.999,
                 num_epochs=15,
                 save_cnt=100,
                 ):

        self.transform = transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize((256, 256))])

        self.path2orign_img = path2orign_img
        self.path2blur_img = path2blur_img
        self.train_ds = MakeDataset(path2blur_img, path2orign_img, transform=self.transform)
        self.train_dl = DataLoader(self.train_ds, batch_size=8, shuffle=False)
        self.model_gen = GeneratorUNet(input_channels).to(device)
        self.model_dis = Discriminator().to(device)

        self.loss_func_gan = loss_func_gan
        self.loss_func_pix = loss_func_pix

        self.lambda_pixel = lambda_pixel

        self.patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.opt_dis = optim.Adam(self.model_dis.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.opt_gen = optim.Adam(self.model_dis.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        self.num_epochs = num_epochs
        self.view_cnt = save_cnt

        self.path2models = path2models

        self.gen_img_dir = gen_img_dir

    def initialize_weights(model):
        class_name = model.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)

    def train(self):

        self.model_gen.train()
        self.model_dis.train()

        batch_count = 0
        start_time = time.time()

        loss_hist = {'gen': [],
                     'dis': []}

        for epoch in range(self.num_epochs):
            for a, b in self.train_dl:
                ba_si = a.size(0)

                # real image
                real_a = a.to(device)
                real_b = b.to(device)

                # patch label
                real_label = torch.ones(ba_si, *self.patch, requires_grad=False).to(device)
                fake_label = torch.zeros(ba_si, *self.patch, requires_grad=False).to(device)

                # generator
                self.model_gen.zero_grad()

                fake_b = self.model_gen(real_a)  # 가짜 이미지 생성
                out_dis = self.model_dis(fake_b, real_b)  # 가짜 이미지 식별

                gen_loss = self.loss_func_gan(out_dis, real_label)
                pixel_loss = self.loss_func_pix(fake_b, real_b)

                g_loss = gen_loss + self.lambda_pixel * pixel_loss
                g_loss.backward()
                self.opt_gen.step()

                # discriminator
                self.model_dis.zero_grad()

                out_dis = self.model_dis(real_b, real_a)  # 진짜 이미지 식별
                real_loss = self.loss_func_gan(out_dis, real_label)

                out_dis = self.model_dis(fake_b.detach(), real_a)  # 가짜 이미지 식별
                fake_loss = self.loss_func_gan(out_dis, fake_label)

                d_loss = (real_loss + fake_loss) / 2.
                d_loss.backward()
                self.opt_dis.step()

                loss_hist['gen'].append(g_loss.item())
                loss_hist['dis'].append(d_loss.item())

            print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' % (
            epoch + 1, g_loss.item(), d_loss.item(), (time.time() - start_time) / 60))

            path2models = './models/'
            os.makedirs(path2models, exist_ok=True)
            path2weights_gen = os.path.join(path2models, 'epoch' + str(epoch) + '_weights_gen.pt')
            path2weights_dis = os.path.join(path2models, 'epoch' + str(epoch) + '_weights_dis.pt')
            torch.save(self.model_gen.state_dict(), path2weights_gen)
            torch.save(self.model_dis.state_dict(), path2weights_dis)

            with torch.no_grad():
                for a, b in self.train_dl:
                    fake_imgs = self.model_gen(a.to(device)).detach().cpu()
                    real_imgs = b
                    break

            for ii in range(0, 1, 1):
                path2results_fake = os.path.join(self.gen_img_dir,
                                                 'epoch_' + str(epoch + 1) + '_fake_img' + str(ii) + '.jpg')
                to_pil_image(0.5 * fake_imgs[ii] + 0.5).save(path2results_fake)