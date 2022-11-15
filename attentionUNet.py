import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch import optim

import numpy

import time

from os import listdir
from os.path import join

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


class MakeDataset(Dataset):
    def __init__(self, path2limg, path2himg, transform=False):
        super().__init__()
        self.pathlimg = path2limg
        self.path2himg = path2himg
        self.img_filenames = [x for x in listdir(self.path2orign_img)]
        self.transform = transform

    def __getitem__(self, index):
        a = Image.open(join(self.path2orign_img, self.img_filenames[index])).convert('RGB')
        b = Image.open(join(self.path2blur_img, self.img_filenames[index])).convert('RGB')

        if self.transform:
            a = self.transform(a)
            a = a / 255.
            b = self.transform(b)

        return a, b

    def __len__(self):
        return len(self.img_filenames)

transform = transforms.Compose([
                    transforms.ToTensor(),
])

# path2img = (이미지 경로 입력)
# train_ds = MakeDataset(path2limg, path2himg, transform)
# train_dl = DataLoader(train_ds, batch_size = 32, shuffle=False)

class AttentionLayer(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, bias=False):
        super(AttentionLayer, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = x
        k_out = padded_x
        v_out = padded_x

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out = k_out.contiguous().view(batch, channels, height, width, -1)
        v_out = v_out.contiguous().view(batch, channels, height, width, -1)

        q_out = q_out.view(batch, channels, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bchwk,bchwk -> bchw', out, v_out).view(batch, -1, height, width)

        return out

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),  # or batchnomaliztion ? ? ?
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.up1 = UNetUp(in_channels, 64)
        self.attention1 = AttentionLayer(3, 1, 1)

        self.up2 = UNetUp(64, 128)
        self.attention2 = AttentionLayer(3, 1, 1)

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.attention3 = AttentionLayer(3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.attention4 = AttentionLayer(3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up1(x)  # 64
        x = self.attention1(x)  # 64

        x = self.up2(x)  # 128
        x = self.attention2(x)  # 128

        x = self.conv1(x)
        x = self.attention3(x)
        x = self.conv2(x)
        x = self.attention4(x)
        x = self.conv3(x)

        return x

# 가중치 초기화

def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

attentionunet = AttentionUNet()

attentionunet.apply(initialize_weights)

loss_func = nn.L1Loss()

lr = 0.00002
beta1 = 0.5
beta2 = 0.999

optimizer = optim.Adam(attentionunet.parameters(),lr=lr,betas=(beta1,beta2))

scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

def train(train_dl, model, loss_fn, optimizer, num_epochs):
  model.train()

  start_time = time.time()

  loss_hist = {'model':[]}

  for epoch in range(num_epochs):
    for a, b in train_dl:
      img = a.to(device)

      model.zero_grad
      fake_img = model(a)

      a = a * 255.

      loss = loss_fn(b, a)
      loss.backward()
      optimizer.step()

      scheduler.step()

      loss_hist['model'].append(loss.item())

    print('Epoch: %.0f, Loss: %.6f,  time: %.2f min' %(epoch, loss.item(), (time.time()-start_time)/60))
