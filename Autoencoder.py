# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:21:22 2019

@author: suchismitasa
"""

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from shampoo import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preprocessing
batch_size = 1000
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST(root='./data',  train=True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)
testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=1)

# Defining Model

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784,1000),
            nn.Tanh(),
            nn.Linear(1000,500),
            nn.Tanh(),
            nn.Linear(500,250),
            nn.Tanh(),
            nn.Linear(250, 30),
            nn.Tanh())

        self.decoder = nn.Sequential(
            nn.Linear(30,250),
            nn.Tanh(),
            nn.Linear(250,500),
            nn.Tanh(),
            nn.Linear(500,1000),
            nn.Tanh(),
            nn.Linear(1000, 784),
            nn.Tanh())

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 6, kernel_size=5),
        #     nn.ReLU(True),
        #     nn.Conv2d(6,16,kernel_size=5),
        #     nn.ReLU(True))

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(16,6,kernel_size=5),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(6,1,kernel_size=5),
        #     nn.ReLU(True),
        #     nn.Sigmoid())

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Defining Parameters

num_epochs = 300
model = Autoencoder().to(device)
distance = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
optimizer = Shampoo(model.parameters(), lr=0.01)
debug = False
tot_size=0
if debug:
    num_epochs=1
for epoch in range(num_epochs):
    model.train()
    loss_total=0
    tot_size=0
    for data in dataloader:
        img, _ = data
        img = Variable(img).to(device)
        if debug:
            print("img is: " + str(img))
            print("img.min() is: " + str(img.min()))
            print("img.max() is: " + str(img.max()))
        # ===================forward=====================
        output = model(img.view(img.size(0), -1))
        if debug:
            print("output is: " + str(output))
            print("output.min() is: " + str(output.min()))
            print("output.max() is: " + str(output.max()))
        loss = distance(output.view(img.size(0), 1, 28, 28), img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total+=loss.item()*img.size(0)
        tot_size+=img.size(0)
    # ===================log========================
    print('epoch [{}/{}], loss_total:{:.4f}, loss:{:.4f}'.format(epoch+1, num_epochs, loss_total/tot_size, loss.item()))
    print("Evaluation TEST Set:")
    model.eval()
    loss_total=0
    tot_size=0
    for data in testloader:
        img, _ = data
        img = Variable(img).to(device)
        output = model(img.view(img.size(0), -1))
        loss = distance(output.view(img.size(0), 1, 28, 28), img)
        loss_total+=loss.item()*img.size(0)
        tot_size+=img.size(0)
    print('test_loss_total:{:.4f}'.format(loss_total/tot_size))

