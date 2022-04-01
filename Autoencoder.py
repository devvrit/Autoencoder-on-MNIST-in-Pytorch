import warnings
from shampoo import *
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision as tv
import torch
#%load_ext autoreload
#%autoreload


warnings.filterwarnings("ignore")


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Data Preprocessing

batch_size = 1000


transform = transforms.Compose([transforms.ToTensor()])


trainset = tv.datasets.MNIST(
    root='./data',  train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False, num_workers=1)

testset = tv.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=1)


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        torch.manual_seed(3)
        self.encoder = nn.Sequential(
            nn.Linear(784, 1000),
            nn.Tanh(),
            
            nn.Linear(1000, 500),
            nn.Tanh(),

            nn.Linear(500, 250),
            nn.Tanh(),

            nn.Linear(250, 30),
            nn.Tanh())

        self.decoder = nn.Sequential(

            nn.Linear(30, 250),

            nn.Tanh(),
            nn.Linear(250, 500),

            nn.Tanh(),
            nn.Linear(500, 1000),

            nn.Tanh(),
            nn.Linear(1000, 784))

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# Defining Parameters

num_epochs = 2

model = Autoencoder().to(device)

distance = torch.nn.BCEWithLogitsLoss(
    weight=None, size_average=None, reduce=False, reduction='none', pos_weight=None)


# optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
optimizer = Shampoo(model.parameters(), lr=0.001)
train_loss = []
test_loss = []

debug = False

tot_size=0
if debug:

    num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    loss_total = 0
    tot_size = 0

    numBatches = len(dataloader)

    for it, data in enumerate(dataloader):
        optimizer.zero_grad()
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
        loss = distance(output.view(img.size(0), -1),
                        img.view(img.size(0), -1)).sum(dim=1).mean(dim=0)

        # ===================backward====================

        loss.backward()
        optimizer.step()
        loss_total += loss.item()*img.size(0)
        tot_size += img.size(0)
        print('loss: {} iteration: {}/{}'.format(loss.item(), it, numBatches))
    # ===================log========================
    print('epoch [{}/{}], loss_total:{:.4f}, loss:{:.4f}'.format(epoch +
                                                                 1, num_epochs, loss_total/tot_size, loss.item()))
    train_loss.append(loss_total/tot_size)
    print("Evaluation TEST Set:")
    model.eval()
    loss_total = 0
    tot_size = 0
    for data in testloader:
        img, _ = data
        img = Variable(img).to(device)
        output = model(img.view(img.size(0), -1))
        loss = distance(output.view(img.size(0), -1),
                        img.view(img.size(0), -1)).sum(dim=1).mean(dim=0)
        loss_total += loss.item()*img.size(0)
        tot_size += img.size(0)
    print('test_loss_total:{:.4f}'.format(loss_total/tot_size))
    test_loss.append(loss_total/tot_size)

import matplotlib.pyplot as plt
plt.plot([i for i in range(len(train_loss))], train_loss)
plt.xlabel("epochs")
plt.ylabel("train_loss")
#plt.title("Shampoo optimizer")
plt.savefig("train_loss.png")

plt.clf()
plt.plot([i for i in range(len(test_loss))], test_loss)
plt.xlabel("epochs")
plt.ylabel("test_loss")
#plt.title("Shampoo optimizer")
plt.savefig("test_loss.png")

print("train_loss: " + str(train_loss))
print("test_loss: " + str(test_loss))
