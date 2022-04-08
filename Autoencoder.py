import warnings
import argparse
import time
from shampoo import *
import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision as tv
import matplotlib.pyplot as plt
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



parser = argparse.ArgumentParser(description='Autoencoder.py')
parser.add_argument('--quic', action='store_true', help='need to use quic?')
parser.add_argument('--Lambda', type=float, required=True, help='regularization prameter')
parser.add_argument('--beta1', type=float, required=True, help='beta1')
parser.add_argument('--beta2', type=float, required=True, help='beta2')
parser.add_argument('--max_iter', type=int,  default=100, help='quic maximum iterations to run')
parser.add_argument('--msg', type=int,  default=0, help='how much messages to print')
parser.add_argument('--update', type=int,  default=1, help='compute precondition every x steps')
parser.add_argument('--warmup', type=int,  default=0, help='linear warmup steps')
parser.add_argument('--epochs', type=int,  required=True, help='num_epochs for training')
parser.add_argument('--lr', type=float,  required=True, help='learning_rate')
args = parser.parse_args()

print("Args:")
print(args)

description="_reg_"+str(args.Lambda)+"_max_iter_"+str(args.max_iter)+"_update_"+str(args.update)+"_warmup_"+str(args.warmup)+"_lr_"+str(args.lr)
print("description: " + str(description))

@dataclass
class ShampooHyperParams:
  """Shampoo hyper parameters."""
  beta2: float = args.beta2
  diagonal_eps: float = 1e-6
  matrix_eps: float = 1e-12
  weight_decay: float = 0.0
  inverse_exponent_override: int = 0  # fixed exponent for preconditioner, if >0
  start_preconditioning_step: int = 1
  # Performance tuning params for controlling memory and compute requirements.
  # How often to compute preconditioner.
  preconditioning_compute_steps: int = args.update
  # How often to compute statistics.
  statistics_compute_steps: int = 1
  # Block size for large layers (if > 0).
  # Block size = 1 ==> Adagrad (Don't do this, extremely inefficient!)
  # Block size should be as large as feasible under memory/time constraints.
  block_size: int = 100000
  # Automatic shape interpretation (for eg: [4, 3, 1024, 512] would result in
  # 12 x [1024, 512] L and R statistics. Disabled by default which results in
  # Shampoo constructing statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
  best_effort_shape_interpretation: bool = True
  # Type of grafting (SGD or AdaGrad).
  # https://arxiv.org/pdf/2002.11803.pdf
  graft_type: int = LayerwiseGrafting.SGD
  # Nesterov momentum
  nesterov: bool = True
  quic = args.quic
  print("quic is: " + str(quic))
  ## quic params
  nondiagRegul = True ## regularization only on non-diagonal elements or everything
  quicInit = "invdiag" ## "invdiag" (X0=inv(diag(S))) | "inv" (X0=inv(S))
  quicIters = args.max_iter
  quicLambda = args.Lambda





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

num_epochs = args.epochs

model = Autoencoder().to(device)

distance = torch.nn.BCEWithLogitsLoss(
    weight=None, size_average=None, reduce=False, reduction='none', pos_weight=None)


# optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
# shampooHps = ShampooHyperParams()
optimizer = Shampoo(model.parameters(), momentum = args.beta1, hyperparams=ShampooHyperParams(), lr=args.lr)
train_loss = []
test_loss = []
lrVec = np.concatenate([np.linspace(0,args.lr,args.warmup),np.linspace(args.lr,0,num_epochs-args.warmup+1)[1:]],axis=0)
# lrVec = np.concatenate([np.linspace(args.lr,args.lr,args.warmup),np.linspace(args.lr,args.lr,num_epochs-args.warmup+1)[1:]],axis=0)

debug = False

tot_size=0
if debug:
    num_epochs = 1

size=[]
denseness=[]

for epoch in range(num_epochs):
    tLtrain = time.time()
    model.train()
    loss_total = 0
    tot_size = 0
    for g in optimizer.param_groups:
        g['lr'] = lrVec[epoch]

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
        if epoch==0 and it==0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    preconditioner = state['preconditioner']
                    for i, stat in enumerate(preconditioner.statistics):
                        size.append(stat.size(0))
                        denseness.append([])
        loss_total += loss.item()*img.size(0)
        tot_size += img.size(0)
        print('loss: {} iteration: {}/{}'.format(loss.item(), it, numBatches))
        # for group in optimizer.param_groups:
        #     for p in group['params']:
        #         state = optimizer.state[p]
        #         preconditioner = state['preconditioner']
        #         for i, stat in enumerate(preconditioner.statistics):
        #             print("stat shape: " + str(stat.size()))
        #             print("denseness: " + str(preconditioner.dense[i]))
        #             print("avg. denseness: " + str(preconditioner.dense[i]/numBatches))
    # ===================log========================
    print('epoch [{}/{}], loss_total:{:.4f}, loss:{:.4f}'.format(epoch+1, num_epochs, loss_total/tot_size, loss.item()))
    print('epoch ' + str(epoch) + ' train time: '+str(time.time()-tLtrain))
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

    plt.plot([i for i in range(len(train_loss))], train_loss)
    plt.xlabel("epochs")
    plt.ylabel("train_loss")
    #plt.title("Shampoo optimizer")
    plt.savefig("train_loss" + description + ".png")

    plt.clf()
    plt.plot([i for i in range(len(test_loss))], test_loss)
    plt.xlabel("epochs")
    plt.ylabel("test_loss")
    #plt.title("Shampoo optimizer")
    plt.savefig("test_loss" + description + ".png")

    if args.quic:
        num=0
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                preconditioner = state['preconditioner']
                for i, stat in enumerate(preconditioner.statistics):
                    denseness[num].append(preconditioner.dense[i])
                    num+=1
                    # print("stat shape: " + str(stat.size()))
                    # print("denseness: " + str(preconditioner.dense[i]))
                    # print("avg. denseness: " + str(preconditioner.dense[i]/numBatches))
        plt.clf()
        plt.plot([i for i in range(len(denseness[0]))], denseness[0])
        plt.xlabel("epochs")
        plt.ylabel("denseness")
        plt.savefig("denseness" + description + ".png")

    torch.save(train_loss, "train_loss" + description + ".pt")
    torch.save(test_loss, "test_loss" + description + ".pt")
    torch.save(denseness, "denseness" + description + ".pt")

# plt.plot([i for i in range(len(train_loss))], train_loss)
# plt.xlabel("epochs")
# plt.ylabel("train_loss")
# #plt.title("Shampoo optimizer")
# plt.savefig("train_loss.png")

# plt.clf()
# plt.plot([i for i in range(len(test_loss))], test_loss)
# plt.xlabel("epochs")
# plt.ylabel("test_loss")
# #plt.title("Shampoo optimizer")
# plt.savefig("test_loss.png")

print("train_loss: " + str(train_loss))
print("test_loss: " + str(test_loss))
