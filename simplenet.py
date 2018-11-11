import torch
#import torch.nn as nn
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import argparse
import torch.nn.functional as F

class simple_net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simple_net,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),
                                  nn.BatchNorm1d(n_hidden_1),
                                  nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),
                                  nn.BatchNorm1d(n_hidden_2),
                                  nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(n_hidden_2,out_dim),
                                  nn.BatchNorm1d(out_dim),
                                  nn.ReLU(True))
    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

class cnn_net(nn.Module):
    def __init__(self,num_class):
        super(cnn_net,self).__init__()
        self.layer1=nn.Sequential(
            #1 28 28
            nn.Conv2d(1,16,kernel_size=3),
            #16 26 26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3),
            #32 24 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
            #32 12 12
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            #64 10 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3),
            #128 8 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
            #128 4 4
        )
        self.fc=nn.Sequential(
            nn.Linear(128*4*4,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,num_class)
        )
    def forward(self, x):
        x=self.layer1(x)
        #print(x.size())
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*320
        #print(x.size()) ([64, 20, 4, 4])
        x = x.view(in_size, -1) # flatten the tensor
        # print(x.size()) ([64, 320])
        #  x: 64*10
        x = self.fc(x)
        #print(x.size()) ([64, 10])
        return F.log_softmax(x)


parser = argparse.ArgumentParser()
parser.add_argument('--network',default='Net',help='type of network')
parser.add_argument('--batchsize',type=int,default=64)
parser.add_argument('--learningrate',type=float,default=0.01)
parser.add_argument('--numepoches',type=int,default=100)
opt = parser.parse_args()
batch_sizes=opt.batchsize
learning_rate=opt.learningrate
num_epoches=opt.numepoches
network=opt.network
print(opt)
#print(batch_sizes,learning_rate,num_epoches)
data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf)
train_loader=DataLoader(train_dataset,batch_size=batch_sizes,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_sizes,shuffle=False)


if network=='simple_net':
    print(network)
    model1=simple_net(28*28,300,300,10)
    if torch.cuda.is_available():
        model1=model1.cuda()
    criterion=nn.CrossEntropyLoss()
    optimer=optim.SGD(model1.parameters(),lr=0.001, momentum=0.9)
    lr=learning_rate

    model1.train(mode=True)
    train_loss=0
    num=0
    for epoch in range(num_epoches):
        for data_tf in train_loader:
            num+=1
            img,label=data_tf
            optimer.zero_grad()
            img=img.view(img.size(0),-1)
            img=Variable(img)
            label=Variable(label)
            out=model1(img)
            loss=criterion(out,label)
            loss.sum().backward()
            if num%20==0:
                print(float(loss.data))
            optimer.step()
        print(loss.data)
elif network=='Net':
    print(network)
    model2=Net()
    if torch.cuda.is_available():
        model2 = model2.cuda()
    criterion = nn.CrossEntropyLoss()
    optimer = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
    lr = learning_rate

    model2.train(mode=True)
    train_loss = 0
    num = 0
    for epoch in range(num_epoches):
        for batch_idx, (img, label) in enumerate(train_loader):
                num += 1
               # img, label = data_tf
                optimer.zero_grad()
                # print(img.size(),label.size())
                # torch.Size([64, 1, 28, 28]) torch.Size([64])
                #img = img.view(img.size(0), -1)
                # print(img.size())
                # torch.Size([64, 784])
                if torch.cuda.is_available():
                    img = Variable(img).cuda()
                    label = Variable(label).cuda()
                else:
                    img = Variable(img)
                    label = Variable(label)
                #print(img.size())
                out = model2(img)
                loss = criterion(out, label)
                loss.sum().backward()
                if num % 20 == 0:
                    print(float(loss.data))
                optimer.step()
        print(loss.data)

elif network=='cnn_net':
    print(network)
    model2=cnn_net(10)
    if torch.cuda.is_available():
        model2 = model2.cuda()
    criterion = nn.CrossEntropyLoss()
    optimer = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
    lr = learning_rate

    model2.train(mode=True)
    train_loss = 0
    num = 0
    for epoch in range(num_epoches):
        for batch_idx, (img, label) in enumerate(train_loader):
                num += 1
               # img, label = data_tf
                optimer.zero_grad()
                # print(img.size(),label.size())
                # torch.Size([64, 1, 28, 28]) torch.Size([64])
                #img = img.view(img.size(0), -1)
                # print(img.size())
                # torch.Size([64, 784])
                if torch.cuda.is_available():
                    img = Variable(img).cuda()
                    label = Variable(label).cuda()
                else:
                    img = Variable(img)
                    label = Variable(label)
                #print(img.size())
                out = model2(img)
                loss = criterion(out, label)
                loss.sum().backward()
                if num % 20 == 0:
                    print(float(loss.data))
                optimer.step()
        print(loss.data)


