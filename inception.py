import torch
import torch.nn as nn
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import argparse
import torch.nn.functional as F

class BasicConv2d(nn.Module):#conv+bn+relu
    def __init__(self,in_channel,out_channel,**kwargs):#**kwargs,任意个关键字参数，为字典
        super(BasicConv2d,self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,bias=False,**kwargs)
        self.bn=nn.BatchNorm2d(out_channel,eps=0.001)
        def forward(self,x):
            x=self.conv(x)
            x=self.bn(x)
            return F.relu(x,inplace=True)
class Inception(nn.Module):
    def __init__(self,in_channle,out_channle,pool_feature):
        super(Inception,self).__init__()
        self.branch1x1=BasicConv2d(in_channle,64,kernel_size=1)
        self.branch5x5_1=BasicConv2d(in_channle,48,kernel_size=1)
        self.branch5x5_2=BasicConv2d(48,64,kernel_size=5,padding=2)
        self.branch3x3_1=BasicConv2d(in_channle,64,kernel_size=1)
        self.branch3x3_2=BasicConv2d(64,96,kernel_size=3,padding=1)
        self.branch3x3_3=BasicConv2d(96,96,kernel_size=3,padding=1)
        self.branchpool=BasicConv2d(in_channle,pool_feature,kernel_size=1)
    def forward(self, x):
        branch1x1=self.branch1x1(x)
        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)
        branch3x3=self.branch3x3_1(x)
        branch3x3=self.branch3x3_2(branch3x3)
        branch3x3=self.branch3x3_3(branch3x3)
        branchpool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branchpool=self.branchpool(branchpool)
        outputs=[branch1x1,branch3x3,branch5x5,branchpool]
        return  torch.cat(outputs,1)

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        self.layer1=nn.Sequential(
            #input 224 224 3
            nn.Conv2d(3,64,7,2,padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(32,192,3,1,padding=1),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Inception(96,64),
            Inception(64,120),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Inception(60,128),
            Inception(128,128),
            Inception(128,128),
            Inception(128,132),
            Inception(132,208),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Inception(104,208),
            Inception(208,256),
            nn.AvgPool2d(kernel_size=7,stride=7,padding=1),
        )
        inpt = Input(shape=(224, 224, 3))
        # padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = Inception(x, 64)  # 256
        x = Inception(x, 120)  # 480
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = Inception(x, 128)  # 512
        x = Inception(x, 128)
        x = Inception(x, 128)
        x = Inception(x, 132)  # 528
        x = Inception(x, 208)  # 832
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = Inception(x, 208)
        x = Inception(x, 256)  # 1024
        x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
        x = Dropout(0.4)(x)
        x = Dense(1000, activation='relu')(x)
        x = Dense(1000, activation='softmax')(x)
        model = Model(inpt, x, name='inception')
        return model


parser = argparse.ArgumentParser()
parser.add_argument('--network',default='cnn_net',help='type of network')
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
            #print(img.size(),label.size())
            #torch.Size([64, 1, 28, 28]) torch.Size([64])
            #print(img.size())
            img=img.view(img.size(0),-1)
            #print(img.size())
            #torch.Size([64, 784])
            if torch.cuda.is_available():
                img=Variable(img).cuda()
                label=Variable(label).cuda()
            else:
                img=Variable(img)
                label=Variable(label)
            #print(img.size())
            out=model1(img)
            loss=criterion(out,label)
            loss.sum().backward()
            if num%20==0:
                print(float(loss.data))
            optimer.step()
        print(loss.data)