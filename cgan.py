import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torchvision
import argparse
import torch.nn.functional as F


parser = argparse.ArgumentParser()

parser.add_argument('--batchsize',type=int,default=1)
#parser.add_argument('--learningrate',type=float,default=0.01)
parser.add_argument('--numepoches',type=int,default=20)
opt = parser.parse_args()
batch_sizes=opt.batchsize
#learning_rate=opt.learningrate
num_epoches=opt.numepoches

data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf)
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,5,padding=2),
            nn.LeakyReLU(0.2,True),
            nn.AvgPool2d(2,stride=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,5,padding=2),
            nn.LeakyReLU(0.2,True),
            nn.AvgPool2d(2,stride=2)
        )
        '''
        self.fc=nn.Sequential(
            nn.Linear(3136,1),
            nn.LeakyReLU(0.2,True),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
        '''
        self.fc=nn.Linear(3136,1)
        self.relu=nn.LeakyReLU(0.2,True)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        #print(x.size())
        x=x.view(x.size(0),-1)
       # print(x.size())
        x=self.fc(x)
        x=self.relu(x)
        x=self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()
        '''
        self.fc=nn.Sequential(
            nn.Linear(input_size,num_feature)#1*56*56
        )
        self.br=nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        '''
        self.downsample1=nn.Sequential(
            nn.Conv2d(1,50,3,stride=1,padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2=nn.Sequential(
            nn.Conv2d(50,25,3,stride=1,padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3=nn.Sequential(
            nn.Conv2d(25,1,2,stride=2),
            nn.Tanh()
        )
    def forward(self, x):
        '''
        x=self.fc(x)
        print('x.size',x.size())
        print('x.size(0)',x.size(0))
        x=x.view(x.size(0),1,28,28)
        x=self.br(x)
        '''
        x=self.downsample1(x)
        x=self.downsample2(x)
        x=self.downsample3(x)# 1 28 28
        return x


D=Discriminator()
G=Generator()
'''
if torch.cuda.is_available():
    D=D.cuda()
if torch.cuda.is_available():
    G=G.cuda()
'''
cirterion=nn.BCELoss()
d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0003)
g_optimizer=torch.optim.Adam(G.parameters(),lr=0.0003)
#num_img=img.size(0)
#print(num_img)
Tensor = torch.FloatTensor
for epoch in range(8):
    for i, (img, _) in enumerate(train_loader):
        #num_img = img.size()
        #img=img.view(num_img,-1)
        real_img = Variable(img.type(Tensor))
        #real_img=Variable(img)
        #real_img=real_img.view(32,-1)
        #print(real_img.size())
        real_out=D(real_img)
        #print(real_out.size())
        #print(label.size())
        #print((img.size(0)))
        valid = Variable(Tensor(img.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(img.size(0), 1).fill_(0.0), requires_grad=False)

#        label.data.fill_(fake_label)
        d_loss_real=cirterion(real_out,valid)
        real_score=real_out

        z=Variable(torch.randn(1,1,56,56))
        #print('z.size',z.size())
        fake_img=G(z)
        #print('fakeIng',fake_img.size())
        fake_out=D(fake_img)
        d_loss_fake=cirterion(fake_out,fake)
        fake_score=fake_out

        d_loss=d_loss_fake+d_loss_real
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        if i%40==0:
            print('d_loss',d_loss[0])
        #z = Variable(torch.randn(3136))
        fake_img=G(z)
        output=D(fake_img)
        g_loss=cirterion(output,valid)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if i%40==0:
            print('g_loss',g_loss[0])
