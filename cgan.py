import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torchvision
import argparse
import torch.nn.functional as F


parser = argparse.ArgumentParser()

parser.add_argument('--batchsize',type=int,default=32)
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--learningrate',type=float,default=0.0003)
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')

opt = parser.parse_args()
print(opt)
batch_sizes=opt.batchsize
learning_rate=opt.learningrate
num_epoches=opt.n_epochs
imgsize=opt.img_size
imgchannel=opt.channels

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf)
train_loader=DataLoader(train_dataset,batch_size=batch_sizes,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_sizes,shuffle=False)
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

        self.fc1=nn.Linear(3136,32)
        self.relu=nn.LeakyReLU(0.2,True)
        self.fc2=nn.Linear(32,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        #print(x.size())
        x=x.view(x.size(0),-1)
        #print(x.size())
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
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
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1,50,3,stride=1,padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2=nn.Sequential(
            nn.Upsample(scale_factor=2),
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
        #print(x.size())# 32 1 56 56
        x=self.downsample1(x)
        x=self.downsample2(x)
        x=self.downsample3(x)# 1 28 28
        return x

cuda = True if torch.cuda.is_available() else False
print('cuda',cuda)


if cuda:
    Generator().cuda()
    Discriminator().cuda()

D=Discriminator().cuda()
G=Generator().cuda()


G.apply(weights_init_normal)
D.apply(weights_init_normal)

cirterion=nn.BCELoss()
cirterion.cuda()

d_optimizer=torch.optim.Adam(D.parameters(),lr=learning_rate)
g_optimizer=torch.optim.Adam(G.parameters(),lr=learning_rate)
#num_img=img.size(0)
#print(num_img)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
for epoch in range(num_epoches):
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
        #print('img_size(0)',img.size(0))
        valid = Variable(Tensor(img.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(img.size(0), 1).fill_(0.0), requires_grad=False)

        #label.data.fill_(fake_label)
        d_loss_real=cirterion(real_out,valid)
        real_score=real_out
        num_pic=int(imgsize/2)

        z=Variable(torch.randn(batch_sizes,imgchannel,num_pic,num_pic))
        z = Variable(z.type(Tensor))
        #print('z.size',z.size())
        fake_img=G(z)
        #print(fake_img.size())#torch.Size([32, 1, 28, 28])
        #print('fakeIng',fake_img.size())
        fake_out=D(fake_img)
        #print(fake_out.size())

        d_loss_fake=cirterion(fake_out,fake)
        fake_score=fake_out

        d_loss=d_loss_fake+d_loss_real
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        #if i%40==0:
            #print('d_loss',d_loss[0])
        #z = Variable(torch.randn(3136))
        fake_img=G(z)

        output=D(fake_img)
        g_loss=cirterion(output,valid)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if i%40==0:
            #print('g_loss',g_loss[0])
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(train_loader),
                                                            d_loss.item(), g_loss.item()))
