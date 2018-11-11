import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torchvision
import argparse
import torch.nn.functional as F

g_input_size = 1     # Random noise dimension coming into generator, per output vector
g_hidden_size = 256  # Generator complexity
g_output_size = 784    # size of generated output vector
d_input_size = 784 # Minibatch size - cardinality of distributions
d_hidden_size = 256  # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
parser = argparse.ArgumentParser()

parser.add_argument('--batchsize',type=int,default=32)
#parser.add_argument('--learningrate',type=float,default=0.01)
parser.add_argument('--numepoches',type=int,default=20)
opt = parser.parse_args()
batch_sizes=opt.batchsize
#learning_rate=opt.learningrate
num_epoches=opt.numepoches

data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size,output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=self.gen(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size,output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=self.dis(x)
        return x


D=Discriminator(d_input_size,d_hidden_size,d_output_size)
G=Generator(g_input_size,g_hidden_size,g_output_size)
'''
if torch.cuda.is_available():
    D=D.cuda()
G=Generator(g_input_size,g_hidden_size,g_output_size)
if torch.cuda.is_available():
    G=G.cuda()
'''
cirterion=nn.BCELoss()
d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0003)
g_optimizer=torch.optim.Adam(G.parameters(),lr=0.0003)
#num_img=img.size(0)
#print(num_img)
d_steps=10
g_step=10
label = torch.FloatTensor(32)
label = Variable(label)
Tensor = torch.FloatTensor
for epoch in range(8):
    for i, (img, _) in enumerate(train_loader):
        #img, _ = data_tf
        #num_img = img.size()
        #img=img.view(num_img,-1)
        real_img = Variable(img.type(Tensor))
        #real_img=Variable(img)
        real_img=real_img.view(32,-1)
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

        z=Variable(torch.randn(32,1))
        fake_img=G(z)
        fake_out=D(fake_img)
        d_loss_fake=cirterion(fake_out,fake)
        fake_score=fake_out

        d_loss=d_loss_fake+d_loss_real
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        if i%40==0:
            print(d_loss[0])
        z = Variable(torch.randn(32,1))
        fake_img=G(z)
        output=D(fake_img)
        g_loss=cirterion(output,valid)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if i%40==0:
            print(g_loss[0])
