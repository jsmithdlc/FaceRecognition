
import torch
from torch import nn
from torch.nn import functional as F


class Conv2dBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding = 0):
        super(Conv2dBlock,self).__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size,
                              stride = stride, padding = padding, bias = False)

        self.bn = nn.BatchNorm2d(output_channels, 
                                 eps=0.001,      # default tensorflow value
                                 momentum = 0.1, # default pytorch value
                                 )

    def forward(self, x):
        x = self.conv(x)

        x = self.bn(x)

        x = F.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35,self).__init__()

        self.scale = scale

        self.branch0 = Conv2dBlock(256, 32, 1, stride = 1)

        self.branch1 = nn.Sequential(Conv2dBlock(256,32,1,stride = 1),
                                      Conv2dBlock(32,32,3,stride = 1, padding = 1))

        self.branch2 = nn.Sequential(Conv2dBlock(256,32,1,stride = 1),
                                      Conv2dBlock(32,32,3,stride = 1, padding = 1),
                                      Conv2dBlock(32,32,3,stride = 1, padding = 1))

        self.conv2d = nn.Conv2d(96, 256, kernel_size = 1)

        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = torch.cat((self.branch0(x_in),self.branch1(x_in),self.branch2(x_in)),dim = 1) 
        x = self.conv2d(x)
        x = x_in + x * self.scale
        x = self.relu(x)
        return x


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17,self).__init__()

        self.scale = scale

        self.branch0 = Conv2dBlock(896, 128, 1, stride = 1)

        self.branch1 = nn.Sequential(Conv2dBlock(896,128,1,stride = 1),
                                      Conv2dBlock(128,128,(1,7),stride = 1, padding = (0,3)),
                                      Conv2dBlock(128,128,(7,1),stride = 1, padding = (3,0)))

        self.conv2d = nn.Conv2d(256, 896, 1)

        self.relu = nn.ReLU()

    def forward(self,x_in):
        x = torch.cat((self.branch0(x_in), self.branch1(x_in)),dim=1)
        x = self.conv2d(x)
        x = x_in + x*self.scale
        x = self.relu(x)
        return x

class Block8(nn.Module):

    def __init__(self, scale=1.0, activate = True):
        super(Block8,self).__init__()
        self.activate = activate
        self.scale = scale

        self.branch0 = Conv2dBlock(1792, 192, 1, stride = 1)

        self.branch1 = nn.Sequential(Conv2dBlock(1792,192,1,stride = 1),
                                      Conv2dBlock(192,192,(1,3),stride = 1, padding = (0,1)),
                                      Conv2dBlock(192,192,(3,1),stride = 1, padding = (1,0)))

        self.conv2d = nn.Conv2d(384, 1792, 1)

        self.relu = nn.ReLU()

    def forward(self,x_in):
        x = torch.cat((self.branch0(x_in),self.branch1(x_in)),dim=1)
        x = self.conv2d(x)
        x = x_in + x*self.scale
        if self.activate:
            x = self.relu(x)
        return x


class Mixed_6a(nn.Module):
    "Input C = 256"
    def __init__(self):
        super(Mixed_6a,self).__init__()
        self.branch0 = Conv2dBlock(256,384,3, stride=2)

        self.branch1 = nn.Sequential(Conv2dBlock(256,192,1,stride = 1),
                                         Conv2dBlock(192,192,3,stride = 1, padding = 1),
                                         Conv2dBlock(192,256,3,stride = 2))

        self.branch2 = nn.MaxPool2d(3, stride = 2)


    def forward(self,x):
        x = torch.cat((self.branch0(x),self.branch1(x),self.branch2(x)),dim=1)
        return x

class Mixed_7a(nn.Module):
    "Input C = 896"
    def __init__(self):
        super(Mixed_7a,self).__init__()
        self.branch0 = nn.Sequential(Conv2dBlock(896,256,1,stride = 1),
                                      Conv2dBlock(256,384,3,stride = 2))

        self.branch1 = nn.Sequential(Conv2dBlock(896,256,1,stride = 1),
                                      Conv2dBlock(256,256,3,stride = 2))

        self.branch2 = nn.Sequential(Conv2dBlock(896,256,1,stride = 1),
                                      Conv2dBlock(256,256,3,stride = 1,padding=1),
                                      Conv2dBlock(256,256,3,stride = 2))

        self.branch3 = nn.MaxPool2d(3, stride = 2)

    def forward(self,x):
        x = torch.cat((self.branch0(x),self.branch1(x),self.branch2(x),self.branch3(x)),dim=1)
        return x




class InceptionModelV1(nn.Module):
    "Inception Model V1 !!"
    def __init__(self):
        super(InceptionModelV1,self).__init__()
        
        self.conv2d_1a = Conv2dBlock(3,32,3,stride=2)
        self.conv2d_2a = Conv2dBlock(32,32,3,stride=1)
        self.conv2d_2b = Conv2dBlock(32,64,3,stride=1,padding=1)
        self.maxpool_3a = nn.MaxPool2d(3,stride=2)
        self.conv2d_3b = Conv2dBlock(64,80,1,stride=1)
        self.conv2d_4a = Conv2dBlock(80,192,3,stride=1)
        self.conv2d_4b = Conv2dBlock(192,256,3,stride=2)
        self.repeat_1 = nn.Sequential(Block35(scale=0.17),
                                      Block35(scale=0.17),
                                      Block35(scale=0.17),
                                      Block35(scale=0.17),
                                      Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(Block17(scale=0.10),
                                      Block17(scale=0.10),
                                      Block17(scale=0.10),
                                      Block17(scale=0.10),
                                      Block17(scale=0.10),
                                      Block17(scale=0.10),
                                      Block17(scale=0.10),
                                      Block17(scale=0.10),
                                      Block17(scale=0.10),
                                      Block17(scale=0.10))
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(Block8(scale=0.8),
                                      Block8(scale=0.8),
                                      Block8(scale=0.8),
                                      Block8(scale=0.8),
                                      Block8(scale=0.8))
        self.block8 = Block8(activate=False)

        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.8)
        self.last_linear = nn.Linear(1792,512,bias=False)
        self.last_bn = nn.BatchNorm1d(512,eps=0.001, momentum=0.1)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0],-1))
        x = self.last_bn(x)
        x = F.normalize(x,p=2,dim=1)
        return x

def load_weights(model, path):
    state_dict = torch.load(path)
    del state_dict['logits.weight']
    del state_dict['logits.bias']
    model.load_state_dict(state_dict)
    return model
    



def test_block35():
    t_test = torch.randn(4,256,128,128)
    model = Block35()
    output = model(t_test)
    assert output.shape == (4,256,128,128)

def test_block17():
    t_test = torch.randn(4,896,128,128)
    model = Block17()
    output = model(t_test)
    assert output.shape == (4,896,128,128)

def test_block8():
    t_test = torch.randn(4,1792,32,32)
    model = Block8()
    output = model(t_test)
    assert output.shape == (4,1792,32,32)


def test_mixed6a():
    t_test = torch.randn(4,256,32,32)
    model = Mixed6a()
    output = model(t_test)
    assert output.shape == (4,896,15,15)

def test_mixed7a():
    t_test = torch.randn(4,896,32,32)
    model = Mixed7a()
    output = model(t_test)
    assert output.shape == (4,1792,15,15)

def test_Inception():
    t_test = torch.randn(2,3,149,149)
    model = InceptionModelV1()
    output = model(t_test)
    assert output.shape == (2,512)
    print(output[0,:])



if __name__ == '__main__':
    model = InceptionModelV1()
    load_weights(model, "/home/javier/Ramblings/FaceRecognition/models/20180402-114759-vggface2.pt")

