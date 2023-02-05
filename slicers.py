import torch
import torch.nn as nn


class Hierarhcial_Slicer(nn.Module):
    def __init__(self,d,Ls,device='cuda',act='linear'):
        super(Hierarhcial_Slicer, self).__init__()
        self.d=d
        self.Ls=Ls
        self.device=device
        self.act= act
        self.U_list = nn.Sequential()
        for i in range(len(self.Ls)):
            if(i==0):
                self.U_list.add_module('linear%02d'%(i),nn.Linear(self.d,self.Ls[i],bias=False))
            else:
                self.U_list.add_module('linear%02d' % (i), nn.Linear(self.Ls[i-1], self.Ls[i], bias=False))
            if (i != len(self.Ls) - 1 and self.act == 'tanh'):
                self.U_list.add_module('activation%02d' % (i), nn.Tanh())
        self.reset()

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        out = self.U_list(x)
        return out
    def project_parameters(self):
        for U in self.U_list.modules():
            if isinstance(U, nn.Linear):
                U.weight.data = U.weight/torch.sqrt(torch.sum(U.weight**2,dim=1,keepdim=True))

    def reset(self):
        for U in self.U_list.modules():
            if isinstance(U, nn.Linear):
                U.weight.data = torch.randn(U.weight.shape, device=self.device)
        self.project_parameters()


class Hierarhcial_Boosted_Slicer(nn.Module):
    def __init__(self,d,Ls,device='cuda',act='linear'):
        super(Hierarhcial_Boosted_Slicer, self).__init__()
        self.d=d
        self.Ls=Ls
        self.device=device
        self.act= act
        self.U_list = []
        for i in range(len(self.Ls)):
            if(i==0):
                self.U_list.append(nn.Linear(self.d,self.Ls[i],bias=False))
            else:
                self.U_list.append(nn.Linear(self.Ls[i-1], self.Ls[i], bias=False))
        self.reset()

    def forward(self,x):
        out=[]
        x = x.view(x.shape[0],-1)
        for U in self.U_list:
            x = U(x)
            out.append(x)
        return out
    def project_parameters(self):
        for U in self.U_list:
            U.weight.data = U.weight/torch.sqrt(torch.sum(U.weight**2,dim=1,keepdim=True))

    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape, device=self.device)
        self.project_parameters()


class ConvSlicer(nn.Module):
    def __init__(self,L,ch=128,bottom_width=8,type='csw'):
        super(ConvSlicer, self).__init__()
        if(type=='cswold'):
            self.U1 = nn.Conv2d(ch, L, kernel_size=4, stride=2, padding=1, bias=False)
            if (bottom_width == 8):
                self.U2 = nn.Conv2d(L, L, kernel_size=4, stride=2, padding=1, bias=False, groups=L)
                self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2, self.U3]
            elif(bottom_width==6):
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2]
        elif(type=='csw'):
            if (bottom_width == 8):
                self.U1 = nn.Conv2d(ch, L, kernel_size=5, stride=1, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U1 = nn.Conv2d(ch, L, kernel_size=4, stride=1, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2]
        elif (type == 'csws'):
            if (bottom_width == 8):
                self.U1 = nn.Conv2d(ch, L, kernel_size=2, stride=2, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=2, stride=2, padding=0, bias=False, groups=L)
                self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U1 = nn.Conv2d(ch, L, kernel_size=2, stride=2, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2]
        elif (type == 'cswd'):
            if (bottom_width == 8):
                self.U1 = nn.Conv2d(ch, L, kernel_size=2, stride=1,dilation=4, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=2, stride=1,dilation=2, padding=0, bias=False, groups=L)
                self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U1 = nn.Conv2d(ch, L, kernel_size=2, stride=1,dilation=3, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2]
        self.reset()

    def forward(self, x):
        for U in self.U_list:
            x=U(x)
        return x

    def project_parameters(self):
        for U in self.U_list:
            U.weight.data = U.weight / torch.sqrt(torch.sum(U.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape,device='cuda')
        self.project_parameters()


class PRWConvSlicer(nn.Module):
    def __init__(self,ch=128,bottom_width=8,k=2,type='csw'):
        super(PRWConvSlicer, self).__init__()
        if(type=='cprwold'):
            self.U1 = nn.Conv2d(ch, k, kernel_size=4, stride=2, padding=1, bias=False)
            if (bottom_width == 8):
                self.U2 = nn.Conv2d(k, k, kernel_size=4, stride=2, padding=1, bias=False)
                self.U3 = nn.Conv2d(k, k, kernel_size=2, stride=1, padding=0, bias=False)
                self.U_list = [self.U1, self.U2, self.U3]
            elif(bottom_width==6):
                self.U2 = nn.Conv2d(k, k, kernel_size=3, stride=1, padding=0, bias=False)
                self.U_list = [self.U1, self.U2]
        elif(type=='cprw'):
            if (bottom_width == 8):
                self.U1 = nn.Conv2d(ch, k, kernel_size=5, stride=1, padding=0, bias=False)
                self.U2 = nn.Conv2d(k, k, kernel_size=3, stride=1, padding=0, bias=False)
                self.U3 = nn.Conv2d(k, k, kernel_size=2, stride=1, padding=0, bias=False)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U1 = nn.Conv2d(ch,k, kernel_size=4, stride=1, padding=0, bias=False)
                self.U2 = nn.Conv2d(k, k, kernel_size=3, stride=1, padding=0, bias=False)
                self.U_list = [self.U1, self.U2]
        elif (type == 'cprws'):
            if (bottom_width == 8):
                self.U1 = nn.Conv2d(ch, k, kernel_size=2, stride=2, padding=0, bias=False)
                self.U2 = nn.Conv2d(k, k, kernel_size=2, stride=2, padding=0, bias=False)
                self.U3 = nn.Conv2d(k, k, kernel_size=2, stride=1, padding=0, bias=False)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U1 = nn.Conv2d(ch, k, kernel_size=2, stride=2, padding=0, bias=False)
                self.U2 = nn.Conv2d(k, k, kernel_size=3, stride=1, padding=0, bias=False)
                self.U_list = [self.U1, self.U2]
        elif (type == 'cprwd'):
            if (bottom_width == 8):
                self.U1 = nn.Conv2d(ch, k, kernel_size=2, stride=1,dilation=4, padding=0, bias=False)
                self.U2 = nn.Conv2d(k, k, kernel_size=2, stride=1,dilation=2, padding=0, bias=False)
                self.U3 = nn.Conv2d(k, k, kernel_size=2, stride=1, padding=0, bias=False)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U1 = nn.Conv2d(ch, k, kernel_size=2, stride=1,dilation=3, padding=0, bias=False)
                self.U2 = nn.Conv2d(k, k, kernel_size=3, stride=1, padding=0, bias=False)
                self.U_list = [self.U1, self.U2]
        self.reset()

    def forward(self, x):
        for U in self.U_list:
            x=U(x)
        return x

    def project_parameters(self):
        for U in self.U_list:
            U.weight.data = U.weight / torch.sqrt(torch.sum(U.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape,device='cuda')
        self.project_parameters()
class NonLinearConvSlicer(nn.Module):
    def __init__(self,L,ch=128,bottom_width=8,type='ncsw',activation=nn.Sigmoid()):
        super(NonLinearConvSlicer, self).__init__()
        self.activation=activation
        if (type == 'ncswold'):
            self.U1 = nn.Conv2d(ch, L, kernel_size=4, stride=2, padding=1, bias=False)
            if (bottom_width == 8):
                self.U2 = nn.Conv2d(L, L, kernel_size=4, stride=2, padding=1, bias=False, groups=L)
                self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2]
        elif (type == 'ncsw'):
            if (bottom_width == 8):
                self.U1 = nn.Conv2d(ch, L, kernel_size=5, stride=1, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U1 = nn.Conv2d(ch, L, kernel_size=4, stride=1, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2]
        elif (type == 'ncsws'):
            if (bottom_width == 8):
                self.U1 = nn.Conv2d(ch, L, kernel_size=2, stride=2, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=2, stride=2, padding=0, bias=False, groups=L)
                self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U1 = nn.Conv2d(ch, L, kernel_size=2, stride=2, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2]
        elif (type == 'ncswd'):
            if (bottom_width == 8):
                self.U1 = nn.Conv2d(ch, L, kernel_size=2, stride=1, dilation=4, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=2, stride=1, dilation=2, padding=0, bias=False, groups=L)
                self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2, self.U3]
            elif (bottom_width == 6):
                self.U1 = nn.Conv2d(ch, L, kernel_size=2, stride=1, dilation=3, padding=0, bias=False)
                self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
                self.U_list = [self.U1, self.U2]
        self.reset()

    def forward(self, x):
        for i in  range(len(self.U_list)-1):
            x = self.activation(self.U_list[i](x))
        x=self.U_list[-1](x)
        return x

    def project_parameters(self):
        for U in self.U_list:
            U.weight.data = U.weight / torch.sqrt(torch.sum(U.weight ** 2, dim=[1, 2, 3], keepdim=True))

    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape, device='cuda')
        self.project_parameters()
class Conv_MNIST_Slicer(nn.Module):
    def __init__(self,L):
        super(Conv_MNIST_Slicer, self).__init__()
        image_sizes=(1,28,28)
        self.U1 = nn.Conv2d(image_sizes[0], 1*L, kernel_size=4, stride=2, padding=1, bias=False)
        self.U2 = nn.Conv2d(1*L, 1 * L, kernel_size=4, stride=2, padding=1, bias=False,groups=L)
        self.U3 = nn.Conv2d(1*L, 1 * L, kernel_size=7, stride=1, padding=0, bias=False, groups=L)
        self.U1.weight.data = torch.randn(self.U1.weight.shape).cuda()
        self.U2.weight.data = torch.randn(self.U2.weight.shape).cuda()
        self.U3.weight.data = torch.randn(self.U3.weight.shape).cuda()
        self.project_parameters()

    def forward(self, x):
        self.project_parameters()
        return self.U3(self.U2(self.U1(x)))

    def project_parameters(self):
        self.U1.weight.data = self.U1.weight / torch.sqrt(torch.sum(self.U1.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U2.weight.data = self.U2.weight / torch.sqrt(torch.sum(self.U2.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U3.weight.data = self.U3.weight / torch.sqrt(torch.sum(self.U3.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        self.U1.weight.data = torch.randn(self.U1.weight.shape).cuda()
        self.U2.weight.data = torch.randn(self.U2.weight.shape).cuda()
        self.U3.weight.data = torch.randn(self.U3.weight.shape).cuda()
        self.project_parameters()
