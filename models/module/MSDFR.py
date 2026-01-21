import torch
import torch.nn as nn
import torch.nn.functional as F

class TransferConv_l(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        if resnet:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c//4, in_c // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )

        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output


class TransferConv_m(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        if resnet:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output

class TransferConv_d(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        if resnet:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output


class TransferConv_h(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1,padding=0),
            nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output
    
class TransferConv_a(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        if resnet:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output

class MSDFR(nn.Module):
    def __init__(self, resnet, in_c, args):
        super().__init__()
        self.args = args
        self.transferconv_h = TransferConv_h(in_c)
        self.transferconv_m = TransferConv_m(resnet, in_c)
        self.transferconv_l = TransferConv_l(resnet, in_c)
        self.transferconv_d = TransferConv_d(resnet, in_c)
        self.transferconv_a = TransferConv_a(resnet, in_c)

    def softmax_weight_adjustment(self, f_h, f_l):
        f_h = f_h.view(f_h.size(0), f_h.size(1), -1)
        f_l = f_l.view(f_l.size(0), f_l.size(1), -1)

        weights_h_l = torch.exp(f_h) / (torch.exp(f_h) + torch.exp(f_l))
        weights_l_h = torch.exp(f_l) / (torch.exp(f_h) + torch.exp(f_l))
        return weights_h_l, weights_l_h

    def forward(self, f_4, f_3, f_2):
        f_4 = self.transferconv_h(f_4)
        f_3 = self.transferconv_m(f_3)
        f_2 = self.transferconv_l(f_2)
        f_2_ = self.transferconv_a(f_2)

        weights_4_2, weights_2_4 = self.softmax_weight_adjustment(f_4, f_2_)
        weights_3_2, weights_2_3 = self.softmax_weight_adjustment(f_3, f_2)

        f_2 = f_2.view(f_2.shape[0], f_2.shape[1], -1)
        f_3 = f_3.view(f_3.shape[0], f_3.shape[1], -1)
        f_d = 0.5 * (weights_3_2 * f_3 + weights_2_3 * f_2) + 0.5 * f_3 
        f_d = f_d.view(f_3.size(0), f_3.size(1), 10, 10)
        f_d = self.transferconv_d(f_d)
        f_2_ = f_2_.view(f_2.shape[0], f_2.shape[1], -1)
        f_4 = f_4.view(f_4.shape[0], f_4.shape[1], -1)
        f_h = 0.5*(weights_4_2 * f_4 + weights_2_4 * f_2_) + 0.5 * f_4
        f_h = f_h.view(f_4.size(0), f_4.size(1), 5, 5)
        return f_h, f_d
        
        
        

        
        
