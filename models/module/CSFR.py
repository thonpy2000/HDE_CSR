import torch
import torch.nn as nn
import torch.nn.functional as F
from .TDM import TDM


class MLP(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        self.resnet = resnet
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        if self.resnet:
            self.liner_layer = nn.Sequential(
                nn.Linear(in_features=in_c,
                          out_features=in_c * 1,
                          bias=False),
                nn.BatchNorm1d(in_c * 1),
                nn.ReLU(),
                nn.Linear(in_features=in_c * 1,
                          out_features=in_c,
                          bias=False)
            )
        else:
            self.liner_layer = nn.Sequential(
                nn.Linear(in_features=in_c,
                          out_features=in_c * 2,
                          bias=False),
                nn.BatchNorm1d(in_c * 2),
                nn.ReLU(),
                nn.Linear(in_features=in_c * 2,
                          out_features=in_c,
                          bias=False)
            )
    def add_noise(self, x):
        if self.training:
            noise = ((torch.rand(x.shape).to(x.device) - .5) * 2) * 0.2
            x = x + noise
            x = x.clamp(min=0., max=2.)
        return x
    def forward(self,x):
        output = self.gmp(x)
        output = output.view(output.size(0), -1)
        output = self.liner_layer(output)
        output = torch.tanh(output)
        output = 1 + output
        output = self.add_noise(output)

        return output

class FeatureAdjustment(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.in_c = in_c
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.channel_act = nn.Sequential(
            nn.Linear(in_features=in_c,
                      out_features=in_c // 2),
            nn.BatchNorm1d(in_c // 2),
            nn.ReLU(),
            nn.Linear(in_features=in_c // 2,
                      out_features=in_c // 2),
        )


    def forward(self, x, q, s):
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        output_ca = self.channel_act(x)
        output_ca = 1 + torch.tanh(output_ca)
        output_ca = output_ca.unsqueeze(dim=-1).unsqueeze(dim=-1)
        q = q * output_ca

        return q

class CSFR(nn.Module):
    def __init__(self,resnet, in_c):
        super().__init__()
        self.mlp_a = MLP(resnet, in_c)
        self.adjustment_a = FeatureAdjustment(in_c * 2)
        self.tdm = TDM(resnet)

    def forward(self, f_refine_a, way, shot):
        w_a = self.mlp_a(f_refine_a).unsqueeze(dim=-1).unsqueeze(dim=-1)
        
        f_refine_a = f_refine_a * w_a

        _, c, h, w = f_refine_a.shape
        m = h * w
        support_a = f_refine_a[:way * shot].view(way, shot, c, m)
        centroid_a = support_a.mean(dim=1).unsqueeze(dim=1).view(-1, 1, c, m)  # 30, 1, c, m
        query_a = f_refine_a[way * shot:].view(-1, 1, c, m) #   原来 1,450, c, m 
        query_num = query_a.shape[0]
        weight = self.tdm(support_a, query_a).unsqueeze(-1)
        query_a = query_a.permute(1, 0, 2, 3)

        zero_s = torch.zeros([1, query_num, c, m]).cuda()
        zero_q = torch.zeros([way, 1, c, m]).cuda()

        centroid_a = ((centroid_a + zero_s)* weight).view(-1, c, h, w) 
        query_a = ((query_a + zero_q) * weight).view(-1, c, h, w)

        cross_sample_a = torch.cat((centroid_a, query_a), 1)

        query_a = self.adjustment_a(cross_sample_a, query_a, centroid_a)

        centroid_a = centroid_a.view(way, query_num, -1)
        query_a = query_a.view(way, query_num, -1)

        return centroid_a, query_a





