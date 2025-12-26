import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import Conv_4, ResNet
import math
from models.module.MSDFR import MSDFR
from models.module.CSFR import CSFR


class HDE_CSR_Net(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet
        self.resolution = 5*5

        if self.resnet:
            self.num_channel = 640
            self.dim = 640 * 25
            self.feature_extractor = ResNet.resnet12(drop=True)
        else:
            self.num_channel = 64
            self.dim = 64 * 25
            self.feature_extractor = Conv_4.BackBone(self.num_channel)

        self.msdfr = MSDFR(self.resnet, self.num_channel)
        self.csfr_high = CSFR(self.resnet, self.num_channel)
        self.csfr_mid = CSFR(self.resnet, self.num_channel)

        self.scale_1 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.scale_2 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def get_feature_vector(self, inp):
        f_4, f_3, f_2 = self.feature_extractor(inp)

        return f_4, f_3, f_2

    def get_neg_l2_dist(self, inp, way, shot, query_shot):

        f_4, f_3, f_2 = self.get_feature_vector(inp)
        f_g, f_d = self.msdfr(f_4, f_3, f_2)
        centroid_g, query_g = self.csfr_high(f_g, way, shot)
        centroid_d, query_d = self.csfr_mid(f_d, way, shot)

        l2_dist_g = torch.sum(torch.pow(centroid_g - query_g, 2), dim=-1).transpose(0, 1)
        neg_l2_dist_g = l2_dist_g.neg()

        l2_dist_d = torch.sum(torch.pow(centroid_d - query_d, 2), dim=-1).transpose(0, 1)
        neg_l2_dist_d = l2_dist_d.neg()

        return neg_l2_dist_g, neg_l2_dist_d
 
    def meta_test(self, inp, way, shot, query_shot):

        neg_l2_dist_g, neg_l2_dist_d = self.get_neg_l2_dist(inp=inp,
                                                            way=way,
                                                            shot=shot,
                                                            query_shot=query_shot
                                                            )
        neg_l2_dist_all = 0.5*neg_l2_dist_g + 0.5*neg_l2_dist_d
        _, max_index = torch.max(neg_l2_dist_all, 1)

        return max_index

    def forward(self, inp):

        neg_l2_dist_g, neg_l2_dist_d = self.get_neg_l2_dist(inp=inp,
                                                            way=self.way,
                                                            shot=self.shots[0],
                                                            query_shot=self.shots[1]
                                                            )
        logits_g = neg_l2_dist_g / self.dim * self.scale_1
        logits_d = neg_l2_dist_d / self.dim * self.scale_2

        log_prediction_g = F.log_softmax(logits_g, dim=1)
        log_prediction_d = F.log_softmax(logits_d, dim=1)

        return log_prediction_g, log_prediction_d