import torch
import torch.nn as nn
import torch.nn.functional as F



class SandGlassBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_c,
                                 out_features=in_c * 2,
                                 bias=False)
        self.bn1 = nn.BatchNorm1d(in_c * 2)
        self.linear2 = nn.Linear(in_features=in_c * 2,
                                 out_features=in_c,
                                 bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.linear1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.linear2(output)
        output = torch.tanh(output)
        output = 1 + output

        return output

class TDM(nn.Module):

    def __init__(self, resnet):

        super().__init__()
        self.resnet = resnet
        if self.resnet:
            self.in_c = 640
        else:
            self.in_c = 64

        self.prt_self = SandGlassBlock(self.in_c)
        self.prt_other = SandGlassBlock(self.in_c)
        self.qry_self = SandGlassBlock(self.in_c)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def add_noise(self, input):
        if self.training:
            noise = ((torch.rand(input.shape).to(input.device) - .5) * 2) * 0.2
            input = input + noise
            input = input.clamp(min=0., max=2.)

        return input

    # def adjusted_prototype(self, way, shot, spt):
    #     """
    #     调整支持集原型的计算，降低低质量特征样本的影响
        
    #     参数:
    #         spt (Tensor): 支持集特征张量，形状为 [way, shot, feature_dim]
        
    #     返回:
    #         Tensor: 调整后的原型，形状为 [way, feature_dim]
    #     """
    #     spt = spt.view(way, shot, -1)
        
        
    #     # 1. 计算常规原型 (所有样本的平均)
    #     prototype_general = spt.mean(dim=1)  # [way, feature_dim]
        
    #     # 处理shot=1的情况（无需调整）
    #     if shot == 1:
    #         return prototype_general.view(way, self.in_c, -1)
        
    #     # 2. 计算每个样本的排除自身原型
    #     # 创建掩码矩阵: [shot, shot] 对角线为0，其余为1
    #     mask = torch.ones(shot, shot) - torch.eye(shot)
    #     mask = mask.unsqueeze(0).unsqueeze(-1).to(spt.device)  # [1, shot, shot, 1]
        
    #     # 扩展维度用于广播计算
    #     expanded_spt = spt.unsqueeze(2)  # [way, shot, 1, feature_dim]
    #     repeated_spt = expanded_spt.repeat(1, 1, shot, 1)  # [way, shot, shot, feature_dim]
        
    #     # 计算排除自身原型: [way, shot, shot, feature_dim] * [1,shot,shot,1] -> 求和后归一化
    #     excluded_protos = (repeated_spt * mask).sum(dim=1) / (shot - 1)  # [way, shot, feature_dim]
        
    #     # 3. 计算每个排除原型与常规原型的距离
    #     dists = torch.norm(excluded_protos - prototype_general.unsqueeze(1), dim=2)  # [way, shot]
        
    #     # 4. 找到每个类中最不可靠的原型（距离最大的）
    #     _, max_indices = torch.max(dists, dim=1)  # [way]
        
    #     # 5. 调整原型位置
    #     adjusted_protos = []
    #     for i in range(way):
    #         # 获取当前类的常规原型
    #         general_proto = prototype_general[i]  # [feature_dim]
            
    #         # 获取最不可靠的排除原型
    #         unreliable_proto = excluded_protos[i, max_indices[i]]  # [feature_dim]
            
    #         # 计算偏移方向（远离不可靠原型）
    #         shift_direction = general_proto - unreliable_proto
            
    #         # 应用10%的反向偏移
    #         new_proto = general_proto + 0.1 * shift_direction
    #         adjusted_protos.append(new_proto)
        
    #     return torch.stack(adjusted_protos).view(way, self.in_c, -1)

    def robust_prototype_4d(self, spt, temperature=0.1, min_weight=0.1):
        """
        处理4D输入张量的鲁棒原型方法
        输入: [way, shot, c, m]
        输出: [way, c, m]
        
        参数:
            spt (Tensor): 支持集特征张量 [way, shot, c, m]
            temperature (float): softmax温度参数 (默认: 0.1)
            min_weight (float): 最小权重阈值 (默认: 0.1)
        
        返回:
            Tensor: 加权原型 [way, c, m]
        """
        way, shot, c, m = spt.shape
        
        # 处理shot=1的情况
        if shot == 1:
            return spt.squeeze(1)  # 移除shot维度
        
        # 1. 计算每个样本的可靠性分数
        # 展平空间维度以计算相似度
        spt_flat = spt.reshape(way, shot, -1)  # [way, shot, c*m]
        
        # 归一化特征向量
        spt_flat_norm = F.normalize(spt_flat, p=2, dim=-1)
        
        # 计算类内样本相似度矩阵
        sim_matrix = torch.matmul(spt_flat_norm, spt_flat_norm.transpose(1, 2))  # [way, shot, shot]
        
        # 创建掩码排除对角线
        mask = torch.eye(shot, device=spt.device).bool()
        sim_matrix_masked = sim_matrix.masked_fill(mask, 0)
        
        # 计算平均相似度分数 [way, shot]
        reliability_scores = sim_matrix_masked.sum(dim=-1) / (shot - 1)
        
        # 2. 生成样本权重
        weights = F.softmax(reliability_scores / temperature, dim=-1)
        weights = (1 - min_weight) * weights + min_weight / shot
        
        # 3. 加权平均计算原型（保持空间维度）
        # 扩展权重维度以匹配空间维度 [way, shot] -> [way, shot, 1, 1]
        weights_expanded = weights.view(way, shot, 1, 1)
        
        # 计算加权原型
        weighted_spt = spt * weights_expanded
        prototypes = weighted_spt.sum(dim=1)  # 沿shot维度求和
        
        return prototypes

    def adaptive_robust_prototype_4d(self, spt):
        """
        自适应参数的4D鲁棒原型方法
        输入: [way, shot, c, m]
        输出: [way, c, m]
        
        参数:
            spt (Tensor): 支持集特征张量 [way, shot, c, m]
        
        返回:
            Tensor: 加权原型 [way, c, m]
        """
        way, shot, c, m = spt.shape
        
        # 处理shot=1的情况
        if shot == 1:
            return spt.squeeze(1)
        
        # 展平空间维度以计算特征维度
        spt_flat = spt.reshape(way, shot, -1)  # [way, shot, c*m]
        feat_dim = spt_flat.size(-1)
        
        # 归一化特征向量
        spt_flat_norm = F.normalize(spt_flat, p=2, dim=-1)
        
        # 自动计算温度参数
        base_temp = 0.1
        adaptive_temp = base_temp * (feat_dim ** 0.5) / 16.0
        
        # 计算类内相似度
        sim_matrix = torch.matmul(spt_flat_norm, spt_flat_norm.transpose(1, 2))
        mask = torch.eye(shot, device=spt.device).bool()
        sim_matrix_masked = sim_matrix.masked_fill(mask, 0)
        
        # 手动计算类内相似度标准差
        sim_stds = []
        for i in range(way):
            # 获取非对角线元素
            non_diag = sim_matrix[i][~mask]
            
            # 计算标准差
            if len(non_diag) > 1:
                std_val = torch.std(non_diag)
            else:
                std_val = torch.tensor(0.0, device=spt.device)
            sim_stds.append(std_val)
        
        sim_std = torch.stack(sim_stds)
        avg_sim_std = torch.mean(sim_std).item()
        
        # 基于相似度标准差调整最小权重
        min_weight = max(0.05, min(0.2, 0.15 - 0.1 * avg_sim_std))
        
        # 应用鲁棒原型计算
        return self.robust_prototype_4d(spt, temperature=adaptive_temp, min_weight=min_weight)

    def safe_prototype_4d(self, spt, version="auto"):
        """
        安全调用4D原型方法
        
        参数:
            spt (Tensor): 支持集特征张量 [way, shot, c, m]
            version (str): 方法选择 ("robust", "adaptive", "auto")
        
        返回:
            Tensor: 原型 [way, c, m]
        """
        way, shot, c, m = spt.shape
        
        # 处理shot=1的情况
        if shot == 1:
            return spt.squeeze(1)
        
        # 自动选择方法
        if version == "auto":
            # 小样本时使用自适应方法，大样本时使用基本方法
            if shot <= 5:
                return self.adaptive_robust_prototype_4d(spt)
            else:
                return self.robust_prototype_4d(spt, temperature=0.1, min_weight=0.1)
        
        # 手动选择方法
        elif version == "robust":
            return self.robust_prototype_4d(spt)
        
        elif version == "adaptive":
            return self.adaptive_robust_prototype_4d(spt)
        
        else:
            raise ValueError(f"未知版本: {version}. 可选: 'robust', 'adaptive', 'auto'")

    def dist(self, input, spt=False, normalize=True):

        if spt:
            way, c, m = input.shape   #  way, c, m
            input_C_gap = input.mean(dim=-2)

            input = input.reshape(way * c, m)
            input = input.unsqueeze(dim=1)
            input_C_gap = input_C_gap.unsqueeze(dim=0)

            dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            if normalize:
                dist = dist / m
            dist = dist.reshape(way, c, -1)
            dist = dist.transpose(-1, -2)

            indices_way = torch.arange(way)
            indices_1 = indices_way.repeat_interleave((way - 1))
            indices_2 = []
            for i in indices_way:
                indices_2_temp = torch.cat((indices_way[:i], indices_way[i + 1:]),
                                           dim=-1)
                indices_2.append(indices_2_temp)
            indices_2 = torch.cat(indices_2, dim=0)

            dist_self = dist[indices_way, indices_way]
            dist_other = dist[indices_1, indices_2]
            dist_other = dist_other.view(way, way-1, -1)

            return dist_self, dist_other

        else:
            batch, c, m = input.shape
            input_C_gap = input.mean(dim=-2).unsqueeze(dim=-2)

            dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            if normalize:
                dist = dist / m

            return dist

    def weight(self, spt, qry):
        
        way, shot, c, m = spt.shape
        # print("spt:", spt.shape)
        batch, _, _, _ = qry.shape

        prt = spt.mean(dim=1)
        qry = qry.squeeze(dim=1)
        # print("ptr:", prt.shape)
        dist_prt_self, dist_prt_other = self.dist(prt, spt=True)
        dist_qry_self = self.dist(qry)

        dist_prt_self = dist_prt_self.view(-1, c)
        dist_prt_other, _ = dist_prt_other.min(dim=-2)
        dist_prt_other = dist_prt_other.view(-1, c)
        dist_qry_self = dist_qry_self.view(-1, c)

        weight_prt_self = self.prt_self(dist_prt_self)
        weight_prt_self = weight_prt_self.view(way, 1, c)
        weight_prt_other = self.prt_other(dist_prt_other)
        weight_prt_other = weight_prt_other.view(way, 1, c)
        weight_qry_self = self.qry_self(dist_qry_self)
        weight_qry_self = weight_qry_self.view(1, batch, c)

        alpha_prt = 0.5
        alpha_prt_qry = 0.5

        beta_prt = 1. - alpha_prt
        beta_prt_qry = 1. - alpha_prt_qry

        weight_prt = alpha_prt * weight_prt_self + beta_prt * weight_prt_other
        weight = alpha_prt_qry * weight_prt + beta_prt_qry * weight_qry_self

        return weight

    def forward(self, spt, qry):
        weight = self.weight(spt, qry)
        weight = self.add_noise(weight)

        return weight