import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """焦点损失（解决类别不平衡）"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SobelEdgeDetection(nn.Module):
    """Sobel边缘检测（用于PID的D分支）"""
    def __init__(self):
        super(SobelEdgeDetection, self).__init__()
        # Sobel算子（x/y方向）
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)  # 避免根号下为0
        return edge


class ChannelAttention(nn.Module):
    """通道注意力（基础版）"""
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """空间注意力（基础版）"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_combined = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv(x_combined)
        return self.sigmoid(spatial_att)


class PIDBranches(nn.Module):
    """PID分支（比例-积分-微分特征融合）"""
    def __init__(self, in_channels):
        super(PIDBranches, self).__init__()
        self.p_branch = nn.Sequential(  # 比例分支（局部特征）
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            ChannelAttention(in_channels)
        )
        self.i_branch = nn.Sequential(  # 积分分支（全局特征）
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            SpatialAttention()
        )
        self.d_branch = nn.Sequential(  # 微分分支（边缘特征）
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            SobelEdgeDetection()
        )

    def forward(self, x):
        p = self.p_branch(x)
        i = self.i_branch(x)
        d = self.d_branch(x)
        return p * i * d


class DualAttentionFusion(nn.Module):
    """双注意力融合（全局+局部）"""
    def __init__(self, in_channels, reduction_ratio=8):
        super(DualAttentionFusion, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        reduced_channels = max(1, in_channels // reduction_ratio)

        # 全局注意力分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 局部通道注意力分支
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 局部空间注意力子分支
        self.spatial_sub_branch = nn.Sequential(
            nn.Conv2d(2 * in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        global_att = self.global_branch(x)
        local_ch_att = self.local_branch(x)
        spatial_input = torch.cat([x, local_ch_att], dim=1)
        spatial_att = self.spatial_sub_branch(spatial_input)
        local_att = local_ch_att * spatial_att
        fused_att = global_att * local_att
        return x * fused_att


class FourierTransformModule(nn.Module):
    """傅里叶变换模块（频率域特征增强）"""
    def __init__(self, in_channels, reduction_ratio=8):
        super(FourierTransformModule, self).__init__()
        self.in_channels = in_channels
        # 频率重要性筛选
        self.filter_kernel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        spatial_features = x
        # 傅里叶变换（实部+虚部）
        x_fft = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(x_fft)  # 幅度谱
        phase = torch.angle(x_fft)    # 相位谱
        # 频率筛选
        freq_importance = self.filter_kernel(magnitude)
        filtered_magnitude = magnitude * freq_importance
        # 逆傅里叶变换
        real = filtered_magnitude * torch.cos(phase)
        imag = filtered_magnitude * torch.sin(phase)
        x_ifft = torch.fft.irfft2(torch.complex(real, imag), s=x.shape[-2:], norm='ortho')
        # 残差连接+归一化
        out = spatial_features + x_ifft
        out = self.norm(out)
        out = self.activation(out)
        return out


class EnhancedChannelAttention(nn.Module):
    """增强版通道注意力（平均+最大池化融合）"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(EnhancedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * channel_att.expand_as(x)


class EnhancedSpatialAttention(nn.Module):
    """增强版空间注意力（简化卷积核）"""
    def __init__(self, kernel_size=7):
        super(EnhancedSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_combined = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv1(x_combined)
        return x * self.sigmoid(spatial_att)


class Squeeze(nn.Module):
    """维度压缩模块（适配动态维度）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if -x.dim() <= self.dim < x.dim():
            return x.squeeze(self.dim)
        return x


class GatedConvFuse(nn.Module):
    """HorNet递归门控卷积融合（替换原GSF，核心模块）"""
    def __init__(self, channels=256, order=3):
        super(GatedConvFuse, self).__init__()
        self.channels = channels  # 目标输出通道（与原GSF一致为256）
        self.order = order        # 递归阶数
        self.dims = [channels // (2 ** i) for i in range(order)]  # 递归通道序列：[256,128,64]

        # 1. 输入投影（光谱+空间→初始特征+递归特征）
        self.proj_in = nn.Conv2d(2 * channels, self.dims[0] + sum(self.dims), kernel_size=1)
        # 2. 深度卷积（局部特征交互）
        self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=3, padding=1, groups=sum(self.dims))
        # 3. 递归投影层（通道匹配）
        self.projs = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=1) for i in range(order - 1)
        ])
        # 4. 通道恢复（递归后64→256，修复通道不匹配）
        self.channel_recover = nn.Conv2d(self.dims[-1], self.channels, kernel_size=1)
        # 5. 输出投影（最终特征调整）
        self.proj_out = nn.Conv2d(self.channels, self.channels, kernel_size=1)

    def forward(self, spectral_feat, spatial_feat):
        # 1. 拼接光谱+空间特征（b, 2*256, 1, 1）
        x = torch.cat([spectral_feat, spatial_feat], dim=1)

        # 2. 输入投影+拆分（初始特征y: b,256,1,1；递归特征x: b,448,1,1）
        x = self.proj_in(x)
        y, x = torch.split(x, [self.dims[0], sum(self.dims)], dim=1)

        # 3. 递归门控高阶交互（3阶：256→128→64）
        x = self.dwconv(x)
        x_list = torch.split(x, self.dims, dim=1)
        x = y * x_list[0]  # 1阶交互
        for i in range(self.order - 1):
            x = self.projs[i](x) * x_list[i + 1]  # 2→3阶交互

        # 4. 通道恢复（64→256）
        x = self.channel_recover(x)

        # 5. 输出投影
        return self.proj_out(x)