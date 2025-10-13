import torch
import torch.nn as nn
from .core_modules import (
    PIDBranches, DualAttentionFusion, FourierTransformModule,
    EnhancedChannelAttention, EnhancedSpatialAttention, Squeeze, GatedConvFuse
)


class AdvancedHSIClassifier(nn.Module):
    def __init__(self, in_channels, spatial_size, num_bands, num_classes):
        super(AdvancedHSIClassifier, self).__init__()

        self.conv3d_block = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 7), padding=(1, 1, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 5), padding=(1, 1, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.pid_branches = PIDBranches(in_channels=32)

        self.spectral_branch = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, None)),
            Squeeze(2),
            Squeeze(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.spatial_branch = nn.Sequential(
            nn.AdaptiveAvgPool3d((spatial_size, spatial_size, 1)),
            Squeeze(4),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.spectral_proj = nn.Linear(128, 256)

        self.gsf = GatedConvFuse(channels=256)

        self.feature_fusion = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.feature_reshape = nn.Sequential(
            nn.Linear(512, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4))
        )

        self.dual_attention_fusion = DualAttentionFusion(32, reduction_ratio=8)

        self.local_branch = nn.Sequential(
            EnhancedChannelAttention(32),
            EnhancedSpatialAttention(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.global_branch = nn.Sequential(
            FourierTransformModule(32, reduction_ratio=8),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.final_fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv3d_block(x)

        x_compressed = nn.AdaptiveAvgPool3d((1, None, None))(x).squeeze(2)
        x_pid = self.pid_branches(x_compressed)
        x_fused = x_compressed + x_pid
        x_fused_3d = x_fused.unsqueeze(2)

        spectral = self.spectral_branch(x)
        spatial = self.spatial_branch(x_fused_3d)

        spectral_proj = self.spectral_proj(spectral)
        spectral_4d = spectral_proj.unsqueeze(-1).unsqueeze(-1)
        spatial_4d = spatial.unsqueeze(-1).unsqueeze(-1)

        fused_gsf = self.gsf(spectral_4d, spatial_4d)
        fused_gsf = fused_gsf.squeeze(-1).squeeze(-1)

        fused_features = self.feature_fusion(fused_gsf)
        reshaped_features = self.feature_reshape(fused_features)
        fused_att = self.dual_attention_fusion(reshaped_features)

        local_features = self.local_branch(fused_att)
        global_features = self.global_branch(fused_att)

        fused_advanced = torch.cat((local_features, global_features), dim=1)
        final_features = self.final_fusion(fused_advanced)
        return self.classifier(final_features)
