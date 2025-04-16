import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(g_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(x_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionVNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AttentionVNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(1, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Attention Gates
        self.att4 = AttentionBlock(256, 128, 128)
        self.att3 = AttentionBlock(128, 64, 64)
        self.att2 = AttentionBlock(64, 32, 32)
        self.att1 = AttentionBlock(32, 16, 16)

        # Decoder
        self.decoder4 = self.conv_block(256 + 128, 128)
        self.decoder3 = self.conv_block(128 + 64, 64)
        self.decoder2 = self.conv_block(64 + 32, 32)
        self.decoder1 = self.conv_block(32 + 16, 16)

        # Final layer
        self.final_conv = nn.Conv3d(16, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2, stride=2))

        # Decoder + Attention
        g4 = F.interpolate(bottleneck, scale_factor=2, mode="trilinear", align_corners=True)
        x4 = self.att4(g4, enc4)
        dec4 = self.decoder4(torch.cat([g4, x4], dim=1))

        g3 = F.interpolate(dec4, scale_factor=2, mode="trilinear", align_corners=True)
        x3 = self.att3(g3, enc3)
        dec3 = self.decoder3(torch.cat([g3, x3], dim=1))

        g2 = F.interpolate(dec3, scale_factor=2, mode="trilinear", align_corners=True)
        x2 = self.att2(g2, enc2)
        dec2 = self.decoder2(torch.cat([g2, x2], dim=1))

        g1 = F.interpolate(dec2, scale_factor=2, mode="trilinear", align_corners=True)
        x1 = self.att1(g1, enc1)
        dec1 = self.decoder1(torch.cat([g1, x1], dim=1))

        output = self.final_conv(dec1)
        return output
