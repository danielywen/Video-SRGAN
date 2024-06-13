import math
import torch
from torch import nn

# This is the original Generator
# class Generator(nn.Module):
#     def __init__(self, scale_factor):
#         upsample_block_num = int(math.log(scale_factor, 2))

#         super(Generator, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=9, padding=4),
#             nn.PReLU()
#         )
#         self.block2 = ResidualBlock(64)
#         self.block3 = ResidualBlock(64)
#         self.block4 = ResidualBlock(64)
#         self.block5 = ResidualBlock(64)
#         self.block6 = ResidualBlock(64)
#         self.block7 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
#         block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
#         self.block8 = nn.Sequential(*block8)

#     def forward(self, x):
#         block1 = self.block1(x)
#         block2 = self.block2(block1)
#         block3 = self.block3(block2)
#         block4 = self.block4(block3)
#         block5 = self.block5(block4)
#         block6 = self.block6(block5)
#         block7 = self.block7(block6)
#         block8 = self.block8(block1 + block7)

#         return (torch.tanh(block8) + 1) / 2

# This has the LSTM layer
import math
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, scale_factor, sequence_length):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.sequence_length = sequence_length
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(sequence_length)]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        )
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

        # LSTM layer for temporal consistency
        self.lstm = nn.LSTM(input_size=64 * 64 * 64, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 64 * 64 * 64)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        conv1 = self.conv1(x)
        res_blocks = self.res_blocks(conv1)
        conv2 = self.conv2(res_blocks)
        res_out = conv1 + conv2

        res_out = res_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(res_out)
        lstm_out = lstm_out.contiguous().view(batch_size * seq_len, -1)
        fc_out = self.fc(lstm_out)
        fc_out = fc_out.view(batch_size * seq_len, 64, 64, 64)

        upsampled = self.upsample_blocks(fc_out)
        output = self.conv3(upsampled)
        return (torch.tanh(output) + 1) / 2


# This is the original discriminator without dropout and filter changes
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2),

#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(512, 1024, kernel_size=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(1024, 1, kernel_size=1)
#         )

#     def forward(self, x):
#         batch_size = x.size(0)
#         return torch.sigmoid(self.net(x).view(batch_size))
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        self.dropout = nn.Dropout(0.3)  # Adding dropout for regularization

    def forward(self, x):
        batch_size = x.size(0)
        x = self.net(x)
        x = self.dropout(x)  # Applying dropout
        return torch.sigmoid(x.view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
