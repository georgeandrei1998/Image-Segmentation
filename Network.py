import torch
import torch.nn as nn

class UNet(nn.Module):
    @staticmethod
    def downsampling_block(in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

        return block

    @staticmethod
    def upsampling_block(in_channels, mid_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=2, padding=1, output_padding=1)
        )

        return block

    @staticmethod
    def final_block(in_channels, mid_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

        return block

    @staticmethod
    def bottleneck_block(in_channels, mid_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2,
                               padding=1, output_padding=1)
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        # encode
        self.conv_encode1 = self.downsampling_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode2 = self.downsampling_block(64, 128)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode3 = self.downsampling_block(128, 256)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = self.bottleneck_block(256, 512, 256)

        # decode
        self.conv_decode3 = self.upsampling_block(512, 256, 128)
        self.conv_decode2 = self.upsampling_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

    '''@staticmethod
    def crop_and_concat(upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            if (bypass.size()[2] - upsampled.size()[2]) % 2 == 0:
                bypass = F.pad(bypass, (-c, -c, -c, -c))
            else:
                bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))

        return torch.cat((upsampled, bypass), 1)'''

    @staticmethod
    def crop_and_concat(upsampled, bypass):
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)

        # decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)

        return final_layer
