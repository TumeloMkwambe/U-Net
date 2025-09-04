import torch
from torch import nn

class TwoConvolutions(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = (1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = (1,1)),
            nn.ReLU(),
        )

    def forward(self, input_):

        output = self.block(input_)

        return output
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.TwoConvolutions = TwoConvolutions(in_channels, out_channels)
        self.max_pooling = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, image):

        skip_features = self.TwoConvolutions(image)
        features = self.max_pooling(skip_features)

        return features, skip_features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear = False):
        super().__init__()

        if bilinear:
            self.upSampling = nn.UpSample(scale_factor = 2, mode = 'bilinear')
        else:
            self.upSampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)

        self.TwoConvolutions = TwoConvolutions(out_channels * 2, out_channels)

    def forward(self, input_, skip_input):

        features = self.upSampling(input_)
        features = torch.cat([features, skip_input], dim = 1)
        features = self.TwoConvolutions(features)

        return features

class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels, start_channels, depth, bilinear = False):
        super().__init__()

        self.channels = [in_channels] + [start_channels * (2 ** i) for i in range(depth + 1)]

        self.Encoder = nn.ModuleList([
            EncoderBlock(self.channels[i], self.channels[i+1]) for i in range(depth)
        ])

        self.Bottleneck = TwoConvolutions(self.channels[depth], self.channels[depth + 1])

        self.channels.reverse()
        self.channels.pop()

        self.Decoder = nn.ModuleList([
            DecoderBlock(self.channels[i], self.channels[i+1], bilinear) for i in range(depth)
        ])

        self.FinalConvolution = nn.Conv2d(in_channels = self.channels[-1], out_channels = out_channels, kernel_size = 3, stride = 1, padding = (1,1))

    def forward(self, image):

        encoder_features = []
        features = image

        for block in self.Encoder: # warning: first checkpoint is useless but it wont break the code ofcourse
            features, skip_features = torch.utils.checkpoint.checkpoint(block, features, use_reentrant = False)
            encoder_features.append(skip_features)

        features = torch.utils.checkpoint.checkpoint(self.Bottleneck, features, use_reentrant = False)
        encoder_features.reverse()

        for idx, block in enumerate(self.Decoder):
            features = torch.utils.checkpoint.checkpoint(block, features, encoder_features[idx], use_reentrant = False)

        mask = self.FinalConvolution(features)

        return mask
