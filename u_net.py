import copy
import torch
from torch import nn

class TwoConvolutions(nn.Module):

    '''
    Defines the necessary attributes and methods to compute two consecutive convolutions while maintaining the image resolution. The first
    intended to double or halve the number of channels depending on the block it resides (encoder or decoder), the second maintains the channels.

    Args:
        in_channels (int): Number of input channels of the first convolion, from which we double or halve.
        out_channels (int): Number of output channels, double or half of the input channels.
    '''

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

    '''
    This defines the attributes and forward propagation of a single encoder block in the contracting path of the U-Net architecture. This block
    constists of two convolutions (the first which doubles the number of input channels, and second which maintains the channels, both with no
    effect on the resolution) followed by a max-pooling operation which halves the image resolution (halves width and height).

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels prior to the max-pooling operation.
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.TwoConvolutions = TwoConvolutions(in_channels, out_channels)
        self.max_pooling = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, image):

        skip_features = self.TwoConvolutions(image)
        features = self.max_pooling(skip_features)

        return features, skip_features


class DecoderBlock(nn.Module):

    '''
    Implementation of a single decoder block found in the expansive part of the U-Net architecture. It consists of an upsampling operation on
    the image image from an immediate previous layer, the upsampling operation doubles the image resolution but halves the number of input
    channels. We then concatenate the output of the upsampling operation with the output of two convolutions from a decoder block in the
    contracting path over a skip connection, both concatenated tensors share the same image resolution and the number of channels.

    Args:
        in_channels (int): Number of channels prior to the upsampling operation, also same number of channels after concatenation.
        out_channels (int): Number of channels after upsampling operation (prior to concatenation), also number of channels after the two
                            consecutive convolutions.
        bilinear (bool): The argument dictates the type of upsampling operation to use, allowing one to choose between the two variants,
                         set True to use torch.nn.Upsample for bilinear upsampling or False to use torch.nn.ConvTranspose2d for upsampling.
    '''

    def __init__(self, in_channels, skip_channels, bilinear = False):
        super().__init__()

        if bilinear:
            self.conc_channels = in_channels + skip_channels
            self.upSampling = nn.Upsample(scale_factor = 2, mode = 'bilinear')
            self.TwoConvolutions = TwoConvolutions(self.conc_channels, self.conc_channels // 2)
        else:
            self.conc_channels = (in_channels // 2) + skip_channels
            self.upSampling = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
            self.TwoConvolutions = TwoConvolutions(self.conc_channels, self.conc_channels // 2)

    def forward(self, input_, skip_input):

        features = self.upSampling(input_)
        features = torch.cat([features, skip_input], dim = 1)
        features = self.TwoConvolutions(features)

        return features


class U_Net(nn.Module):

    '''
    The implementation allows for the creation of a U-Net model given any number of channels and image resolution with specified depth,
    not just 512x512 images with 3 channels.

    Args:
        in_channels (int): Number of channels of the input image.
        out_channels (int): Number of output channels which corresponds with predicted classes.
        start_channels (int): This is the number of channels to output in the first convolution of the first encoder layer, from this value
                              we double the number of channels at each encoder block (after max pooling) until we reach the bottleneck.
        depth (int): Depth of the model, can also be interpreted as the number of downsampling operations in the encoder until the bottleneck
                     or the number of upsampling operations from the bottleneck to the last output layer in the decorder.
        bilinear (bool): The argument dictates the type of upsampling operation to use, allowing one to choose between the two variants,
                         set True to use torch.nn.Upsample for bilinear upsampling or False to use torch.nn.ConvTranspose2d for upsampling.
    '''

    def __init__(self, in_channels, out_channels, start_channels, depth, bilinear = False):
        super().__init__()

        self.bilinear = bilinear

        encoder_channels = [in_channels] + [start_channels * (2 ** i) for i in range(depth + 1)]

        self.Encoder = nn.ModuleList([
            EncoderBlock(encoder_channels[i], encoder_channels[i+1]) for i in range(depth)
        ])

        self.Bottleneck = TwoConvolutions(encoder_channels[depth], encoder_channels[depth + 1])

        if bilinear:
            skip_channels = copy.copy(encoder_channels)
            skip_channels.pop()
            skip_channels.reverse()
            skip_channels.pop()

            curr_channels = encoder_channels[-1]
            self.Decoder = nn.ModuleList()

            for skip_ch in skip_channels:
                self.Decoder.append(DecoderBlock(curr_channels, skip_ch, bilinear = True))
                curr_channels = (curr_channels + skip_ch) // 2


            self.FinalConvolution = nn.Conv2d(
                in_channels = curr_channels,
                out_channels = out_channels,
                kernel_size=1
            )
        else:
            decoder_channels = copy.copy(encoder_channels)
            decoder_channels.reverse()
            decoder_channels.pop()

            self.Decoder = nn.ModuleList([
                DecoderBlock(decoder_channels[i], decoder_channels[i+1], bilinear) for i in range(depth)
            ])

            self.FinalConvolution = nn.Conv2d(in_channels = decoder_channels[-1], out_channels = out_channels, kernel_size = 3, stride = 1, padding = (1,1))

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
