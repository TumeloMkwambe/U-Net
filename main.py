import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

    def forward(self, feature_bank):
        feature_bank = self.block(feature_bank)
        return feature_bank
    
class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

    def forward(self, decoder_feature_bank, encoder_feature_bank):
        
        if decoder_feature_bank.shape[2] != encoder_feature_bank.shape[2] or decoder_feature_bank.shape[3] != encoder_feature_bank.shape[3]:
            raise ValueError("incompatible shapes for skip connection")
        else:
            feature_bank = torch.cat([decoder_feature_bank, encoder_feature_bank], dim = 1)
            feature_bank = self.block(feature_bank)
            return feature_bank

class U_Net(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [3, 64, 128, 256, 512, 1024]
        self.encoder = nn.ModuleList([
            Encoder(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])
        channels.reverse()
        self.decoder = nn.ModuleList([
            Decoder(channels[i], channels[i + 1]) for i in range(len(channels) - 2)
        ])

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.convUps = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size = 2, stride = 2) for i in range(len(channels) - 2)
        ])
        
        self.final_conv = nn.Conv2d(64, 2, kernel_size = 3, padding = 1)


    def forward(self, img):

        encoder_feature_banks = []
        feature_bank = img

        for idx, block in enumerate(self.encoder):
            feature_bank = block(feature_bank)

            if idx < len(self.encoder) - 1:
                encoder_feature_banks.append(feature_bank)
                feature_bank = self.max_pool(feature_bank)

        encoder_feature_banks.reverse()

        for idx, block in enumerate(self.decoder):
            feature_bank = self.convUps[idx](feature_bank)
            feature_bank = block(feature_bank, encoder_feature_banks[idx])

        output = self.final_conv(feature_bank)

        return output