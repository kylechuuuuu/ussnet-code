import torch
import torch.nn as nn
from .model_part import *

class pretrain_reseg(nn.Module):
    def __init__(self, 
                 inp_channels=3, 
                 out_channels=3, 
                 seg_classes=1,
                 dim=72,
                 num_blocks=[3],
                 pretrained_weights=None):  # Add pretrained_weights parameter
        super(pretrain_reseg, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed2 = OverlapPatchEmbed2(seg_classes, dim)

        self.encoder_level1 = nn.Sequential(*[Transformer(dim) for _ in range(num_blocks[0])])

        self.segmentation_output = nn.Conv2d(int(dim), seg_classes, kernel_size=3, padding=1)

        # Load pretrained weights if provided
        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)

    def load_pretrained_weights(self, weights_path):
        """ Load pretrained weights from a given path. """
        pretrained_dict = torch.load(weights_path)
        model_dict = self.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        output = self.segmentation_output(out_enc_level1)

        return output


class reseg(nn.Module):
    def __init__(self, 
                 inp_channels=3, 
                 out_channels=3, 
                 seg_classes=1,
                 dim=72,
                 num_blocks=[2, 3, 3, 4],
                 pretrained_weights=None):  # Add pretrained_weights parameter
        super(reseg, self).__init__()

        self.pretrain = pretrain_reseg(pretrained_weights=pretrained_weights)  # Pass pretrained_weights to pretrain
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed2 = OverlapPatchEmbed2(seg_classes, dim)

        self.encoder_level1 = nn.Sequential(*[Transformer(dim) for _ in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[Transformer(dim * 2 ** 1) for _ in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim * 2 ** 1))  # From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[Transformer(dim * 2 ** 2) for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  # From Level 3 to Level 4
        self.latent = nn.Sequential(*[Transformer(dim * 2 ** 3) for _ in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim * 2 ** 3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1)
        self.decoder_level3 = nn.Sequential(*[Transformer(dim * 2 ** 2) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1)
        self.decoder_level2 = nn.Sequential(*[Transformer(dim * 2 ** 1) for _ in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim * 2 ** 1))  # From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[Transformer(dim * 2 ** 1) for _ in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[Transformer(dim * 2 ** 1) for _ in range(num_blocks[0])])

        self.restoration_output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, padding=1)
        self.segmentation_output = nn.Conv2d(int(dim * 2 ** 1), seg_classes, kernel_size=3, padding=1)

    def forward(self, inp_img):
        ## stage one
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)     

        latent = self.latent(inp_enc_level4) 
        
        inp_dec_level3_restoration = self.up4_3(latent)
        inp_dec_level3_restoration = torch.cat([inp_dec_level3_restoration, out_enc_level3], 1)
        inp_dec_level3_restoration = self.reduce_chan_level3(inp_dec_level3_restoration)
        out_dec_level3_restoration = self.decoder_level3(inp_dec_level3_restoration) 

        inp_dec_level2_restoration = self.up3_2(out_dec_level3_restoration)
        inp_dec_level2_restoration = torch.cat([inp_dec_level2_restoration, out_enc_level2], 1)
        inp_dec_level2_restoration = self.reduce_chan_level2(inp_dec_level2_restoration)
        out_dec_level2_restoration = self.decoder_level2(inp_dec_level2_restoration) 

        inp_dec_level1_restoration = self.up2_1(out_dec_level2_restoration)
        inp_dec_level1_restoration = torch.cat([inp_dec_level1_restoration, out_enc_level1], 1)
        out_dec_level1_restoration = self.decoder_level1(inp_dec_level1_restoration)
        
        out_dec_level1_restoration = self.refinement(out_dec_level1_restoration)

        restoration = self.restoration_output(out_dec_level1_restoration) + inp_img
        ## stage two
        pretrain = self.pretrain(restoration)
        inp_enc_level1 = self.patch_embed(restoration)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)     

        latent = self.latent(inp_enc_level4) 
        
        inp_dec_level3_restoration = self.up4_3(latent)
        inp_dec_level3_restoration = torch.cat([inp_dec_level3_restoration, out_enc_level3], 1)
        inp_dec_level3_restoration = self.reduce_chan_level3(inp_dec_level3_restoration)
        out_dec_level3_restoration = self.decoder_level3(inp_dec_level3_restoration) 

        inp_dec_level2_restoration = self.up3_2(out_dec_level3_restoration)
        inp_dec_level2_restoration = torch.cat([inp_dec_level2_restoration, out_enc_level2], 1)
        inp_dec_level2_restoration = self.reduce_chan_level2(inp_dec_level2_restoration)
        out_dec_level2_restoration = self.decoder_level2(inp_dec_level2_restoration) 

        inp_dec_level1_restoration = self.up2_1(out_dec_level2_restoration)
        inp_dec_level1_restoration = torch.cat([inp_dec_level1_restoration, out_enc_level1], 1)
        out_dec_level1_restoration = self.decoder_level1(inp_dec_level1_restoration)
        
        out_dec_level1_restoration = self.refinement(out_dec_level1_restoration)

        segmentation = self.segmentation_output(out_dec_level1_restoration) + pretrain

        ## stage three
        inp_enc_level1 = self.patch_embed2(segmentation)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)     

        latent = self.latent(inp_enc_level4) 
        
        inp_dec_level3_restoration = self.up4_3(latent)
        inp_dec_level3_restoration = torch.cat([inp_dec_level3_restoration, out_enc_level3], 1)
        inp_dec_level3_restoration = self.reduce_chan_level3(inp_dec_level3_restoration)
        out_dec_level3_restoration = self.decoder_level3(inp_dec_level3_restoration) 

        inp_dec_level2_restoration = self.up3_2(out_dec_level3_restoration)
        inp_dec_level2_restoration = torch.cat([inp_dec_level2_restoration, out_enc_level2], 1)
        inp_dec_level2_restoration = self.reduce_chan_level2(inp_dec_level2_restoration)
        out_dec_level2_restoration = self.decoder_level2(inp_dec_level2_restoration) 

        inp_dec_level1_restoration = self.up2_1(out_dec_level2_restoration)
        inp_dec_level1_restoration = torch.cat([inp_dec_level1_restoration, out_enc_level1], 1)
        out_dec_level1_restoration = self.decoder_level1(inp_dec_level1_restoration)
        
        out_dec_level1_restoration = self.refinement(out_dec_level1_restoration)

        output = self.segmentation_output(out_dec_level1_restoration) + segmentation

        return restoration, segmentation, output

# Example of how to create the model and load pretrained weights
# model = reseg(pretrained_weights='path/to/pretrained/weights.pth')


