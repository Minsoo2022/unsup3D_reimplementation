import torch
from torch import nn
import numpy as np
import math
import torchvision

class Encoder(nn.Module):
    def __init__(self, opt, ch_output):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.ch_img = opt.ch_img
        self.ch_latent = opt.ch_latent
        self.activation = nn.Tanh()

        for i in range(4):
            if i == 0:
                ch_in = self.ch_img
                ch_out = self.ch_latent
            else:
                ch_in = self.ch_latent * (2**(i-1))
                ch_out = self.ch_latent * (2**i)
            # else:
            #     ch_in = self.ch_latent * (2**(i-1))
            #     ch_out = self.ch_latent * (2 ** (i-1))

            self.layers.append(nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False))
            self.layers.append(nn.ReLU(inplace=False))

        self.layers.append(nn.Conv2d(self.ch_latent * 8, self.ch_latent * 8, 4, 1, 0, bias=False))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Conv2d(ch_out, ch_output, 1, 1, 0, bias=False))
        self.layers.append(self.activation)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.view(x.shape[0], -1)
        return x

class Encoder_Decoder(nn.Module):
    def __init__(self, opt, ch_output, activation=True, value_norm=False):
        super(Encoder_Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.ch_img = opt.ch_img
        self.ch_latent = opt.ch_latent
        self.ch_mid_latent = opt.ch_mid_latent
        self.activation = nn.Tanh() if activation else None
        self.value_norm = value_norm

        self.layers.append(nn.Conv2d(self.ch_img, self.ch_latent, 4, 2, 1, bias=False))
        self.layers.append(nn.GroupNorm(16, self.ch_latent))
        self.layers.append(nn.LeakyReLU(0.2))
        for i in range(3):
            self.layers.append(nn.Conv2d(self.ch_latent*(2**(i)), self.ch_latent*(2**(i+1)), 4, 2, 1, bias=False))
            self.layers.append(nn.GroupNorm(16*(2**(i+1)), self.ch_latent*(2**(i+1))))
            self.layers.append(nn.LeakyReLU(0.2))
        self.layers.append(nn.Conv2d(self.ch_latent*(2**(i+1)), self.ch_mid_latent, 4, 1, 0, bias=False))
        self.layers.append(nn.ReLU())

        self.layers.append(conv_trans_block(self.ch_mid_latent, self.ch_latent * 8, None,
                                            kernel_size=4, stride=1, padding=0, no_norm=True))
        self.layers.append(conv_trans_block(self.ch_latent * 8, self.ch_latent * 4, 16 * 4))
        self.layers.append(conv_trans_block(self.ch_latent * 4, self.ch_latent * 2, 16 * 2))
        self.layers.append(conv_trans_block(self.ch_latent * 2, self.ch_latent * 1, 16 * 1))
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(nn.Conv2d(self.ch_latent, self.ch_latent, 3, 1, 1, bias=False))
        self.layers.append(nn.GroupNorm(16, self.ch_latent))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(self.ch_latent, self.ch_latent, 5, 1, 2, bias=False))
        self.layers.append(nn.GroupNorm(16, self.ch_latent))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(self.ch_latent, ch_output, 5, 1, 2, bias=False))
        # if self.activation is not None:
        #     self.layers.append(self.activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.value_norm:
            x = x - x.view(x.shape[0], -1).mean(1).view(x.shape[0], 1, 1, 1)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Conf_Encoder_decoder(nn.Module):
    def __init__(self, opt):
        super(Conf_Encoder_decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.ch_img = opt.ch_img
        self.ch_latent = opt.ch_latent
        self.ch_mid_latent = opt.ch_mid_latent

        self.layers.append(nn.Conv2d(self.ch_img, self.ch_latent, 4, 2, 1, bias=False))
        self.layers.append(nn.GroupNorm(16, self.ch_latent))
        self.layers.append(nn.LeakyReLU(0.2))
        for i in range(3):
            self.layers.append(nn.Conv2d(self.ch_latent*(2**(i)), self.ch_latent*(2**(i+1)), 4, 2, 1, bias=False))
            self.layers.append(nn.GroupNorm(16*(2**(i+1)), self.ch_latent*(2**(i+1))))
            self.layers.append(nn.LeakyReLU(0.2))
        self.layers.append(nn.Conv2d(self.ch_latent*(2**(i+1)), self.ch_mid_latent // 2, 4, 1, 0, bias=False))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.ConvTranspose2d(self.ch_mid_latent // 2, self.ch_mid_latent * 4, 4, 1, 0, bias=False))
        self.layers.append(nn.ReLU())
        lastlayer1 = [nn.Conv2d(self.ch_latent * 2, 2, 3, 1, 1, bias=False), nn.Softplus()]
        self.lastlayer1 = nn.Sequential(*lastlayer1)
        lastlayer2 = [nn.Conv2d(self.ch_latent, 2, 5, 1, 2, bias=False), nn.Softplus()]
        self.lastlayer2 = nn.Sequential(*lastlayer2)

        for i in range(3):
            self.layers.append(nn.ConvTranspose2d(self.ch_latent * (2**(3-i)), self.ch_latent * (2**(2-i)), 4, 2, 1, bias=False))
            self.layers.append(nn.GroupNorm(64 // (2**(i)), self.ch_latent * (2**(2-i))))
            self.layers.append(nn.ReLU())
            if i == 1:
                self.layers.append(self.lastlayer1)
        self.layers.append(nn.ConvTranspose2d(self.ch_latent, self.ch_latent, 4, 2, 1, bias=False))
        self.layers.append(nn.GroupNorm(16, self.ch_latent))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            if layer._get_name() == 'Sequential':
                y = layer(x)
            else:
                x = layer(x)
        x = self.lastlayer2(x)
        return x, y

norm_diff = 0.8
class conv_trans_block(nn.Module):
    def __init__(self, ch_in, ch_out, group, kernel_size=4, stride=2, padding=1, no_norm=False):
        super(conv_trans_block, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding, bias=False))
        if not no_norm:
            self.layers.append(nn.GroupNorm(group, ch_out))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False))
        if not no_norm:
            self.layers.append(nn.GroupNorm(group, ch_out))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.mean = torch.cuda.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.cuda.FloatTensor([0.229, 0.224, 0.225])

        vgg = torchvision.models.vgg16(pretrained=True)
        features = vgg.features
        self.layers1 = nn.Sequential()
        self.layers2 = nn.Sequential()
        self.layers3 = nn.Sequential()
        self.layers4 = nn.Sequential()
        for x in range(4):
            self.layers1.add_module(str(x), features[x])
        for x in range(4, 9):
            self.layers2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.layers3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.layers4.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def __call__(self, img1, img2, sigma, mask):
        img = torch.cat([img1,img2], 0)
        img = img /2 +0.5
        img = (img -self.mean.view(1,3,1,1)) / self.std.view(1,3,1,1)

        feature_list = []
        f = self.layers1(img)
        feature_list +=[torch.chunk(f, 2, dim=0)]
        f = self.layers2(f)
        feature_list +=[torch.chunk(f, 2, dim=0)]
        f = self.layers3(f)
        feature_list +=[torch.chunk(f, 2, dim=0)]
        f = self.layers4(f)
        feature_list +=[torch.chunk(f, 2, dim=0)]

        loss_list = []
        for f1, f2 in feature_list[2:3]:
            loss = (f1-f2)**2
            loss = loss / (2*sigma**2 + 1e-7) + (sigma + 1e-7).log()
            _, _, height, width = loss.shape
            _, _, mask_height, mask_width = mask.shape
            mask_pooled = nn.functional.avg_pool2d(mask, kernel_size=(mask_height//height,mask_width//width),
                                             stride=(mask_height//height,mask_width//width)).expand_as(loss)
            loss = (loss * mask_pooled).sum() / mask_pooled.sum()
            loss_list.append(loss)
        return sum(loss_list)

def L1loss_with_confidence(gt, pred, confidence, mask):
    criterian = nn.L1Loss(reduction='none')
    loss = criterian(gt, pred) / (confidence + 1e-7) + torch.log(confidence + 1e-7)
    loss = (loss * mask).sum() / mask.sum()
    return loss

def cal_SIDE(d_pred, d_gt, mask):
    batch_size = d_pred.size(0)
    diff = d_pred - d_gt
    diff = diff * mask
    avg = diff.view(batch_size, -1).sum(1) / (mask.view(batch_size, -1).sum(1))
    score = (diff - avg.view(batch_size,1,1))**2 * mask
    SIDE = ((score.view(batch_size, -1).sum(1) / mask.view(batch_size, -1).sum(1)) ** 0.5 + norm_diff)
    return SIDE


def cal_MAD(n1, n2, mask):
    dist = (n1*n2).sum(3).clamp(-1,1).acos() /np.pi*180
    return dist*mask