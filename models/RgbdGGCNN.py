import torch
import torch.nn as nn
import torch.nn.functional as F

filter_sizes = [32, 16, 8, 16, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]


class RgbdGGCNN(nn.Module):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.BN1 = nn.BatchNorm2d(filter_sizes[0])
        self.BN2 = nn.BatchNorm2d(filter_sizes[1])
        self.BN3 = nn.BatchNorm2d(filter_sizes[2])
        self.d_conv1 = nn.Conv2d(3, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.d_conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.d_conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.convt1 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)

        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x, y):
        x = F.relu(self.BN1(self.conv1(x)))
        x = F.relu(self.BN2(self.conv2(x)))
        x = F.relu(self.BN3(self.conv3(x)))

        y = F.relu(self.BN1(self.d_conv1(y)))
        y = F.relu(self.BN2(self.d_conv2(y)))
        y = F.relu(self.BN3(self.d_conv3(y)))

        x = torch.cat((x,y),1) #concact x and y

        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = torch.sigmoid(self.pos_output(x))
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)

        return pos_output, cos_output, sin_output

    def compute_loss(self, xc, yc, zc):
        z_pos, z_cos, z_sin = zc
        pos_pred, cos_pred, sin_pred = self(xc,yc)
        p_loss = F.mse_loss(pos_pred, z_pos)
        cos_loss = F.mse_loss(cos_pred, z_cos)
        sin_loss = F.mse_loss(sin_pred, z_sin)
        return {
            'loss': p_loss + cos_loss + sin_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred
            }
        }
