import torch
import torch.nn as nn
import torch.nn.functional as F
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)


    def forward(self, x):
        return self.features(x)



def mobilenetv3(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s    # input c       output c
        [3, 1, 16, 0, 0, 1],    # 16              16
        [3, 4, 24, 0, 0, 2],    # 64              24
        [3, 3, 24, 0, 0, 1],    # 72              24
        [5, 3, 40, 1, 0, 2],    # 72              40
        [5, 3, 40, 1, 0, 1],    # 120             40
        [5, 3, 40, 1, 0, 1],    # 120             40
        [3, 6, 80, 0, 1, 2],    # 240             80
        [3, 2.5, 80, 0, 1, 1],  # 200             80
        [3, 2.3, 80, 0, 1, 1],  # 184             80
        [3, 2.3, 80, 0, 1, 1],  # 184             80
        [3, 6, 112, 1, 1, 1],   # 480             112
        [3, 6, 112, 1, 1, 1]    # 672             112
    ]
    return MobileNetV3(cfgs, **kwargs)

class RgbdMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.RGB_MobileNet = mobilenetv3()
        self.D_MobileNet = mobilenetv3()
        self.convt1 = nn.ConvTranspose2d(112, 112, 3, stride=2, padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(112, 80, 5, stride=2, padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(80, 40, 3, stride=2, padding=1, output_padding=1)
        self.convt4 = nn.ConvTranspose2d(40, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv0 = nn.ConvTranspose2d(224,112,1,stride=1,padding=0,output_padding=0)
        self.SE = SELayer(224)

        self.pos_output = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x, y):
        x = self.RGB_MobileNet(x)

        y = self.D_MobileNet(y)

        # Fusion layer
        x = torch.cat((x,y),1) #concact x and y
        x=self.SE(x) # channelwise attention
        x=self.conv0(x) #pointwise attention

        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt4(x))

        pos_output = self.pos_output(x)
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

