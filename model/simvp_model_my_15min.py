import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

from simvp_modules import (ConvSC, gInception_ST,TAUSubBlock)


# class TemporalChannelAttentionWeighting(nn.Module):
#     def __init__(self, time_steps, channels):
#         super().__init__()
#         self.time_steps = time_steps
#         self.channels = channels
#
#         # 池化到 [B,C,T,1,1]
#         self.avg_pool = nn.AdaptiveAvgPool3d((time_steps,1,1))
#         self.max_pool = nn.AdaptiveMaxPool3d((time_steps,1,1))
#         # 用于时间卷积的层：kernel_size=(10,1,1), padding=0
#         self.conv1 = nn.Conv3d(channels, channels, (time_steps,1,1), padding=0, bias=False)
#         self.conv2 = nn.Conv3d(channels, channels, (time_steps,1,1), padding=0, bias=False)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         # 精确计算非对称 pad: pad_left=4, pad_right=5
#         self.pad = (0,0, 0,0, (time_steps-1)//2, (time_steps-1) - (time_steps-1)//2)
#
#     def forward(self, x):
#         # x: [B, T, C, H, W] -> [B, C, T, H, W]
#         x_c = x.permute(0,2,1,3,4)
#
#         # 池化后 [B, C, T, 1, 1]
#         a = self.avg_pool(x_c)
#         m = self.max_pool(x_c)
#
#         # 手动在时间维度上 pad
#         a = F.pad(a, self.pad)
#         m = F.pad(m, self.pad)
#
#         # 卷积 + 激活
#         a = self.conv2(self.relu(self.conv1(a)))
#         m = self.conv2(self.relu(self.conv1(m)))
#
#         # 注意力权重 & 加权
#         attn = self.sigmoid(a + m)   # [B, C, T, 1, 1]
#         out = attn * x_c           # [B, C, T, H, W]
#
#         # 恢复到 [B, T, C, H, W]
#         return rearrange(out, 'b c t h w -> b t c h w')
class TemporalChannelAttentionWeighting(nn.Module):
    def __init__(self, time_steps, channels, kernel_size=3):
        super(TemporalChannelAttentionWeighting, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((time_steps, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((time_steps, 1, 1))

        padding = (kernel_size // 2, 0, 0)
        self.fc1 = nn.Conv3d(channels, channels, (time_steps, 1, 1), padding=padding, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(channels, channels, (time_steps, 1, 1), padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  #[batch_size, T, C, H, W]

        x_reshaped = x.permute(0, 2, 1, 3, 4)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x_reshaped))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x_reshaped))))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        result = attention * x_reshaped
        result = rearrange(result, "b c t h w -> b t c h w")
        return result
class TemporalChannelAttentionWeighting_adapt(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(TemporalChannelAttentionWeighting_adapt, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]

        # 自适应池化到当前时间长度
        avg_pool = F.adaptive_avg_pool3d(x_reshaped, (T, 1, 1))
        max_pool = F.adaptive_max_pool3d(x_reshaped, (T, 1, 1))

        # 动态设置 kernel_size（不能超过 T）
        k = min(self.kernel_size, T)
        padding = (k // 2, 0, 0)
        conv1 = nn.Conv3d(self.channels, self.channels, kernel_size=(k,1,1), padding=padding, bias=False).to(x.device)
        conv2 = nn.Conv3d(self.channels, self.channels, kernel_size=(k,1,1), padding=padding, bias=False).to(x.device)

        # 分支计算
        avg_out = conv2(self.relu(conv1(avg_pool)))
        max_out = conv2(self.relu(conv1(max_pool)))
        attention = self.sigmoid(avg_out + max_out)

        out = attention * x_reshaped
        out = out.permute(0, 2, 1, 3, 4)  # -> [B, T, C, H, W]
        return out

class SimVP_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T*hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        # self.TemporalConvBlock1 = nn.Sequential(
        #     nn.Conv2d(T*C, T*C, kernel_size=3, padding=1),
        #     nn.GroupNorm(num_groups=T, num_channels=T*C),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Dropout2d(p=0.5),
        #     nn.Conv2d(T*C, T*C, kernel_size=3, padding=1),
        #     nn.GroupNorm(num_groups=T, num_channels=T*C),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Dropout2d(p=0.5),
        # )
        # self.TemporalConvBlock2 = nn.Sequential(
        #     nn.Conv2d(T*C, T*C, kernel_size=3, padding=1),
        #     nn.GroupNorm(num_groups=T, num_channels=T*C),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Dropout2d(p=0.5),
        #     nn.Conv2d(T*C, T*C, kernel_size=3, padding=1),
        #     nn.GroupNorm(num_groups=T, num_channels=T*C),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Dropout2d(p=0.5),
        # )
        self.conv1 = nn.Conv2d(T*C, T*C*2, kernel_size=1)
        self.conv2 = nn.Conv2d(T*C, T*C*3, kernel_size=1)
        self.model1 = TemporalChannelAttentionWeighting(T,C)
        self.model2 = TemporalChannelAttentionWeighting(T,C)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T*C, H, W)
        tme_x2 = self.conv1(Y)
        tme_x2 = rearrange(tme_x2, "b (c t) h w -> b c t h w", c=T * 2)
        in_x2 = tme_x2[:, T:, :, :, :]
        in_x2 = rearrange(in_x2, "b c t h w -> b (c t) h w")

        tmp_x3 = self.conv2(in_x2)
        in_x3 = rearrange(tmp_x3, "b (c t) h w -> b c t h w", c=T * 3)
        in_x3 = in_x3[:, T * 2:, :, :, :]
        Y = rearrange(Y, "b (c t) h w -> b c t h w", c=T)
        y1 = Y + self.model1(in_x3)
        in_x2 = rearrange(in_x2, "b (c t) h w -> b c t h w", c=T)
        y2 = in_x2 + self.model2(y1)
        y = torch.cat((Y, y1, y2), dim=1)

        return y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_in,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y

if __name__ == '__main__':
    model_ori = SimVP_Model((5, 20, 64, 64),hid_S = 64,hid_T = 512,N_T = 8,N_S = 4,
                                       model_type="tau", drop_path=0.1, spatio_kernel_enc=3, spatio_kernel_dec = 3)
    print(model_ori)
    x=torch.randn(16, 5, 20, 64, 64)
    y = model_ori(x)
    print(y.shape)
