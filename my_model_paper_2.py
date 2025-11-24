import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F

class TemporalChannelAttentionWeighting(nn.Module):
    def __init__(self, time_steps, channels):
        super().__init__()
        self.time_steps = time_steps
        self.channels = channels

        # 池化到 [B,C,T,1,1]
        self.avg_pool = nn.AdaptiveAvgPool3d((time_steps,1,1))
        self.max_pool = nn.AdaptiveMaxPool3d((time_steps,1,1))
        # 用于时间卷积的层：kernel_size=(10,1,1), padding=0
        self.conv1 = nn.Conv3d(channels, channels, (time_steps,1,1), padding=0, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, (time_steps,1,1), padding=0, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # 精确计算非对称 pad: pad_left=4, pad_right=5
        self.pad = (0,0, 0,0, (time_steps-1)//2, (time_steps-1) - (time_steps-1)//2)

    def forward(self, x):
        # x: [B, T, C, H, W] -> [B, C, T, H, W]
        x_c = x.permute(0,2,1,3,4)

        # 池化后 [B, C, T, 1, 1]
        a = self.avg_pool(x_c)
        m = self.max_pool(x_c)

        # 手动在时间维度上 pad
        a = F.pad(a, self.pad)
        m = F.pad(m, self.pad)

        # 卷积 + 激活
        a = self.conv2(self.relu(self.conv1(a)))
        m = self.conv2(self.relu(self.conv1(m)))

        # 注意力权重 & 加权
        attn = self.sigmoid(a + m)   # [B, C, T, 1, 1]
        out = attn * x_c           # [B, C, T, H, W]

        # 恢复到 [B, T, C, H, W]
        return rearrange(out, 'b c t h w -> b t c h w')

class STFGP(nn.Module):

    def __init__(self, time, channels, ):
        super(STFGP, self).__init__()

        self.conv1 = nn.Conv2d(time*channels, time*channels, kernel_size=1)
        self.conv2 = nn.Conv2d(time*channels, time*channels, kernel_size=1)

        self.model1 = TemporalChannelAttentionWeighting(time,channels)
        self.model2 = TemporalChannelAttentionWeighting(time,channels)
        self.T = time
    def forward(self, x):
        """x:  [B,T,C,H,W]"""
        in_x2 = self.conv1(x)
        in_x3 = self.conv2(in_x2)
        in_x2 = rearrange(in_x2, "b (c t) h w -> b c t h w",c=self.T)
        in_x3 = rearrange(in_x3, "b (c t) h w -> b c t h w",c=self.T)

        x = rearrange(x, "b (c t) h w -> b c t h w",c=self.T)

        y1 = x + self.model1(in_x3)
        y2 = in_x2 + self.model2(y1)

        y = torch.cat((x, y1, y2), dim=1)
        return y

class MY_model(nn.Module):

    def __init__(self, in_shape, model):
        super(MY_model, self).__init__()
        T, C, H, W = in_shape
        self.model = model
        self.stfgp = STFGP(T, C)
        ########################################加的model_1
        self.end = nn.Sequential(
            nn.Conv2d(T*6*C, T*6*C,kernel_size=3, padding=1),
            nn.BatchNorm2d(T*6*C),
            nn.ReLU(inplace=True),
            nn.Conv2d(T*6*C, T*6*C, kernel_size=3, padding=1),
            nn.BatchNorm2d(T*6*C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        y_15 = self.model(x_raw)
        y_10_15 = y_15[:, 10:, :, :, :]
        y_10_15 = y_10_15.reshape(B, T*C, H, W)
        y_15_30 = self.stfgp(y_10_15)
        y = torch.cat((y_15, y_15_30), dim=1)
        ########################################加的model_1
        # y = y.reshape(B, T*C*6, H, W)
        # y = self.end(y)
        # y = y.reshape(B, T*6, C, H, W)
        return y