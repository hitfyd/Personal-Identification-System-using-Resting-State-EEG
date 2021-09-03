import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, channels=64, points=80, ica_enable=True, conv_layers=3, classes=109,
                 ica_output=32, conv_filters=32, fc1_output=512, dropout=0.5):
        super().__init__()
        self.ica_enable, self.conv_layers, self.dropout = ica_enable, conv_layers, dropout
        # 独立成分分析
        self.ICA = nn.Linear(channels, ica_output)
        # 卷积层1
        self.Conv_1 = nn.Conv2d(1, conv_filters, (5, 3), (2, 1), padding=(2, 1))
        # 卷积层2
        self.Conv_2 = nn.Conv2d(conv_filters, conv_filters, (3, 3), (1, 1), (1, 1))
        # 卷积层3
        self.Conv_3 = nn.Conv2d(conv_filters, conv_filters, (3, 3), (1, 1), (1, 1))
        # 计算稠密层数量
        out_points = int(points / 8)  # 采样点数
        if self.ica_enable:
            out_channels = int(ica_output / 2)  # 通道数
        else:
            if self.conv_layers != 3:
                out_channels = int(channels / 2)
            else:
                out_channels = int(channels / 4)
        # 全连接层
        self.fc1 = nn.Linear(conv_filters * out_points * out_channels, fc1_output)
        # 输出层
        self.fc2 = nn.Linear(fc1_output, classes)

    def forward(self, x):
        batch_size = x.size(0)
        # print(x.shape, batch_size)
        out = x
        # 是否使用独立成分分析
        if self.ica_enable:
            out = self.ICA(x)

        # conv1
        out = self.Conv_1(out)
        out = F.elu(out)
        out = F.max_pool2d(out, (2, 1), (2, 1))  # ->batch*1*100*32 时间维度上池化因子为2，步长为2
        # conv2
        out = self.Conv_2(out)
        out = F.elu(out)
        if self.conv_layers == 3:
            out = F.max_pool2d(out, (1, 2), (1, 2))
        else:
            out = F.max_pool2d(out, (2, 2), (2, 2))  # 只有2层卷积时，为了降低参数数量加大池化因子

        # conv3
        if self.conv_layers == 3:
            out = self.Conv_3(out)
            out = F.elu(out)
            if self.ica_enable:
                out = F.max_pool2d(out, (2, 1), (2, 1))
            else:
                out = F.max_pool2d(out, (2, 2), (2, 2))
        # 稠密
        out = out.view(batch_size, -1)

        out = self.fc1(out)
        out = F.elu(out)
        out = F.dropout(out, self.dropout)

        out = self.fc2(out)

        # # 启用独立成分分析时多执行一次Softmax
        # if self.ica_enable:
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out
