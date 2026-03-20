import torch
import torch.nn as nn

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNet34_UNet, self).__init__()
        """
        TODO: 
        1. 從零開始實作 ResNet34 的架構作為 Encoder。
        2. 結合 UNet 的 Decoder 架構。
        不可呼叫 torchvision.models.resnet34，也不可載入預訓練權重。
        """
        pass

    def forward(self, x):
        # TODO: 定義前向傳播邏輯
        pass
