from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models


class FaceBinaryClassifier(nn.Module):
    def __init__(self, backbone: Literal["mobilenet_v3_small", "mobilenet_v3_large"] = "mobilenet_v3_small"):
        super().__init__()
        if backbone == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            in_features = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_features, 1)
        elif backbone == "mobilenet_v3_large":
            net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            in_features = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.squeeze(1)
