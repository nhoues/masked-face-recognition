from facenet_pytorch import InceptionResnetV1

import torch.nn as nn
from torch.nn import functional as F


class FaceNet(nn.Module):
    def __init__(self, pretrained=True):
        super(FaceNet, self).__init__()
        if pretrained:
            self.model = InceptionResnetV1(
                classify=True, num_classes=1, pretrained="vggface2"
            )
        else:
            self.model = InceptionResnetV1(
                classify=True, num_classes=1, pretrained=None
            )

    def forward(self, x):
        return self.model(x)
