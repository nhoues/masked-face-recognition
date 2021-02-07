import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A

from PIL import Image


class MaskFaceDetectionDataset:
    def __init__(self, paths, targets, resize_to=(224, 224), is_train=True):
        self.image_path = paths
        self.target = targets

        if is_train:
            train_transform = [
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.9
                ),
                A.GridDistortion(p=0.5),
                A.Resize(resize_to[0], resize_to[1], p=1.0),
                A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0, p=1.0,),
            ]
            self.aug = A.Compose(train_transform, p=1)
        else:
            self.aug = A.Compose(
                [
                    A.Resize(resize_to[0], resize_to[1], p=1.0),
                    A.Normalize(
                        mean=[0.485], std=[0.229], max_pixel_value=255.0, p=1.0,
                    ),
                ],
                p=1.0,
            )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        out = dict()
        image = Image.open(self.image_path[item])
        image = self.aug(image=np.array(image))["image"]
        image = np.transpose(image, (2, 0, 1)).astype(float)
        out["image"] = torch.tensor(image, dtype=torch.float)
        out["target"] = torch.tensor(self.target[item], dtype=torch.long)
        return out
