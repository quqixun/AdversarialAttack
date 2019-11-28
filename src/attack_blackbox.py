import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from imageio import imread
from numpy.linalg import norm
from torchvision import transforms


class AttackBlackBox(object):

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    # RANGE: [(0 - mean) / std, (1 - mean) / std]
    RANGE = np.array([[-2.1179, 2.2489], [-2.0357, 2.4285], [-1.8044, 2.6400]])

    def __init__(self, model, input_size=299, pixel_limit=32, epsilon=16, alpha=5,
                 num_iters=50, early_stopping=None, num_threads=1, use_cuda=True):
        torch.set_num_threads(num_threads)

        self.alpha = alpha / 255
        self.num_iters = num_iters
        self.epsilon = epsilon / 255
        self.pixel_limit = pixel_limit
        self.early_stopping = early_stopping
        self.use_cuda = torch.cuda.is_available() and use_cuda

        self.model = model
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        return


if __name__ == '__main__':
    import os
    import pandas as pd

    from tqdm import *
    from imageio import imwrite
    from torchvision.models import inception_v3
