import os
import torch
import numpy as np

from tqdm import *
from imageio import imread
from torchvision import models
from torch.autograd import Variable


imagenet_classes_txt = '../data/imagenet_classes.txt'
with open(imagenet_classes_txt, 'r') as f:
    lines = f.readlines()
imagenet_classes = {}
for line in lines:
    class_info = line.strip().split()[:2]
    imagenet_classes[class_info[0]] = class_info[1]

output_dir = '../data/imagenet'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

model = models.inception_v3(pretrained=True)
model.eval()

imagenet_dir = ''
classes_dirs = os.listdir(imagenet_dir)
for class_dir in tqdm(classes_dirs, ncols=75):
    class_no = imagenet_classes[class_dir]
    class_output_dir = os.path.join(output_dir, class_no)
    samples = os.listdir(imagenet_dir, class_dir)
    for sample in samples:
        image_path = os.path.join(imagenet_dir, class_dir, sample)
        image = imread(image_path) / 255

        image_ = (image - MEAN) / STD
        image_ = np.transpose(image_, (2, 0, 1))
        image_ = np.expand_dims(image_, 0)

        image_var = Variable(torch.Tensor(image_))
        pred = model(image_var)


