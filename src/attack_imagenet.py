import os
import torch
import numpy as np

from tqdm import *
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from attack_models import inception_v3


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
torch.set_num_threads(6)


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


output_dir = '../data/imagenet'
create_dir(output_dir)

model = inception_v3(pretrained=True)
if use_cuda:
    model.cuda()
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

imagenet_dir = '/data1/pub/dataset/imagenet/train'
classes_dirs = os.listdir(imagenet_dir)

for class_name in tqdm(classes_dirs, ncols=75):
    class_dir = os.path.join(imagenet_dir, class_name)
    samples = os.listdir(class_dir)

    class_feats, preds = [], []
    for sample in tqdm(samples, ncols=75):
        sample_name = sample.split('.')[0].split('_')[1]
        image_path = os.path.join(imagenet_dir, class_dir, sample)
        image = Image.open(image_path)
        image = image.convert('RGB')

        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_var = Variable(image_tensor)
        if use_cuda:
            image_var = image_var.cuda()

        try:
            pred, feat = model(image_var)
            pred = torch.softmax(pred, dim=1)
            pred = pred.data.cpu().numpy().flatten()
            class_idx = np.argmax(pred) + 1
            preds.append(class_idx)

            feat = feat.view(1, -1)
            feat_np = feat.data.cpu().numpy().flatten()
            class_feats.append(list(feat_np))
        except Exception:
            continue

    preds = np.array(preds)
    class_no = str(np.argmax(np.bincount(preds)))
    class_output_dir = os.path.join(output_dir, class_no)
    create_dir(class_output_dir)

    np.save(os.path.join(class_output_dir, 'mean.npy'), np.mean(class_feats, axis=0))
    np.save(os.path.join(class_output_dir, 'std.npy'), np.std(class_feats, axis=0))
