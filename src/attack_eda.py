import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import *
from imageio import imread
from matplotlib_venn import venn2
from torch.autograd import Variable
from torchvision import models, transforms


MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

output_dir = '../data/eda'
images_dir = '../data/images'
dev = pd.read_csv('../data/dev.csv')

imagenet_classes_txt = '../data/imagenet_classes.txt'
with open(imagenet_classes_txt, 'r') as f:
    lines = f.readlines()
imagenet_classes = {}
for line in lines:
    class_info = line.strip().split()[1:]
    imagenet_classes[int(class_info[0])] = class_info[1]

true_classes = dev['TrueLabel'].unique()
true_classes_count = dev['TrueLabel'].value_counts()
target_classes = dev['TargetClass'].unique()
target_classes_count = dev['TargetClass'].value_counts()
tre_target_intersect = set(true_classes).intersection(set(target_classes))

plt.figure(figsize=(15, 10))
plt.title('Number of images in each class')
plt.bar(true_classes_count.index, true_classes_count,
        label='True Classes: {}'.format(len(true_classes)))
plt.bar(target_classes_count.index, target_classes_count, alpha=0.75,
        label='Target Classes: {}'.format(len(target_classes)))
plt.ylabel('Number of Images')
plt.xlabel('Classes')
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'classes.png'))
plt.close()

plt.figure()
venn2(subsets=(len(true_classes) - len(tre_target_intersect),
               len(target_classes) - len(tre_target_intersect),
               len(tre_target_intersect)),
      set_labels=('True Classes', 'Target Classes'))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'venn.png'))
plt.close()

model = models.inception_v3(pretrained=True)
model.eval()

imagenet_pred = []
for _, row in tqdm(dev.iterrows(), total=len(dev), ncols=75):
    image_name = row['ImageId']
    true_class = row['TrueLabel']
    target_class = row['TargetClass']
    image_path = os.path.join(images_dir, image_name)
    image = imread(image_path) / 255

    image_ = (image - MEAN) / STD
    image_ = np.transpose(image_, (2, 0, 1))
    image_ = np.expand_dims(image_, 0)

    image_var = Variable(torch.Tensor(image_))
    pred = model(image_var)
    pred = torch.softmax(pred, dim=1)
    pred = pred.data.cpu().numpy().flatten()
    class_idx = np.argmax(pred) + 1
    pred_class = imagenet_classes[class_idx]

    imagenet_pred.append([image_name, true_class, target_class, class_idx])

    # title_str = '{}\n{}({}-{}) --> {}'.format(
    #     image_name, true_class, pred_class, class_idx, target_class)
    # plt.figure()
    # plt.title(title_str)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

pred_columns = dev.columns.tolist() + ['Pred']
pred_df = pd.DataFrame(data=imagenet_pred, columns=pred_columns)
pred_csv = os.path.join(output_dir, 'pred.csv')
pred_df.to_csv(pred_csv, index=False)

dev_correct = pred_df.copy()
for i, row in dev_correct.iterrows():
    if row['TrueLabel'] != row['Pred']:
        dev_correct.loc[i, 'TrueLabel'] = row['Pred']
dev_correct.drop(columns=['Pred'], inplace=True)
dev_correct.to_csv('../data/dev_correct.csv', index=False)
