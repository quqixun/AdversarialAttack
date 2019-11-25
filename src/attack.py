import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
# import matplotlib.pyplot as plt

from tqdm import *
from PIL import Image
from imageio import imwrite
from torchvision import transforms
from attack_models import inception_v3


class Attack(object):

    CHANNEL_MEAN = np.array([0.485, 0.456, 0.406])
    CHANNEL_STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, epsilon=32, alpha=1,
                 num_iters=None, num_threads=1,
                 early_stopping=10):
        torch.set_num_threads(num_threads)

        self.alpha = alpha / 255
        self.epsilon = epsilon / 255
        self.early_stopping = early_stopping
        self.use_cuda = torch.cuda.is_available()

        if num_iters is not None:
            self.num_iters = num_iters
        else:
            self.num_iters = int(min(epsilon + 4, epsilon * 1.25))

        self.model = inception_v3(pretrained=True)
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.CHANNEL_MEAN,
                std=self.CHANNEL_STD
            ),
        ])
        return

    def forward(self, image_path, true_label, target_label):
        print()
        image = self.preprocess(Image.open(image_path))
        image = image.unsqueeze(0)
        origin_image = image.clone()

        true_label = torch.LongTensor([true_label - 1])
        target_label = torch.LongTensor([target_label - 1])

        if self.use_cuda:
            image = image.cuda()
            true_label = true_label.cuda()
            origin_image = origin_image.cuda()
            target_label = target_label.cuda()

        true_ce = nn.CrossEntropyLoss()
        target_ce = nn.CrossEntropyLoss()

        best_loss, best_adv_image = None, None
        for i in range(self.num_iters):
            image.requires_grad = True
            pred, _ = self.model(image)

            true_ce_loss = true_ce(pred, true_label)
            target_ce_loss = target_ce(pred, target_label)
            loss = 1 / (true_ce_loss + 1e-8) + target_ce_loss

            grad = torch.autograd.grad(
                loss, image,
                retain_graph=False,
                create_graph=False
            )[0]

            perturbation = self.alpha * grad.sign()
            perturbation = torch.clamp(
                (image + perturbation) - origin_image,
                min=-self.epsilon, max=self.epsilon
            )
            adv_image = image - perturbation
            image = adv_image.detach()

            print('Image {} - Iter {} - Loss: {}'.format(image_path, i, loss), end='\r')

            if best_loss is None or loss.item() < best_loss:
                best_adv_image = adv_image
                best_loss = loss.item()

        pred, _ = self.model(best_adv_image)
        pred = torch.softmax(pred, dim=1)
        pred = pred.data.cpu().numpy().flatten()
        pred_label = np.argmax(pred) + 1

        adv_image = adv_image.squeeze(0)
        adv_image = adv_image.data.cpu().numpy()
        adv_image = np.transpose(adv_image, (1, 2, 0))
        adv_image = adv_image * self.CHANNEL_STD + self.CHANNEL_MEAN
        adv_image = (adv_image * 255).astype(np.uint8)
        return adv_image, pred_label


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    images_dir = '../data/images'
    output_dir = '../data/adv_images'
    create_dir(output_dir)
    dev_df = pd.read_csv('../data/dev.csv')

    attack = Attack(epsilon=1, alpha=10, num_iters=50, num_threads=2)

    num_error = 0
    for i, sample in tqdm(dev_df.iterrows(), total=len(dev_df), ncols=80):
        image_path = os.path.join(images_dir, sample['ImageId'])
        true_label, target_label = sample['TrueLabel'], sample['TargetClass']
        adv_image, pred_label = attack.forward(image_path, true_label, target_label)

        adv_image_path = os.path.join(output_dir, sample['ImageId'])
        imwrite(adv_image_path, adv_image)

        if pred_label != target_label:
            num_error += 1

    error_rate = num_error / len(dev_df)
    print('\nError rate: {:.6f}'.format(error_rate))
