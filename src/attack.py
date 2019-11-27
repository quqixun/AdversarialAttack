import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import *
from PIL import Image
from imageio import imwrite
from torchvision import transforms
from attack_models import inception_v3


class Attack(object):

    CHANNEL_MEAN = np.array([0.485, 0.456, 0.406])
    CHANNEL_STD = np.array([0.229, 0.224, 0.225])
    CHANNEL_RANGE = np.array([[-2.1179, 2.2489],
                              [-2.0357, 2.4285],
                              [-1.8044, 2.6400]])

    def __init__(self, epsilon=1, alpha=10, num_iters=None,
                 early_stopping=None, num_threads=1):
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
        origin_image = image.clone().detach()

        true_label = torch.LongTensor([true_label - 1])
        target_label = torch.LongTensor([target_label - 1])

        if self.use_cuda:
            image = image.cuda()
            true_label = true_label.cuda()
            origin_image = origin_image.cuda()
            target_label = target_label.cuda()

        true_ce = nn.CrossEntropyLoss()
        target_ce = nn.CrossEntropyLoss()
        diff = nn.L1Loss()

        num_no_improve = 0
        best_loss, best_adv_image = None, None
        for i in range(self.num_iters):
            image.requires_grad = True
            pred, _ = self.model(image)

            true_ce_loss = true_ce(pred, true_label)
            target_ce_loss = target_ce(pred, target_label)
            diff_loss = diff(image, origin_image)
            loss = 1 / (true_ce_loss + 1e-8) + target_ce_loss + diff_loss
            print('Image {} - Iter {} - Loss: {}'.format(image_path, i + 1, loss), end='\r')

            if best_loss is None or loss.item() < best_loss:
                best_adv_image = image.clone()
                best_loss = loss.item()
                num_no_improve = 0
            else:
                num_no_improve += 1

            if self.early_stopping is not None and \
               num_no_improve == self.early_stopping:
                break

            grad = torch.autograd.grad(
                loss, image,
                retain_graph=False,
                create_graph=False
            )[0]

            adv_image = image - self.alpha * grad.sign()
            perturbation = adv_image - origin_image
            perturbation = torch.clamp(
                perturbation, min=-self.epsilon, max=self.epsilon
            )
            image = origin_image - perturbation

            image_clamp = []
            for i in range(image.size()[1]):
                cmin, cmax = self.CHANNEL_RANGE[i]
                image_i = torch.clamp(image[0][i], min=cmin, max=cmax)
                image_clamp.append(image_i)
            image_clamp = torch.stack(image_clamp, dim=0).unsqueeze(0)
            image = image_clamp.detach()

        pred, _ = self.model(best_adv_image)
        pred = torch.softmax(pred, dim=1)
        pred = pred.data.cpu().numpy().flatten()
        pred_label = np.argmax(pred) + 1

        best_adv_image = best_adv_image.squeeze(0)
        best_adv_image = best_adv_image.data.cpu().numpy()
        best_adv_image = np.transpose(best_adv_image, (1, 2, 0))
        best_adv_image = best_adv_image * self.CHANNEL_STD + self.CHANNEL_MEAN
        best_adv_image = np.round(best_adv_image * 255).astype(np.uint8)

        return best_adv_image, pred_label


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    images_dir = '../data/images'
    output_dir = '../data/adv_images_e8'
    create_dir(output_dir)
    dev_df = pd.read_csv('../data/dev.csv')

    attack = Attack(epsilon=16, alpha=5, num_iters=200,
                    early_stopping=20, num_threads=1)

    num_error, pred_labels = 0, []
    for i, sample in tqdm(dev_df.iterrows(), total=len(dev_df), ncols=80):
        image_path = os.path.join(images_dir, sample['ImageId'])
        true_label, target_label = sample['TrueLabel'], sample['TargetClass']
        adv_image, pred_label = attack.forward(image_path, true_label, target_label)

        adv_image_path = os.path.join(output_dir, sample['ImageId'])
        imwrite(adv_image_path, adv_image)

        pred_labels.append(pred_label)
        if pred_label != target_label:
            num_error += 1

        print('\n', image_path, target_label, pred_label)

    dev_df['PredLabel'] = pred_labels
    dev_df.to_csv('../data/dev_pred.csv', index=False)

    error_rate = num_error / len(dev_df)
    print('\n\nError rate: {:.6f}'.format(error_rate))
