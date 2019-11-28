import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from imageio import imread
from numpy.linalg import norm
from torchvision import transforms


class AttackWhiteBox(object):

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

    def __call__(self, image_path, true_label, target_label, return_score=False):
        adv_image, pred_label = self.forward(image_path, true_label, target_label)
        adv_image_uint8 = adv_image.astype(np.uint8)

        if not return_score:
            return adv_image_uint8
        else:
            score = self.score(adv_image, pred_label, image_path, true_label, target_label)
            return adv_image_uint8, score

    def forward(self, image_path, true_label, target_label):
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
            pred = self.model(image)

            true_ce_loss = 1 / (true_ce(pred, true_label) + 1e-8)
            target_ce_loss = target_ce(pred, target_label)
            diff_loss = diff(image, origin_image)
            loss = true_ce_loss + target_ce_loss + diff_loss

            if best_loss is None or loss.item() < best_loss:
                best_adv_image = image.clone()
                best_loss = loss.item()
                num_no_improve = 0
            else:
                num_no_improve += 1

            if self.early_stopping is not None and \
               num_no_improve == self.early_stopping:
                break

            # ---------------------------------- PGD ------------------------------------
            grad = torch.autograd.grad(loss, image, retain_graph=False)[0]
            perturbation = image - self.alpha * grad.sign() - origin_image
            perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
            image = origin_image - perturbation
            # ---------------------------------- PGD ------------------------------------

            image_clamp = []
            for i in range(image.size()[1]):
                cmin, cmax = self.RANGE[i]
                image_i = torch.clamp(image[0][i], min=cmin, max=cmax)
                image_clamp.append(image_i)
            image_clamp = torch.stack(image_clamp, dim=0).unsqueeze(0)
            image = image_clamp.detach()

        pred = self.model(best_adv_image)
        pred = torch.softmax(pred, dim=1)
        pred = pred.data.cpu().numpy().flatten()
        pred_label = np.argmax(pred) + 1

        best_adv_image = best_adv_image.squeeze(0).data.cpu().numpy()
        best_adv_image = np.transpose(best_adv_image, (1, 2, 0))
        best_adv_image = best_adv_image * self.STD + self.MEAN
        best_adv_image = np.round(best_adv_image * 255.0)
        return best_adv_image, pred_label

    def score(self, adv_image, pred_label, image_path,
              true_label, target_label):
        image = imread(image_path).astype(float).flatten()
        norm_score = norm(adv_image.flatten() - image, ord=np.inf)

        score = 0
        if pred_label == true_label:
            score = 0
        elif pred_label != true_label and pred_label != target_label:
            score = 2 * (2 - norm_score / self.pixel_limit)
        elif pred_label == target_label:
            score = 5 * (2 - norm_score / self.pixel_limit)
        else:
            pass
        return score


if __name__ == '__main__':
    import os
    import pandas as pd

    from tqdm import *
    from imageio import imwrite
    from torchvision.models import inception_v3
    # from pretrainedmodels import inceptionv4, inceptionresnetv2

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    images_dir = '../data/images'
    data = pd.read_csv('../data/dev.csv')
    adv_images_dir = '../data/white_adv_images'
    if not os.path.isdir(adv_images_dir):
        os.makedirs(adv_images_dir)

    attack = AttackWhiteBox(
        model=inception_v3(pretrained=True),
        # model=inceptionv4(pretrained='imagenet'),
        # model=inceptionresnetv2(pretrained='imagenet'),
        input_size=299, pixel_limit=32,
        epsilon=16, alpha=5, num_iters=50,
        early_stopping=None, num_threads=2
    )

    scores = []
    for i, sample in tqdm(data.iterrows(), total=len(data), ncols=80):
        true_label = sample['TrueLabel']
        target_label = sample['TargetClass']

        image_path = os.path.join(images_dir, sample['ImageId'])
        adv_image, score = attack(image_path, true_label, target_label, return_score=True)
        scores.append(score)

        adv_image_path = os.path.join(adv_images_dir, sample['ImageId'])
        imwrite(adv_image_path, adv_image)

    mean_score = np.mean(scores)
    print('Attack Score: {:.6f}'.format(mean_score))
