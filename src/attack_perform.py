import os
import torch
import numpy as np
import pandas as pd

from tqdm import *
from PIL import Image
from imageio import imread
from numpy.linalg import norm
from torchvision import transforms


class AttackPerform(object):

    def __init__(self, model, input_size=299, pixel_limit=32, use_cuda=True, num_threads=1):
        torch.set_num_threads(num_threads)
        self.pixel_limit = pixel_limit
        self.use_cuda = torch.cuda.is_available() and use_cuda

        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.model = model
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()
        return

    def by_original(self, image_path, true_label):
        pred_label = self.forward(image_path)
        is_correct = (pred_label == true_label) * 1.0
        return pred_label, is_correct

    def by_adversarial(self, image_path, adv_image_path, true_label, target_label):
        score = 0
        image = imread(image_path).astype(float).flatten()
        adv_image = imread(adv_image_path).astype(float).flatten()
        norm_score = norm(adv_image - image, ord=np.inf)

        pred_label = self.forward(adv_image_path)
        if pred_label == true_label:
            score = 0
        elif pred_label != true_label and pred_label != target_label:
            score = 2 * (2 - norm_score / self.pixel_limit)
        elif pred_label == target_label:
            score = 5 * (2 - norm_score / self.pixel_limit)
        else:
            pass

        is_correct = (pred_label == target_label) * 1.0
        return pred_label, is_correct, score

    def forward(self, image_path):
        image = self.preprocess(Image.open(image_path))
        image = image.unsqueeze(0)
        if self.use_cuda:
            image = image.cuda()

        pred = self.model(image)
        pred = torch.softmax(pred, dim=1)
        pred = pred.data.cpu().numpy().flatten()
        pred_label = np.argmax(pred) + 1
        return pred_label


def predict_original(data, model, images_dir, model_name):
    print('-' * 75)
    print('Test {} by original images'.format(model_name))
    attack = AttackPerform(model=model, input_size=299, use_cuda=True)

    num_correct = 0
    for _, row in tqdm(data.iterrows(), total=len(data), ncols=75):
        image_file, true_label = row['ImageId'], row['TrueLabel']
        image_path = os.path.join(images_dir, image_file)
        pred_label, is_correct = attack.by_original(image_path, true_label)
        num_correct += is_correct

    accuracy = num_correct / len(data)
    print('Model {} accuracy: {:.6f}'.format(model_name, accuracy))
    print('-' * 75)
    return


def predict_adversarial(data, model, images_dir, adv_images_dir, model_name):
    print('-' * 75)
    print('Test {} by adversarial images'.format(model_name))
    attack = AttackPerform(model=model, input_size=299, use_cuda=True)

    num_correct, scores = 0, []
    for _, sample in tqdm(data.iterrows(), total=len(data), ncols=75):
        true_label = sample['TrueLabel']
        target_label = sample['TargetClass']
        image_path = os.path.join(images_dir, sample['ImageId'])
        adv_image_path = os.path.join(adv_images_dir, sample['ImageId'])

        pred_label, is_correct, score = \
            attack.by_adversarial(image_path, adv_image_path, true_label, target_label)
        num_correct += is_correct
        scores.append(score)

    mean_score = np.mean(scores)
    accuracy = num_correct / len(data)
    print('Model {} accuracy: {:.6f} score: {:.6f}'.format(model_name, accuracy, mean_score))
    print('-' * 75)
    return


if __name__ == '__main__':
    from torchvision.models import inception_v3
    from pretrainedmodels import inceptionv4, inceptionresnetv2

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    images_dir = '../data/images'
    adv_images_dir = '../data/adv_images'
    data = pd.read_csv('../data/dev.csv')
    models = {
        'inception-v3': inception_v3(pretrained=True),
        'inception-v4': inceptionv4(pretrained='imagenet'),
        'inception-resnet-v2': inceptionresnetv2(pretrained='imagenet')
    }

    # Original images
    # for model_name, model in models.items():
    #     predict_original(data, model, images_dir, model_name)

    #         Model        |   Accuracy(true)
    # ------------------------------------------
    #     inception v3     |     0.982730
    #     inception v4     |     0.981086
    #  inceptionresnet v2  |     0.988487

    # Adversarial images
    for model_name, model in models.items():
        predict_adversarial(data, model, images_dir, adv_images_dir, model_name)

    #         Model        |   Accuracy(target)   |    Score
    # ----------------------------------------------------------
    #     inception v3     |      0.969572        |   9.188425
    #     inception v4     |      0.000000        |   0.117188
    #  inceptionresnet v2  |      0.000000        |   0.083265
