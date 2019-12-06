import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import transforms


class AttackWhiteBox(object):

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    # RANGE: [(0 - mean) / std, (1 - mean) / std]
    RANGE = np.array([[-2.1179, 2.2489], [-2.0357, 2.4285], [-1.8044, 2.6400]])

    def __init__(self, model, input_size=299, epsilon=16, alpha=5,
                 num_iters=50, early_stopping=None, num_threads=1,
                 use_cuda=True):
        torch.set_num_threads(num_threads)

        self.alpha = alpha / 255
        self.num_iters = num_iters
        self.epsilon = epsilon / 255
        self.early_stopping = early_stopping
        self.use_cuda = torch.cuda.is_available() and use_cuda

        self.input_size = input_size
        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        self.model = model.eval()
        if self.use_cuda:
            self.model.cuda()

        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        return

    def __call__(self, image_path, label, target=False):
        self.target = target
        adv_image, pred_label = self.forward(image_path, label)
        return adv_image, pred_label

    def forward(self, image_path, label):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        origin = image.clone().detach()
        label = torch.LongTensor([label - 1])
        if self.use_cuda:
            image, origin, label = image.cuda(), origin.cuda(), label.cuda()

        num_no_improve, best_loss, best_adv_image = 0, None, None
        for i in range(self.num_iters):
            image.requires_grad = True
            pred = self.model(image)
            loss = self.__loss(pred, label, image, origin)

            if best_loss is None or loss.item() < best_loss:
                best_adv_image = image.clone()
                best_loss = loss.item()
                num_no_improve = 0
            else:
                num_no_improve += 1
            if self.__stop(num_no_improve):
                break

            image = self.__PGD(loss, image, origin)
            image_clamp = self.__clamp(image)
            image = image_clamp.detach()

        adv_image, pred_label = self.__post_process(best_adv_image)
        return adv_image, pred_label

    def __loss(self, pred, label, image, origin):
        def compute_ce_loss(p, l):
            ce_loss = self.ce_loss(p, l)
            if not self.target:
                ce_loss = 1 / torch.clamp(ce_loss, min=1e-8)
            return ce_loss

        ce_loss = compute_ce_loss(pred, label)
        return ce_loss  # + self.l1_loss(image, origin)

    def __stop(self, num_no_improve):
        return (self.early_stopping is not None) and \
            (num_no_improve == self.early_stopping)

    def __PGD(self, loss, image, origin):
        grad = torch.autograd.grad(loss, image, retain_graph=False)[0]
        perturbation = image - self.alpha * grad.sign() - origin
        perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
        return origin - perturbation

    def __clamp(self, image):
        image_clamp = []
        for i in range(image.size()[1]):
            cmin, cmax = self.RANGE[i]
            image_i = torch.clamp(image[0][i], min=cmin, max=cmax)
            image_clamp.append(image_i)
        image_clamp = torch.stack(image_clamp, dim=0).unsqueeze(0)
        return image_clamp

    def __post_process(self, best_adv_image):
        pred = self.model(best_adv_image)
        pred = torch.softmax(pred, dim=1)
        pred = pred.data.cpu().numpy().flatten()
        pred_label = np.argmax(pred) + 1

        adv_image = best_adv_image.squeeze(0).data.cpu().numpy()
        adv_image = np.transpose(adv_image, (1, 2, 0))
        adv_image = adv_image * self.STD + self.MEAN
        adv_image = np.round(adv_image * 255.0)
        return adv_image.astype(np.uint8), pred_label


if __name__ == '__main__':
    import os
    import pandas as pd

    from tqdm import *
    from pretrainedmodels import *
    from torchvision.models import *
    from imageio import imread, imwrite
    from attack_utils import create_dir
    from attack_utils import compute_score

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    images_dir = '../data/images'
    data = pd.read_csv('../data/dev.csv')
    adv_images_dir_base = '../data/white_adv_images'

    # models = [
    #     ['resnet18', resnet18(pretrained=True), 299],
    #     ['resnet34', resnet34(pretrained=True), 299],
    #     ['resnet50', resnet50(pretrained=True), 299],
    #     ['resnet101', resnet101(pretrained=True), 299],
    #     ['resnet152', resnet152(pretrained=True), 299],
    #     ['resnext50_32x4d', resnext50_32x4d(pretrained=True), 299],
    #     ['resnext101_32x8d', resnext101_32x8d(pretrained=True), 299],
    #     ['wide_resnet50_2', wide_resnet50_2(pretrained=True), 299],
    #     ['wide_resnet101_2', wide_resnet101_2(pretrained=True), 299]
    # ]

    models = [
        ['densenet121', densenet121(pretrained=True), 299],
        ['densenet161', densenet161(pretrained=True), 299],
        ['densenet169', densenet169(pretrained=True), 299],
        ['densenet201', densenet201(pretrained=True), 299],
        ['mobilenet_v2', mobilenet_v2(pretrained=True), 299],
        ['squeezenet1_0', squeezenet1_0(pretrained=True), 299],
        ['squeezenet1_1', squeezenet1_1(pretrained=True), 299],
        ['inception-v3', inception_v3(pretrained=True), 299],
        ['inception-v4', inceptionv4(pretrained='imagenet'), 299],
        ['inception-resnet-v2', inceptionresnetv2(pretrained='imagenet'), 299]
    ]

    for model_name, model, input_size in models:
        adv_images_dir = os.path.join(adv_images_dir_base, model_name)
        create_dir(adv_images_dir)

        attack = AttackWhiteBox(
            model=model, input_size=input_size,
            epsilon=16, alpha=1, num_iters=100,
            early_stopping=10, num_threads=2
        )

        print('-' * 75)
        print('Attacking - Model[{}]'.format(model_name))
        scores = []
        for i, sample in tqdm(data.iterrows(), total=len(data), ncols=75):
            true_label = sample['TrueLabel']
            target_label = sample['TargetClass']

            image_path = os.path.join(images_dir, sample['ImageId'])
            adv_image, pred_label = attack(image_path, target_label, target=True)

            score = compute_score(imread(image_path), adv_image, pred_label,
                                  true_label, target_label, pixel_limit=32)
            scores.append(score)

            adv_image_path = os.path.join(adv_images_dir, sample['ImageId'])
            imwrite(adv_image_path, adv_image)

        mean_score = np.mean(scores)
        print('\nAttack Score: {:.6f}'.format(mean_score))
        print('-' * 75)
