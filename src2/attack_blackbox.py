import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from imageio import imread
from torchvision import transforms
from attack_utils import compute_score


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

        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        self.model = model.eval()
        if self.use_cuda:
            self.model.cuda()
        return

    def __call__(self, image_path, true_label, target_label, return_score=False):
        self.forward(image_path, true_label, target_label)
        # adv_image, pred_label = self.forward(image_path, true_label, target_label)
        # adv_image_uint8 = adv_image.astype(np.uint8)

        # if not return_score:
        #     return adv_image_uint8
        # else:
        #     score = compute_score(imread(image_path), adv_image, pred_label, true_label,
        #                           target_label, self.pixel_limit)
        #     return adv_image_uint8, score

    def forward(self, image_path, true_label, target_label):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        origin = image.clone().detach()
        true = torch.LongTensor([true_label - 1])
        target = torch.LongTensor([target_label - 1])
        if self.use_cuda:
            image, origin = image.cuda(), origin.cuda()
            true, target = true.cuda(), target.cuda()

        num_pixels = image.view(1, -1).size(1)
        if self.num_iters is None or self.num_iters < 1:
            num_iters = num_pixels
        else:
            num_iters = min([self.num_iters, num_pixels])

        rand_perms = torch.randperm(num_pixels)
        num_no_improve, best_loss, best_adv_image = 0, None, None
        for i in range(num_iters):
            # Compute probs
            prob, log_probs = self.__get_probs(image, target - 1)
            loss = self.__loss(log_probs, true, target, image, origin)
            
            


        #     # Compute loss
        #     pred = self.model(image)
        #     prob = torch.softmax(pred, dim=1)
        #     log_prob = torch.log(torch.clamp(prob, min=1e-45))

        #     loss = 1 / (true_ce(log_prob, true_label) + 1e-8) + \
        #         target_ce(log_prob, target_label) + l1(image, origin_image)

        #     # Update best loss
        #     if best_loss is None or loss.item() < best_loss:
        #         best_adv_image = image.clone()
        #         best_loss = loss.item()
        #         num_no_improve = 0
        #     else:
        #         num_no_improve += 1
        #     if self.__stop(num_no_improve):
        #         break

        #     # Update image
        #     image = self.__SimBA(loss, image, origin_image)
        #     image_clamp = self.__clamp(image)
        #     image = image_clamp.detach()

        # adv_image, pred_label = self.__post_process(best_adv_image)
        # return adv_image, pred_label
        return

    def __get_probs(self, image, label):
        preds = self.model(image)
        probs = torch.softmax(preds, dim=1)
        log_probs = torch.log(torch.clamp(probs, min=1e-45))
        prob = probs[:, label].data
        return prob, log_probs

    def __loss(self, pred, true, target, image, origin):
        return 1 / torch.clamp(nn.NLLLoss()(pred, true), min=1e-8) \
            + nn.NLLLoss()(pred, target) + nn.L1Loss()(image, origin)

    def __stop(self, num_no_improve):
        return (self.early_stopping is not None) and \
            (num_no_improve == self.early_stopping)

    def __SimBA(self, loss, image, origin_image):
        grad = torch.autograd.grad(loss, image, retain_graph=False)[0]
        perturbation = image - self.alpha * grad.sign() - origin_image
        perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
        image = origin_image - perturbation
        return image

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
        return adv_image, pred_label


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
    # adv_images_dir = '../data/white_adv_images'
    # if not os.path.isdir(adv_images_dir):
    #     os.makedirs(adv_images_dir)

    attack = AttackWhiteBox(
        model=inception_v3(pretrained=True),
        # model=inceptionv4(pretrained='imagenet'),
        # model=inceptionresnetv2(pretrained='imagenet'),
        input_size=299, pixel_limit=32,
        epsilon=16, alpha=5, num_iters=1,
        early_stopping=None, num_threads=2
    )

    scores = []
    for i, sample in tqdm(data.iterrows(), total=len(data), ncols=80):
        true_label = sample['TrueLabel']
        target_label = sample['TargetClass']

        image_path = os.path.join(images_dir, sample['ImageId'])
        attack(image_path, true_label, target_label, return_score=True)
        # adv_image, score = attack(image_path, true_label, target_label, return_score=True)
        # scores.append(score)

        # adv_image_path = os.path.join(adv_images_dir, sample['ImageId'])
        # imwrite(adv_image_path, adv_image)

    # mean_score = np.mean(scores)
    # print('Attack Score: {:.6f}'.format(mean_score))
