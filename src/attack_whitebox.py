import torch
import numpy as np
import torch.nn as nn

from tqdm import *
from PIL import Image
from torchvision import transforms


class AttackWhiteBox(object):

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    # RANGE: [(0 - mean) / std, (1 - mean) / std]
    RANGE = np.array([[-2.1179, 2.2489], [-2.0357, 2.4285], [-1.8044, 2.6400]])

    def __init__(self, model, input_size=224, epsilon=16, alpha=5,
                 num_iters=50, early_stopping=None, num_threads=1, use_cuda=True):
        '''__INIT__

            reference:
            Kurakin A, Goodfellow I, Bengio S.
            Adversarial examples in the physical world[J].
            arXiv preprint arXiv:1607.02533, 2016.

            model: model instance or list of model instances
            input_size: int, size of input tentor to model
            epsilon: int, limit on the perturbation size
            alpha: int, step size for gradient-based attack
            num_iters: int, number of iterations
            early_stopping: int ot None, attack will not stop unless loss stops improving
            num_threads: int, number of threads to use
            use_cuda: bool, True or False, whether to use GPU

        '''

        torch.set_num_threads(num_threads)

        self.alpha = alpha / 255
        self.num_iters = num_iters
        self.epsilon = epsilon / 255
        self.early_stopping = early_stopping
        self.use_cuda = torch.cuda.is_available() and use_cuda

        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        if not isinstance(model, list):
            model = [model]
        for m in model:
            m.eval()
            if self.use_cuda:
                m.cuda()
        self.model = model

        self.l1_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        return

    def __call__(self, image_path, label, target=False):
        '''__CALL__

            image_path: string, path of input image
            label: int, the true label of input image if target is False,
                   the target label to learn if target is True
            target: bool, if True, perform target adversarial attack;
                    if False, perform non-target adversarial attack

        '''

        self.target = target
        src_image = Image.open(image_path)
        adv_image, pred_label = self.forward(src_image, label)
        return adv_image, pred_label

    def forward(self, src_image, label):
        image = self.preprocess(src_image).unsqueeze(0)
        origin = image.clone().detach()
        label = torch.LongTensor([label])
        if self.use_cuda:
            image, origin, label = image.cuda(), origin.cuda(), label.cuda()

        num_no_improve, best_loss, best_adv_image = 0, None, None
        for i in tqdm(range(self.num_iters), ncols=75):
            image.requires_grad = True
            pred = [m(image) for m in self.model]
            loss = self.__loss(pred, label, image, origin)

            if best_loss is None or loss.item() < best_loss:
                best_adv_image = image.clone()
                best_loss = loss.item()
                num_no_improve = 0
            else:
                num_no_improve += 1
            if self.__stop(num_no_improve):
                print('\nEarly stopped.')
                break

            image = self.__PGD(loss, image, origin)
            image_clamp = self.__clamp(image)
            image = image_clamp.detach()

        adv_image, pred_label = self.__post_process(best_adv_image)
        return adv_image, pred_label

    def __loss(self, pred, label, image, origin):
        def compute_ce_loss(p, l):
            if self.target:
                ce_loss = self.ce_loss(p, l)
            else:
                ce_loss = 1 / torch.clamp(self.ce_loss(p, l), min=1e-8)
            return ce_loss

        ce_loss = 0
        for p in pred:
            ce_loss += compute_ce_loss(p, label)
        ce_loss /= len(pred)
        return ce_loss + self.l1_loss(image, origin)

    def __stop(self, num_no_improve):
        return (self.early_stopping is not None) and \
            (num_no_improve == self.early_stopping)

    def __PGD(self, loss, image, origin):
        grad = torch.autograd.grad(loss, image, retain_graph=False)[0]
        perturbation = image - self.alpha * grad.sign() - origin
        perturbation = torch.clamp(perturbation, min=-self.epsilon, max=self.epsilon)
        image = origin - perturbation
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
        def pred_adv(model):
            pred = model(best_adv_image)
            pred = torch.softmax(pred, dim=1)
            pred = pred.data.cpu().detach().numpy().flatten()
            pred_label = np.argmax(pred)
            return pred_label

        pred_label = [pred_adv(m) for m in self.model]
        adv_image = best_adv_image.squeeze(0).data.cpu().detach().numpy()
        adv_image = np.transpose(adv_image, (1, 2, 0))
        adv_image = adv_image * self.STD + self.MEAN
        adv_image = np.round(adv_image * 255.0).astype(np.uint8)
        return adv_image, pred_label
