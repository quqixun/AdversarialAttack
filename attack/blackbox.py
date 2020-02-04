import copy
import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import transforms


class BlackBoxAttack(object):

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    # RANGE: [(0 - mean) / std, (1 - mean) / std]
    RANGE = np.array([[-2.1179, 2.2489], [-2.0357, 2.4285], [-1.8044, 2.6400]])

    def __init__(self, model, input_size=224, epsilon=16,
                 num_iters=50, early_stopping=None, use_cuda=False):
        '''__INIT__
            reference:
            Guo C, Gardner J R, You Y, et al.
            Simple black-box adversarial attacks[J].
            arXiv preprint arXiv:1905.07121, 2019.

            model: model instance or list of model instances
            input_size: int, size of input tentor to model
            epsilon: int, limit on the perturbation size
            num_iters: int, number of iterations
            early_stopping: int ot None, attack will not stop unless loss stops improving
            use_cuda: bool, True or False, whether to use GPU
        '''

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
        model = [copy.deepcopy(m) for m in model]
        for m in model:
            m.eval()
            if self.use_cuda:
                m.cuda()
        self.model = model
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
        src_image = np.array(Image.open(image_path)) / 255.0
        print(src_image.shape)
        # adv_image, pred_label = self.forward(src_image, label)
        self.forward(src_image, label)
        # return adv_image, pred_label
        return

    def forward(self, src_image, label):
        image = torch.Tensor(src_image)
        image = image.unsqueeze(0)
        n_dims = image.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        last_prob = self.__predict(image, label)

        for i in range(self.num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = self.epsilon

            left_image = image - diff.view(image.size()).clamp(0, 1)
            left_prob = self.__predict(left_image, label)
            if left_prob < last_prob:
                image = left_image
                last_prob = left_prob
            else:
                right_image = image + diff.view(image.size()).clamp(0, 1)
                right_prob = self.__predict(right_image, label)
                if right_prob < last_prob:
                    image = right_image
                    last_prob = right_prob
            print(last_prob)
        return

    def __predict(self, image, label):
        def get_prob(model, image_norm):
            pred = model(image_norm)
            probs = torch.softmax(pred, dim=1)
            probs = probs.data.cpu().detach().numpy().flatten()
            return probs[label]

        image_norm = self.__norm(image)
        if self.use_cuda:
            image_norm = image_norm.cuda()
        probs = [get_prob(model, image_norm) for model in self.model]
        return np.mean(probs)

    def __norm(self, image):
        image_cp = image.clone()
        image_cp = image_cp.squeeze(0).permute(2, 0, 1)
        image_cp = transforms.ToPILImage()(image_cp)
        image_norm = self.preprocess(image_cp)
        image_norm = image_norm.unsqueeze(0)
        return image_norm

    # def __clamp(self, image):
    #     image_clamp = []
    #     for i in range(image.size()[1]):
    #         cmin, cmax = self.RANGE[i]
    #         image_i = torch.clamp(image[0][i], min=cmin, max=cmax)
    #         image_clamp.append(image_i)
    #     image_clamp = torch.stack(image_clamp, dim=0).unsqueeze(0)
    #     return image_clamp

    # def __post_process(self, best_adv_image):
    #     def pred_adv(model):
    #         pred = model(best_adv_image)
    #         pred = torch.softmax(pred, dim=1)
    #         pred = pred.data.cpu().detach().numpy().flatten()
    #         pred_label = np.argmax(pred)
    #         return pred_label

    #     pred_label = [pred_adv(m) for m in self.model]
    #     adv_image = best_adv_image.squeeze(0).data.cpu().detach().numpy()
    #     adv_image = np.transpose(adv_image, (1, 2, 0))
    #     adv_image = adv_image * self.STD + self.MEAN
    #     adv_image = np.round(adv_image * 255.0).astype(np.uint8)
    #     return adv_image, pred_label
