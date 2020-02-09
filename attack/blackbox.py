import copy
import torch
import numpy as np

from PIL import Image
from torchvision import transforms


class BlackBoxAttack(object):

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, model, input_size=224, epsilon=16, num_iters=10000,
                 early_stopping=False, use_cuda=False, random_state=None):
        '''__INIT__
            reference:
            Guo C, Gardner J R, You Y, et al.
            Simple black-box adversarial attacks[J].
            arXiv preprint arXiv:1905.07121, 2019.

            model: model instance or list of model instances
            input_size: int, size of input tentor to model
            epsilon: int, limit on the perturbation size
            num_iters: int, number of iterations
            early_stopping: bool, if True, stop at once if
                            adversarial image has been found
            use_cuda: bool, True or False, whether to use GPU
            random_state: int or None, for reproducing
        '''

        self.num_iters = num_iters
        self.epsilon = epsilon
        # self.epsilon = epsilon / 255
        self.early_stopping = early_stopping
        self.use_cuda = torch.cuda.is_available() and use_cuda
        self.nbits = int(np.ceil(np.log10(num_iters)) + 1)

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

        if random_state is not None:
            np.random.seed(seed=random_state)
        return

    def __call__(self, image_path, label, target=False):
        '''__CALL__
            image_path: string, path of uint8 input image
            label: int, the true label of input image if target is False,
                   the target label to learn if target is True
            target: bool, if True, perform target adversarial attack;
                    if False, perform non-target adversarial attack
        '''

        self.target = target
        src_image = np.array(Image.open(image_path))
        adv_image = self.forward(src_image, label)
        return adv_image.astype(np.uint8)

    def forward(self, src_image, label):
        image = src_image.copy().astype(float)
        n_dims = len(image.flatten())
        perm = np.random.permutation(n_dims)
        last_prob, _ = self.__predict(image, label)
        is_better = np.greater if self.target else np.less

        num_iters = min([self.num_iters, len(perm)])
        for i in range(num_iters):
            diff = np.zeros((n_dims))
            diff[perm[i]] = self.epsilon
            diff = diff.reshape(image.shape)

            left_image = np.clip(image - diff, 0.0, 255.0)
            left_prob, is_stop = self.__predict(left_image, label)
            if is_stop or is_better(left_prob, last_prob):
                image = left_image.copy()
                last_prob = left_prob
                if is_stop:
                    break
            else:
                right_image = np.clip(image + diff, 0.0, 255.0)
                right_prob, is_stop = self.__predict(right_image, label)
                if is_stop or is_better(right_prob, last_prob):
                    image = right_image.copy()
                    last_prob = right_prob
                    if is_stop:
                        break

            iter_msg = '[Running]-[Step:{}/{}]-[Prob:{:.6f}]'
            print(iter_msg.format(i + 1, num_iters, last_prob), end='\r')

        iter_msg = '\n[Stopped]-[Step:{}/{}]-[Prob:{:.6f}]'
        print(iter_msg.format(i + 1, num_iters, last_prob))

        return image

    def __predict(self, image, label):
        def get_prob(model, image_norm):
            pred = model(image_norm)
            probs = torch.softmax(pred, dim=1)
            probs = probs.data.cpu().detach().numpy().flatten()
            pred = np.argmax(probs)
            return probs[label], pred

        image_norm = self.__norm(image)
        if self.use_cuda:
            image_norm = image_norm.cuda()
        prob_preds = [get_prob(model, image_norm) for model in self.model]
        probs = [item[0] for item in prob_preds]
        prob = min(probs) if self.target else max(probs)
        preds = [item[1] for item in prob_preds]

        is_stop = False
        if self.early_stopping:
            if self.target and preds.count(label) == len(preds):
                is_stop = True
            elif (not self.target) and preds.count(label) == 0:
                is_stop = True

        return prob, is_stop

    def __norm(self, image):
        image_cp = Image.fromarray(image.astype(np.uint8))
        image_norm = self.preprocess(image_cp)
        image_norm = image_norm.unsqueeze(0)
        return image_norm
