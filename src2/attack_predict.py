import torch
import numpy as np

from PIL import Image
from torchvision import transforms


class AttackPredict(object):

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, model, input_size=299, class_label=None,
                 use_cuda=True, num_threads=1):
        torch.set_num_threads(num_threads)
        self.use_cuda = torch.cuda.is_available() and use_cuda

        self.preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        self.model = model.eval()
        if self.use_cuda:
            self.model.cuda()

        self.class_label = class_label
        return

    def run(self, image_path):
        image = self.preprocess(Image.open(image_path))
        image = image.unsqueeze(0)
        if self.use_cuda:
            image = image.cuda()

        pred = self.model(image)
        pred = torch.softmax(pred, dim=1)
        pred = pred.data.cpu().detach().numpy().flatten()
        pred_label = np.argmax(pred)
        pred_prob = pred[pred_label]

        if self.class_label is not None:
            pred_class = self.class_label[pred_label]
            return pred_label, pred_prob, pred_class

        return pred_label, pred_prob
