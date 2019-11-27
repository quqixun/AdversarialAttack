import os
import numpy as np
import pandas as pd

from imageio import imread
from numpy.linalg import norm


if __name__ == '__main__':
    dev_pred = pd.read_csv('../data/dev_pred.csv')
    images_dir = '../data/images'
    adv_images_dir = '../data/adv_images'

    score = 0
    for i, sample in dev_pred.iterrows():
        image_file = sample['ImageId']
        true_label = sample['TrueLabel']
        target_label = sample['TargetClass']
        pred_label = sample['PredLabel']

        try:
            image = imread(os.path.join(images_dir, image_file)).astype(float)
            adv_image = imread(os.path.join(adv_images_dir, image_file)).astype(float)
            norm_score = norm(adv_image.flatten() - image.flatten(), ord=np.inf)

            if pred_label == true_label:
                score += 0
            elif pred_label != true_label and pred_label != target_label:
                score += 2 * (2 - norm_score / 32)
            elif pred_label == target_label:
                score += 5 * (2 - norm_score / 32)
            else:
                continue
        except Exception:
            continue

    score /= len(dev_pred)
    print('Attack score: {:.6f}'.format(score))
