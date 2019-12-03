import os
import numpy as np

from numpy.linalg import norm


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def compute_score(src_image, adv_image, pred_label, true_label,
                  target_label, pixel_limit=32, norm_ord=np.inf):
    def prosess(image):
        return image.astype(float).flatten()

    src_image = prosess(src_image)
    adv_image = prosess(adv_image)

    score = 0
    norm_score = norm(adv_image - src_image, ord=norm_ord)
    if pred_label == true_label:
        score = 0
    elif pred_label != true_label and pred_label != target_label:
        score = 2 * (2 - norm_score / pixel_limit)
    elif pred_label == target_label:
        score = 5 * (2 - norm_score / pixel_limit)
    else:
        pass
    return score
