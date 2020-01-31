import json

from imageio import imwrite
from pretrainedmodels import *
from torchvision.models import *
from attack_predict import AttackPredict
from attack_whitebox import AttackWhiteBox


# Load label-class pairs of ImageNet
class_label_dict = json.load(open('../data/imagenet_class_index.json'))
class_label = [class_label_dict[str(k)][1] for k in range(len(class_label_dict))]

# Source image
# src_image_path = '../data/panda.jpg'  # label:388
src_image_path = '../data/central_perk.jpg'  # label:762
print('Source image: [{}]'.format(src_image_path))

# Model to be attacked
model = resnet18(pretrained=True)
print('Model to be attacked: [pretrained ResNet18 on ImageNet]')
print('-' * 75)

# Prediction of source image
predictor = AttackPredict(model=model, input_size=224, class_label=class_label)
src_label, src_prob, src_class = predictor.run(src_image_path)
print('Prediction of source image:\n[Label:{}]-[Class:{}]-[Confidence:{:.6f}]'.format(src_label, src_class, src_prob))
print('-' * 75)

# -------------------------------------------------------------------------------------------------------------------------
# White-Box Adversarial Tttack

# White-Box Non-Target Adversarial Tttack on source image
attack_whitebox = AttackWhiteBox(model=model, input_size=224, epsilon=8, alpha=1, num_iters=50, early_stopping=5)
wb_nt_at_image, _ = attack_whitebox(image_path=src_image_path, label=388, target=True)
# Save adversarial image
wb_nt_at_image_path = '../data/wb_nt_at_central_perk.jpg'
imwrite(wb_nt_at_image_path, wb_nt_at_image)

wb_nt_at_label, wb_nt_at_prob, wb_nt_at_class = predictor.run(wb_nt_at_image_path)
print('White-Box non-target adversarial attack')
print('Prediction of output image:\n[Label:{}]-[Class:{}]-[Confidence:{:.6f}]'.format(wb_nt_at_label, wb_nt_at_class, wb_nt_at_prob))
print()
