import json

from imageio import imwrite
from pretrainedmodels import *
from torchvision.models import *
from attack_predict import AttackPredict
from attack_whitebox import AttackWhiteBox


# ======================================================================================

# Preparation
print('-' * 75)
result_str = 'Prediction of source image:\n[Label:{}]-[Class:{}]-[Confidence:{:.6f}]'

# Load label-class pairs of ImageNet
class_label_dict = json.load(open('../data/imagenet_class_index.json'))
class_label = [class_label_dict[str(k)][1] for k in range(len(class_label_dict))]

# Source image
src_image_path = '../data/central_perk_224.png'  # label:762
print('Source image: [{}]'.format(src_image_path))

# Model to be attacked
model, input_size = resnet18(pretrained=True), 224
# model, input_size = resnet34(pretrained=True), 224
# model, input_size = inception_v3(pretrained=True), 299
print('Model to be attacked: [pretrained ResNet18 on ImageNet]')
print('-' * 75)

# --------------------------------------------------------------------------------------

# Prediction of source image
predictor = AttackPredict(model=model, input_size=input_size, class_label=class_label)
src_label, src_prob, src_class = predictor.run(src_image_path)
print(result_str.format(src_label, src_class, src_prob))
print('-' * 75)

# ======================================================================================

# White-Box Adversarial Attack on source image
attack_whitebox = AttackWhiteBox(
    model=model, input_size=input_size, epsilon=16, alpha=5,
    num_iters=100, early_stopping=5
)

# 'model' also could be a list of model instances
# attack_whitebox = AttackWhiteBox(
#     model=[resnet18(pretrained=True), resnet34(pretrained=True)],
#     input_size=input_size, epsilon=16, alpha=5,
#     num_iters=100, early_stopping=5
# )

# --------------------------------------------------------------------------------------

# Non-Targeted Attack
print('White-Box Non-Targeted Adversarial Attack')
wb_nt_image, _ = attack_whitebox(image_path=src_image_path, label=762, target=False)
wb_nt_image_path = '../data/wb_nt_central_perk.png'
imwrite(wb_nt_image_path, wb_nt_image)

wb_nt_label, wb_nt_prob, wb_nt_class = predictor.run(wb_nt_image_path)
print(result_str.format(wb_nt_label, wb_nt_class, wb_nt_prob), '\n')

# --------------------------------------------------------------------------------------

# Targeted Attack
print('White-Box Targeted Adversarial Attack')
wb_t_image, _ = attack_whitebox(image_path=src_image_path, label=388, target=True)
wb_t_image_path = '../data/wb_t_central_perk.png'
imwrite(wb_t_image_path, wb_t_image)

wb_t_label, wb_t_prob, wb_t_class = predictor.run(wb_t_image_path)
print(result_str.format(wb_t_label, wb_t_class, wb_t_prob))
print('-' * 75)

# ======================================================================================

# Black-Box Adversarial Attack on source image
