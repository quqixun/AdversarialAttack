# Adversarial Attack

Attack models that are pretrained on ImageNet.

- Attack single model or multiple models.
- Apply white-box attack or black-box attack.
- Apply non-targeted attack or targeted attack.

## Method

- White-box attack: PGD[1]
- Black-box attack: [2]

## Examples

```python
from torchvision.models import *
from attack.whitebox import WhiteBoxAttack

# Source image
src_image_path = './data/central_perk_224.png'  # label:762

# Model to be attacked
model, input_size = resnet18(pretrained=True), 224

# ----------------------------------------------------------------------------------
# White-box attack
whitebox_attack = WhiteBoxAttack(
    model=model, input_size=input_size, epsilon=16, alpha=5,
    num_iters=100, early_stopping=5
)

# Non-targeted attack
wb_nt_image, _ = whitebox_attack(image_path=src_image_path, label=762, target=False)
# Targeted attack (label 388 for giant panda)
wb_t_image, _ = whitebox_attack(image_path=src_image_path, label=388, target=True)
# ----------------------------------------------------------------------------------

```

|Image|Source|Model|Attack Type|Target Type|Target Label|Label|Class|Confidence|
|:---:|:----:|:----------:|:---------:|:---------:|:----------:|:---:|:---:|:--------:|
|<img src="./data/central_perk_224.png" alt="drawing" width="150"/>|Yes|ResNet18|-|-|-|762|restaurant|0.957634
|<img src="./data/wb_nt_central_perk.png" alt="drawing" width="150"/>|No|ResNet18|White-box|Non-targeted|-|424|barbershop|0.984476|
|<img src="./data/wb_t_central_perk.png" alt="drawing" width="150"/>|No|ResNet18|White-box|Targeted|388|388|giant_panda|0.999937|


## References

[1] Kurakin A, Goodfellow I, Bengio S. Adversarial examples in the physical world[J]. arXiv preprint arXiv:1607.02533, 2016.  
[2]

## Requirements