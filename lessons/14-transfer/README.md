# Lesson 14 · Transfer learning

- CIFAR-10 with only 5,000 training pictures (and 1,000 for validation), full 10,000-picture test set.
- `models.resnet18(weights=ResNet18_Weights.DEFAULT)`: 11,689,512 parameters, ImageNet top-1 69.758%; `weights.transforms()`
  (resize 256, crop 224, ImageNet mean/std); a new head `model.fc = nn.Linear(512, 10)`.
- Test accuracy: small CNN from scratch 67.0% (15 epochs), ResNet-18 with random weights 59.6%, pretrained ResNet-18 frozen
  (5,130 trainable values) 74.8%, fine-tuned 86.7% (5 epochs each, Adam + OneCycleLR).
- Gym: freeze everything except the head, and two learning rates with optimizer parameter groups (no download needed).

```bash
uv run python lessons/14-transfer/lesson.py
uv run python lessons/14-transfer/gym.py
uv run python lessons/14-transfer/solution.py
```
