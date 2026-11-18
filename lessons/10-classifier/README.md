# Lesson 10 · First real classifier: an MLP on FashionMNIST

- FashionMNIST: 60,000 training + 10,000 test pictures, 28 × 28 grey, 10 classes of clothes (torchvision downloads it).
- `transforms.ToTensor()`: pixels 0..255 → floats 0..1, shape (1, 28, 28).
- The model: `Flatten` → `Linear(784, 256)` → `ReLU` → `Linear(256, 10)` = 203,530 parameters; it outputs logits.
- `CrossEntropyLoss` takes logits and class ids (an untrained model starts near ln 10 ≈ 2.30).
- Accuracy = `argmax` of the logits vs the label; `softmax` only when you want probabilities.
- Train with the loop from lesson 08, keep the best validation weights, report the test set once.
- The confusion matrix shows *which* classes get mixed up (Shirt ↔ T-shirt, Pullover ↔ Coat).

```bash
uv run python lessons/10-classifier/lesson.py
uv run python lessons/10-classifier/gym.py
uv run python lessons/10-classifier/solution.py
```

Videos: [EN](https://youtu.be/1Zk-4fkJMoA) · [UA](https://youtu.be/T87ZeAOEKKA)
