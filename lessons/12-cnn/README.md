# Lesson 12 · Convolutional neural networks

- A convolution by hand: a 3 × 3 vertical-edge kernel slides over a 5 × 5 image (`F.conv2d`), one position checked by hand.
- `nn.Conv2d(1, 32, kernel_size=3, padding=1)` and `nn.MaxPool2d(2)`: (64, 1, 28, 28) → (64, 32, 28, 28) → (64, 32, 14, 14).
- A two-block CNN for FashionMNIST with 220,234 parameters vs lesson 10's MLP with 203,530: test accuracy 91.3% vs 88.5%,
  Shirt 72.6% vs 68.6% (10 epochs, Adam 1e-3, best validation weights).
- Gym: write `my_conv2d` with two loops and the output size formula `(size + 2·padding − kernel) // stride + 1`.

```bash
uv run python lessons/12-cnn/lesson.py
uv run python lessons/12-cnn/gym.py
uv run python lessons/12-cnn/solution.py
```

Videos: [EN](https://youtu.be/sK2Fuub3Slc) · [UA](https://youtu.be/djcJKZepn1c)
