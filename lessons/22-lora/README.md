# Lesson 22 · Hugging Face + fine-tuning with LoRA

Needs the optional extra: `uv sync --extra nlp` (transformers, peft). The first run downloads DistilGPT-2 (about 350 MB).

- A pretrained model in two lines: `AutoTokenizer` + `AutoModelForCausalLM` ("distilbert/distilgpt2": 81,912,576 parameters,
  a 50,257-token BPE vocabulary, about 3.3 characters per token on Shakespeare).
- Loss per character on the last 10% of Tiny Shakespeare, comparable with lesson 21's tiny GPT (1.555):
  DistilGPT-2 before fine-tuning 1.462, after 600 LoRA steps (about a minute on MPS) 1.278.
- LoRA (`peft`): r = 8, alpha = 16 on the attention projection `c_attn`: 147,456 trainable parameters of 82,060,032 (0.18%);
  `lora_A` (8, 768), `lora_B` (2304, 8) starting at zero, so training starts from exactly the pretrained model.
- The adapter on disk: 0.60 MB, against 328 MB for the full model in float32.
- Samples: before, internet text about a game; after, Shakespeare-style dialogue with names and verse lines.
- Gym: count trainable parameters, and the LoRA forward pass by hand (equal to the merged weight W + (alpha / r) B A).

```bash
uv sync --extra nlp
uv run python lessons/22-lora/lesson.py
uv run python lessons/22-lora/gym.py
uv run python lessons/22-lora/solution.py
```
