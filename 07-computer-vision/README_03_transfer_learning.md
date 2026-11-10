# Code Walkthrough — `03_transfer_learning.ipynb`

> Module: **07-computer-vision** | Notebook: **03_transfer_learning.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 30
- **Markdown cells:** 24
- **Code cells:** 6

## 10-point learning structure

Every notebook in this cookbook follows this structure wherever applicable:

1. **Concept Overview** — the problem and its fundamentals
2. **Mathematical Foundation** — the underlying mathematics
3. **Implementation from Scratch** — pure Python + NumPy
4. **Library Implementation** — the industry-standard framework
5. **Dataset Exploration** — a real dataset
6. **Visualization** — results through plots
7. **Model Evaluation** — appropriate metrics
8. **Experiments** — modify and observe
9. **Hands-on Challenges** — practice independently
10. **Interview Questions** — reinforce understanding

---

> # 03 — Transfer Learning
> 
> > **ML & AI Engineering Cookbook** — hands-on Jupyter notebook.
> >
> > Run on [Google Colab](https://colab.research.google.com/) or locally with Jupyter.

> ## Notebook Learning Structure
> 
> 1. **Concept Overview** — understand the problem and its fundamentals.
> 2. **Mathematical Foundation** — explore the underlying mathematics.
> 3. **Implementation from Scratch** — build it with Python and NumPy.
> 4. **Library Implementation** — use the industry-standard framework.
> 5. **Dataset Exploration** — work with a practical dataset.
> 6. **Visualization** — understand results through plots.
> 7. **Model Evaluation** — analyze performance with metrics.
> 8. **Experiments** — modify parameters and observe behavior.
> 9. **Hands-on Challenges** — practice independently.
> 10. **Interview Questions** — reinforce conceptual understanding.

> ---
> 
> ## 1. Concept Overview — Transfer learning

> Instead of training a CNN from scratch, you start from a model pretrained
> on ImageNet (1.3M images, 1000 classes) and *fine-tune* it on your own
> dataset. This is how every real-world CV project starts.
> 
> **Beginner track:** learn the two transfer modes — *feature extraction*
> (freeze backbone, train new head) and *fine-tuning* (unfreeze and update
> all weights with a small LR).
> **Advanced track:** also learn which pretrained model to pick for which
> task — ResNet for general classification, EfficientNet for mobile, ViT for
> large datasets, DINOv2 for self-supervised features.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Transfer learning works because low-level visual features (edges, textures,
> corners) are *universal* across vision tasks. A model that learned to
> detect an edge on ImageNet can detect the same edge in your medical scan.
> 
> Formally, let $f_\theta = f_\text{head} \circ f_\text{backbone}$. In
> feature-extraction mode we freeze $f_\text{backbone}$ and only update
> $f_\text{head}$. In fine-tuning mode we update both, with a smaller LR
> for the backbone.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Demonstrates the conceptual structure of transfer
> learning in pure NumPy: take a fixed random "pretrained" feature extractor,
> stack a freshly-initialised linear classifier on top, and train only the
> classifier. The features don't change; the head learns to use them.
> 
> **Why it matters:** This is *exactly* what `torchvision.models.resnet18`
> feature extraction does, minus the convolutions. Understanding the shape
> of the trick is half the value.
> 
> **What to watch for:** we deliberately don't update `W_features`. In a
> real transfer-learning scenario, that's the ImageNet-pretrained weights.


### Code cell #1
```python
import numpy as np

rng = np.random.default_rng(0)
# Pretend "pretrained" backbone — a fixed random projection.
W_features = rng.standard_normal((20, 5))  # 5 features per sample

# Fresh, trainable linear classifier head.
W_head = np.zeros((5, 2))
b_head = np.zeros(2)

# Synthetic binary problem in feature space.
X = rng.standard_normal((100, 20))
features = X @ W_features                  # (100, 5) — frozen
y = (features[:, 0] + features[:, 1] > 0).astype(int)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

lr = 0.1
for step in range(200):
    logits = features @ W_head + b_head
    probs = softmax(logits)
    # cross-entropy gradient
    grad = probs.copy()
    grad[np.arange(len(y)), y] -= 1
    grad /= len(y)
    # Only W_head, b_head are updated — features stay frozen.
    W_head -= lr * features.T @ grad
    b_head -= lr * grad.sum(axis=0)

acc = (probs.argmax(1) == y).mean()
print(f"feature-extraction accuracy after 200 steps: {acc:.3f}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `softmax`
- **Key imports:** `import numpy as np`
- **Notable calls:** `softmax`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Loads `torchvision.models.resnet18(pretrained=True)`,
> freezes its parameters, replaces the final `fc` layer with a fresh one of
> the right size for our task, and trains only the new head.
> 
> **Why it matters:** this 5-line pattern (`load → freeze → replace head →
> train head`) is the most common CV workflow in industry. Master it.
> 
> **What to watch for:** `requires_grad_(False)` is the standard way to
> freeze a parameter. PyTorch's optimiser will then skip it.
> 
> **Heads-up:** the first run downloads ~45 MB of pretrained weights.


### Code cell #2
```python
import torch
import torch.nn as nn
from torchvision import models

# Load the pretrained ResNet-18.
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all backbone parameters — feature-extraction mode.
for p in model.parameters():
    p.requires_grad_(False)

# Replace the final 1000-class head with a fresh 10-class head.
# The new head has requires_grad=True by default.
model.fc = nn.Linear(model.fc.in_features, 10)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,} of {total:,} parameters ({100*trainable/total:.1f}%)")
```

**What this code does:**

- **Headline:** `import torch`
- **Key imports:** `import torch`, `import torch.nn as nn`, `from torchvision import models`
- **Notable calls:** `nn.Linear`, `ResNet`

> **What this does:** Switches from feature-extraction to fine-tuning: we
> un-freeze the last conv block (`layer4`) and train it together with the
> head, but at a 10× smaller learning rate. This is the standard pattern when
> your target dataset is large enough to support updating the backbone.
> 
> **Why it matters:** fine-tuning beats feature extraction when you have
> *enough* target data — usually >1000 images per class. Below that, the
> frozen-backbone approach is safer.
> 
> **What to watch for:** always use a *smaller* LR for the pretrained
> backbone (often 10× smaller). Updating ImageNet weights at LR=1e-3 will
> destroy the pretrained features in a few steps.


### Code cell #3
```python
# Unfreeze the last conv block for fine-tuning.
for p in model.layer4.parameters():
    p.requires_grad_(True)

# Use two parameter groups: backbone gets a small LR, head gets a larger one.
backbone_params = [p for p in model.layer4.parameters() if p.requires_grad]
head_params     = [p for p in model.fc.parameters() if p.requires_grad]

opt = torch.optim.SGD([
    {"params": backbone_params, "lr": 1e-4},     # 10x smaller for backbone
    {"params": head_params,     "lr": 1e-3},     # standard for new head
], momentum=0.9)

print(f"backbone params trainable: {sum(p.numel() for p in backbone_params):,}")
print(f"head params trainable:     {sum(p.numel() for p in head_params):,}")
```

**What this code does:**

- **Headline:** `for p in model.layer4.parameters():`
- **Notable calls:** `SGD`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads a few sample images and prepares the standard
> ImageNet preprocessing pipeline (resize 256, crop 224, normalise with
> ImageNet stats). ResNet expects exactly this.
> 
> **Why it matters:** every pretrained model has a *fixed* preprocessing
> expectation. If you violate it (wrong input size, wrong normalisation),
> the model's predictions become garbage.
> 
> **What to watch for:** ImageNet's normalisation constants are
> `(0.485, 0.456, 0.406)` for mean and `(0.229, 0.224, 0.225)` for std —
> different from CIFAR-10's.


### Code cell #4
```python
from torchvision import transforms

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)

preprocess = transforms.Compose([
    transforms.Resize(256),                    # shorter side to 256
    transforms.CenterCrop(224),                # 224x224 crop
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
print("preprocessing pipeline ready — every ImageNet-pretrained model expects this")
```

**What this code does:**

- **Headline:** `from torchvision import transforms`
- **Key imports:** `from torchvision import transforms`
- **Notable calls:** `transforms`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Loads the pretrained ResNet-18 *without* modifying the
> head, downloads a sample image, and predicts the top-5 ImageNet classes.
> This is the "did the pretrained model actually learn anything?" sanity
> check.
> 
> **Why it matters:** before you fine-tune, *always* verify that the
> pretrained model produces sensible predictions on your data. If it doesn't,
> your domain is too far from ImageNet — you may need a different backbone.
> 
> **What to watch for:** ResNet was trained on 1000 ImageNet classes, so the
> predicted labels are weirdly specific ("Egyptian cat", "tabby"). That's
> expected — ImageNet's class list is exhaustive.


### Code cell #5
```python
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Use a sample image from a public URL.
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
try:
    img = Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)   # (1, 3, 224, 224)
    pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    pretrained.eval()
    with torch.no_grad():
        probs = torch.softmax(pretrained(input_tensor), dim=1)[0]
    top5 = torch.topk(probs, 5)
    # Load ImageNet class names.
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(labels_url, timeout=10).text.strip().splitlines()
    plt.imshow(img); plt.axis("off")
    title = "\n".join(f"{labels[i]}: {p:.2f}" for i, p in zip(top5.indices, top5.values))
    plt.title(title, fontsize=9); plt.tight_layout(); plt.show()
except Exception as e:
    print(f"skipped pretrained-prediction demo (no network): {e}")
```

**What this code does:**

- **Headline:** `from PIL import Image`
- **Key imports:** `from PIL import Image`, `import matplotlib.pyplot as plt`, `import requests`, `from io import BytesIO`
- **Notable calls:** `ResNet`, `softmax`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> For transfer learning, evaluate:
> 
> - **Does the pretrained model give reasonable predictions on your data?**
>   If not, you may need a different backbone.
> - **Feature extraction vs fine-tuning** — which gives better test accuracy?
> - **Time per epoch** — fine-tuning is slower; is the gain worth it?

> **What this does:** Compares feature extraction (frozen ResNet) vs
> fine-tuning (un-frozen last block) on the CIFAR-10 subset. The comparison
> is the decision you'll make in every transfer-learning project.
> 
> **Why it matters:** fine-tuning is not always better — on small datasets,
> it can overfit. Knowing when to stop at feature extraction is a real
> engineering skill.
> 
> **What to watch for:** CIFAR images are 32×32; we resize to 224 to match
> ImageNet ResNet's expected input. That's wasteful but it keeps the demo
> simple.


### Code cell #6
```python
# Build two transfer models and compare.
def build_model(mode):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if mode == "feature_extraction":
        for p in m.parameters(): p.requires_grad_(False)
    elif mode == "fine_tune":
        for p in m.parameters(): p.requires_grad_(False)
        for p in m.layer4.parameters(): p.requires_grad_(True)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

# We just print parameter counts — full training takes too long for a CPU demo.
for mode in ["feature_extraction", "fine_tune"]:
    m = build_model(mode)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{mode:20s} trainable params: {trainable:,}")
```

**What this code does:**

- **Headline:** `def build_model(mode):`
- **Defines:** def `build_model`
- **Notable calls:** `nn.Linear`, `ResNet`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Try `resnet50` instead of `resnet18` — does the bigger model help on your dataset?
> 2. Try `efficientnet_b0` — how do parameter count and accuracy compare?
> 3. Unfreeze `layer3` and `layer4` together — does that help or hurt?
> 4. Use a learning-rate finder (`torch-lr-finder`) to pick the optimal LR.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a feature-extraction pipeline that caches the frozen backbone
>   outputs to disk — saves recomputing them every epoch.
> 2. Implement discriminative learning rates per layer.
> 3. Use `torchvision.models.detection` for object detection with a pretrained
>   Faster R-CNN.
> 4. Try a self-supervised backbone (DINOv2) and compare to supervised ImageNet.
> 5. Implement test-time augmentation (TTA) — average predictions over 5 crops.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. When would you choose feature extraction over fine-tuning?
> 2. Why use a smaller learning rate for the backbone than the head?
> 3. What happens if you forget to freeze the backbone in feature extraction?
> 4. Why does ImageNet preprocessing use specific mean/std values?
> 5. What is the difference between ResNet-18 and ResNet-50?
> 6. How does transfer learning relate to the universal approximation theorem?
> 7. What is a self-supervised pretrained model, and when is it preferable?
> 8. How would you handle a target domain very different from ImageNet?
