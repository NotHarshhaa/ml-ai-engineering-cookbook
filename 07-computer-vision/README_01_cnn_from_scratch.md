# Code Walkthrough — `01_cnn_from_scratch.ipynb`

> Module: **07-computer-vision** | Notebook: **01_cnn_from_scratch.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 31
- **Markdown cells:** 25
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

> # 01 — CNNs from Scratch
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
> ## 1. Concept Overview — Convolutions from scratch

> A convolution slides a small kernel across an image and computes a weighted
> sum at each location. It's the building block of every CNN. Before we let
> PyTorch do it for us, we'll build a 2-D convolution by hand.
> 
> **Beginner track:** focus on the shape `(batch, channels, height, width)`
> and what a kernel *sees* through its receptive field.
> **Advanced track:** also derive the backward pass for convolution and
> implement a `Conv2d` class with both directions.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> 2-D cross-correlation (what deep learning calls "convolution"):
> 
> $$(I \star K)_{i,j} = \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} K_{u,v}\, I_{i+u,\, j+v}$$
> 
> Padding $p$ and stride $s$ change the output size:
> $$H_\text{out} = \left\lfloor \frac{H_\text{in} + 2p - k}{s} \right\rfloor + 1$$

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements a 2-D cross-correlation in pure NumPy, then
> wraps it as a `Conv2dScratch` class with kernel weights you can read and
> set by hand. We apply three hand-crafted kernels (edge, blur, sharpen) to
> an image and inspect the result.
> 
> **Why it matters:** Convolutions look like magic until you've written one.
> Then you see they're just nested for-loops with a weighted sum — a small
> linear model evaluated at every spatial location.
> 
> **What to watch for:** `Conv2dScratch.forward` uses `np.tensordot` for
> speed; the equivalent 4-deep for-loop would be ~100× slower.


### Code cell #1
```python
import numpy as np

def conv2dscratch(X, K):
    """2-D cross-correlation. X: (H, W), K: (k, k) -> (H-k+1, W-k+1)."""
    kh, kw = K.shape
    out_h = X.shape[0] - kh + 1
    out_w = X.shape[1] - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(X[i:i+kh, j:j+kw] * K)   # weighted sum
    return out

# Three classic 3x3 kernels — what early CV used before deep learning.
edge = np.array([[-1, -1, -1],
                 [-1,  8, -1],
                 [-1, -1, -1]])

blur = np.array([[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]) / 9.0

sharpen = np.array([[ 0, -1,  0],
                    [-1,  5, -1],
                    [ 0, -1,  0]])

# Build a small synthetic image: a black square on a white background.
rng = np.random.default_rng(0)
img = np.ones((10, 10))
img[3:7, 3:7] = 0.0

for name, K in [("edge", edge), ("blur", blur), ("sharpen", sharpen)]:
    out = conv2dscratch(img, K)
    print(f"{name:8s} output shape={out.shape}  mean={out.mean():.3f}")
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `conv2dscratch` — 2-D cross-correlation. X: (H, W), K: (k, k) -> (H-k+1, W-k+1).
- **Key imports:** `import numpy as np`

> **Takeaway:** A convolution is a small linear model evaluated at every
> spatial location. CNNs *learn* the kernels instead of hand-designing them.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses `torch.nn.Conv2d` to apply a 3×3 kernel to a
> synthetic image. We manually set the kernel weights to a Sobel edge filter
> to show that `Conv2d` is doing exactly what our scratch version did.
> 
> **Why it matters:** Once you see that `nn.Conv2d` is just a fast, batched,
> auto-differentiable version of the scratch code, every CNN paper becomes
> much more readable.
> 
> **What to watch for:** PyTorch expects `(batch, channels, H, W)` — a 4-D
> tensor. A single image still needs the leading batch and channel axes.


### Code cell #2
```python
import torch
import torch.nn as nn

# Build a 1-channel "image" with shape (1, 1, 10, 10).
img_t = torch.tensor(img, dtype=torch.float32)[None, None, :, :]

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
                 bias=False, padding=1)  # padding=1 keeps size

# Set the kernel to a Sobel vertical-edge detector.
sobel_y = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=torch.float32)[None, None, :, :]
with torch.no_grad():
    conv.weight.copy_(sobel_y)

out = conv(img_t)
print("input shape :", img_t.shape)
print("output shape:", out.shape)
print("output values (first row):", out[0, 0, 0].detach().numpy().round(2))
```

**What this code does:**

- **Headline:** `import torch`
- **Key imports:** `import torch`, `import torch.nn as nn`
- **Notable calls:** `nn.Conv2d`

> **What this does:** Builds a tiny CNN for MNIST digit classification in
> PyTorch — `Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → Linear`.
> This is the canonical "hello world" of deep learning for vision.
> 
> **Why it matters:** Once you can read this architecture, you can read
> ResNet, VGG, MobileNet — they're all variations on the same conv-block
> pattern.
> 
> **What to watch for:** the `Flatten` is what bridges the conv world
> (4-D tensors) to the classifier world (2-D tensors). Getting the flatten
> size right is the #1 beginner mistake with CNNs.


### Code cell #3
```python
class TinyCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                # 28x28 -> 14x14
            nn.Conv2d(16, 32, 3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (B, 32, 7, 7) -> (B, 32*7*7)
            nn.Linear(32 * 7 * 7, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = TinyCNN()
x = torch.randn(8, 1, 28, 28)  # batch of 8 MNIST-shaped images
print("output shape:", model(x).shape)  # (8, 10)
print("parameter count:", sum(p.numel() for p in model.parameters()))
```

**What this code does:**

- **Headline:** `class TinyCNN(nn.Module):`
- **Defines:** class `TinyCNN`
- **Notable calls:** `nn.Linear`, `nn.Conv2d`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads a small subset of MNIST through `torchvision`,
> visualises 8 sample images, and prints per-class counts.
> 
> **Why it matters:** MNIST is the canonical CV benchmark — small enough to
> train on CPU in seconds, simple enough to verify your pipeline end-to-end.
> If your code works on MNIST, you can scale to ImageNet.
> 
> **What to watch for:** `transform.ToTensor()` scales pixel values from
> 0-255 to 0.0-1.0. Forgetting this leads to loss `NaN` immediately.


### Code cell #4
```python
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),                    # 0-255 -> 0.0-1.0
])
train = datasets.MNIST("./data", train=True, download=True, transform=transform)

# Use only 2000 images for speed in this demo.
subset = Subset(train, list(range(2000)))
print("subset size:", len(subset))

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    img, label = subset[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(label, fontsize=9)
    ax.axis("off")
plt.suptitle("First 8 MNIST samples")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from torchvision import datasets, transforms`
- **Key imports:** `from torchvision import datasets`, `from torch.utils.data import Subset`, `import matplotlib.pyplot as plt`
- **Notable calls:** `transforms`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Visualises the 16 feature maps produced by the first
> conv layer of our `TinyCNN` on a single MNIST image. Each map shows what
> one learned kernel detected.
> 
> **Why it matters:** visualising intermediate features is the standard way
> to *understand* what a CNN learned. Early layers detect edges; later
> layers detect parts; the classifier combines parts into class votes.
> 
> **What to watch for:** before visualising, call `model.eval()` and use
> `torch.no_grad()`. Otherwise the forward pass runs in train mode and may
> produce different outputs (due to dropout/batchnorm).


### Code cell #5
```python
# Grab one MNIST image.
img, label = subset[0]
x = img.unsqueeze(0)  # add batch axis -> (1, 1, 28, 28)

# Hook the first conv layer to capture its output.
activations = {}
def hook(module, inp, out):
    activations["conv1"] = out.detach()
model.features[0].register_forward_hook(hook)

model.eval()
with torch.no_grad():
    logits = model(x)

feature_maps = activations["conv1"][0]   # (16, 28, 28)
fig, axes = plt.subplots(2, 8, figsize=(12, 4))
for i in range(16):
    ax = axes[i // 8, i % 8]
    ax.imshow(feature_maps[i].numpy(), cmap="viridis")
    ax.axis("off")
plt.suptitle(f"16 feature maps from Conv1 (true label = {label})")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `img, label = subset[0]`
- **Defines:** def `hook`
- **Notable calls:** `model.eval`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> For image classification, the standard metrics are:
> 
> - **Top-1 accuracy** — the predicted class matches the true class.
> - **Top-5 accuracy** — the true class is in the model's top 5 predictions.
> - **Confusion matrix** — which classes get confused with which.
> - **Per-class precision / recall / F1** — for imbalanced classes.

> **What this does:** Trains `TinyCNN` for 2 epochs on the 2000-image MNIST
> subset and prints train + test accuracy. We deliberately keep this small
> so it runs in seconds on CPU.
> 
> **Why it matters:** this is the smallest end-to-end CV training loop.
> Once you can read it, you can read any image classification code.
> 
> **What to watch for:** even 2 epochs on 2000 images gets ~90% test accuracy
> — that's how easy MNIST is. Real CV datasets (CIFAR-100, ImageNet) are
> vastly harder.


### Code cell #6
```python
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

train_loader = DataLoader(subset, batch_size=64, shuffle=True)
test_loader = DataLoader(Subset(train, list(range(2000, 3000))),
                         batch_size=64, shuffle=False)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(2):
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
    print(f"epoch {epoch} done")

model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_pred.extend(model(xb).argmax(1).tolist())
        all_true.extend(yb.tolist())
print(f"test accuracy: {accuracy_score(all_true, all_pred):.3f}")
```

**What this code does:**

- **Headline:** `from torch.utils.data import DataLoader`
- **Key imports:** `from torch.utils.data import DataLoader`, `from sklearn.metrics import accuracy_score`
- **Notable calls:** `DataLoader`, `CrossEntropyLoss`, `Adam`, `backward`, `model.train`, `model.eval`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Replace `MaxPool2d` with strided convolutions — does accuracy change?
> 2. Add a `BatchNorm2d` after every conv — does training converge faster?
> 3. Train for 10 epochs and plot the train/test accuracy curves.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement `MaxPool2d` from scratch and verify it matches `nn.MaxPool2d`.
> 2. Implement a `Conv2d` from scratch with both forward and backward passes.
> 3. Build a deeper CNN (5 conv layers) for MNIST and reach 98% test accuracy.
> 4. Add data augmentation (random crop, rotation) and measure the improvement.
> 5. Implement Grad-CAM to visualise which image regions drove the prediction.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why are CNNs better than MLPs for images?
> 2. What is the receptive field, and how does it grow with depth?
> 3. What is the difference between `Conv2d` and `ConvTranspose2d`?
> 4. Why use padding in convolutions?
> 5. What is parameter sharing, and why does it matter?
> 6. How does max pooling help with translation invariance?
> 7. Why do we use `Flatten` between conv and linear layers?
> 8. How would you design a CNN for 1024×1024 satellite images?
