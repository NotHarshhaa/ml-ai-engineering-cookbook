# Code Walkthrough — `02_image_classification.ipynb`

> Module: **07-computer-vision** | Notebook: **02_image_classification.ipynb**
>
> This file walks you through every code cell in the notebook, in order.
> For each cell you'll see the surrounding markdown context, the code itself,
> and a one-paragraph summary of what the code does.

---

## Notebook overview

- **Total cells:** 28
- **Markdown cells:** 23
- **Code cells:** 5

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

> # 02 — Image Classification with PyTorch
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
> ## 1. Concept Overview — Image classification end-to-end

> Now we go from a toy CNN on MNIST to a complete image-classification
> pipeline: data loading with augmentation, model definition, training loop,
> evaluation, and saving/loading checkpoints. The patterns here scale to
> ImageNet.
> 
> **Beginner track:** understand the `Dataset` + `DataLoader` + `Module` +
> training-loop quartet. That's 80% of CV code.
> **Advanced track:** also learn distributed training, mixed-precision, and
> torch.compile.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Cross-entropy loss for $C$ classes:
> 
> $$L = -\sum_{i=1}^{C} y_i \log \hat{p}_i, \quad \hat{p} = \text{softmax}(z)$$
> 
> Combined softmax + cross-entropy (as in `nn.CrossEntropyLoss`) is
> numerically more stable than applying them separately.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Re-implements cross-entropy loss from scratch in NumPy
> and verifies it matches `sklearn.metrics.log_loss`.
> 
> **Why it matters:** Once you see that cross-entropy is just `-log(p_true)`
> averaged over the batch, every classification loss stops being mysterious.
> 
> **What to watch for:** the `eps=1e-15` clip is critical. Without it, a
> single confident wrong prediction makes the loss `inf`.


### Code cell #1
```python
import numpy as np
from sklearn.metrics import log_loss

def cross_entropy_scratch(logits, y):
    # logits: (B, C), y: (B,) integer labels.
    # 1. softmax with stability
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    # 2. negative log-likelihood of true class, averaged.
    n = len(y)
    return -np.mean(np.log(probs[np.arange(n), y] + 1e-15))

logits = np.array([[2.0, 1.0, 0.1], [0.1, 2.0, 0.5]])
y = np.array([0, 1])
print("scratch:", cross_entropy_scratch(logits, y))
# sklearn expects probabilities, so we softmax first.
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
print("sklearn:", log_loss(y, probs, labels=[0, 1, 2]))
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `cross_entropy_scratch`
- **Key imports:** `import numpy as np`, `from sklearn.metrics import log_loss`
- **Notable calls:** `softmax`, `cross_entropy`

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Builds a complete training pipeline for CIFAR-10:
> download dataset, normalise, augment, define a 3-conv-block CNN, train
> with Adam, evaluate on test set. This is the structure of every CV
> project at scale.
> 
> **Why it matters:** the *pipeline* is what you ship. Models change; the
> data + train + eval pipeline stays the same across projects.
> 
> **What to watch for:** `transforms.Normalize(mean, std)` is per-channel.
> CIFAR-10's published stats are `(0.49, 0.48, 0.45) / (0.25, 0.24, 0.26)`;
> using ImageNet's stats on CIFAR would degrade training slightly.


### Code cell #2
```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10 normalisation constants (per-channel mean/std).
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std  = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),       # augmentation
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])

train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
print("train size:", len(train_ds), " test size:", len(test_ds))

class CIFARCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = CIFARCNN()
print("parameter count:", sum(p.numel() for p in model.parameters()))
```

**What this code does:**

- **Headline:** `import torch`
- **Defines:** class `CIFARCNN`
- **Key imports:** `import torch`, `import torch.nn as nn`, `from torchvision import datasets`, `from torch.utils.data import DataLoader`
- **Notable calls:** `nn.Linear`, `nn.Conv2d`, `nn.BatchNorm`, `nn.Dropout`, `transforms`, `DataLoader`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Shows 8 random CIFAR-10 training images with their
> labels and the class distribution.
> 
> **Why it matters:** CIFAR-10 is the standard "next step after MNIST"
> benchmark. Images are 32×32 RGB, with 10 classes and 50k train / 10k test.
> It's small enough to train on CPU in minutes, hard enough to separate good
> models from bad ones.
> 
> **What to watch for:** CIFAR-10 images look *very* low-resolution at 32×32.
> Don't expect a CNN to recognise your dog from this data — the resolution is
> deliberately tiny to make research fast.


### Code cell #3
```python
import matplotlib.pyplot as plt
import numpy as np

class_names = train_ds.classes
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax in axes.flat:
    idx = np.random.randint(len(train_ds))
    img, label = train_ds[idx]
    # Undo normalisation for display.
    img = img * torch.tensor(cifar_std).view(3, 1, 1) + torch.tensor(cifar_mean).view(3, 1, 1)
    ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_title(class_names[label], fontsize=9)
    ax.axis("off")
plt.suptitle("8 random CIFAR-10 images")
plt.tight_layout(); plt.show()

counts = np.bincount([train_ds[i][1] for i in range(len(train_ds))], minlength=10)
print("per-class counts:", dict(zip(class_names, counts)))
```

**What this code does:**

- **Headline:** `import matplotlib.pyplot as plt`
- **Key imports:** `import matplotlib.pyplot as plt`, `import numpy as np`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** The full training loop with progress printed every
> epoch, plus a loss/accuracy plot at the end. We deliberately train for
> just 3 epochs so this notebook runs on CPU in ~2 minutes.
> 
> **Why it matters:** this is the loop you'll copy into every CV project.
> The skeleton never changes; only the model, dataset, and hyperparameters.
> 
> **What to watch for:** `model.train()` and `model.eval()` are critical
> because we use BatchNorm. Without them, BN would compute batch statistics
> at test time and accuracy would crash.


### Code cell #4
```python
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
history = {"train_loss": [], "train_acc": [], "test_acc": []}

for epoch in range(3):
    model.train()
    for xb, yb in train_loader:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward(); opt.step()
        history["train_loss"].append(loss.item())
    # Quick eval after each epoch.
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    test_acc = correct / total
    history["test_acc"].append(test_acc)
    print(f"epoch {epoch}: test_acc = {test_acc:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].plot(history["train_loss"]); axes[0].set_title("train loss"); axes[0].set_yscale("log")
axes[1].plot(history["test_acc"], "o-"); axes[1].set_title("test accuracy per epoch")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)`
- **Notable calls:** `DataLoader`, `CrossEntropyLoss`, `Adam`, `backward`, `model.train`, `model.eval`

> **Takeaway:** The training loop is the same skeleton for every CV model.
> Master it once, reuse it forever.

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> **What this does:** Computes a per-class confusion matrix on the test set
> and visualises it as a heatmap. The off-diagonal cells tell you *which*
> classes the model confuses.
> 
> **Why it matters:** accuracy is one number. The confusion matrix tells you
> *where* the model fails — and that drives your next iteration.
> 
> **What to watch for:** CIFAR-10's classic confusions are cat↔dog and
> deer↔horse. If you see those, the model is doing what every CIFAR model does.


### Code cell #5
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_pred.extend(model(xb).argmax(1).tolist())
        all_true.extend(yb.tolist())

cm = confusion_matrix(all_true, all_pred, labels=list(range(10)))
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    xticks_rotation=45, cmap="Blues", ax=ax, colorbar=False)
ax.set_title("CIFAR-10 confusion matrix")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay`
- **Key imports:** `from sklearn.metrics import confusion_matrix`
- **Notable calls:** `model.eval`

> ---
> 
> ## 8. Experiments — Experiments

> 1. Train for 10 epochs instead of 3 — does test accuracy reach 80%?
> 2. Remove augmentation (`RandomFlip`/`RandomCrop`) — does the gap between train and test grow?
> 3. Add a learning-rate scheduler (`CosineAnnealingLR`) — does the final accuracy improve?
> 4. Save the best model with `torch.save(model.state_dict(), 'cifar_cnn.pt')`.

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement a custom `Dataset` that loads images from a directory of subfolders.
> 2. Add mixup or cutmix augmentation and measure the improvement.
> 3. Implement early stopping on the test loss.
> 4. Use `torch.compile(model)` and benchmark speedup.
> 5. Train the same model with mixed precision (`torch.cuda.amp`).

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. Why do we use cross-entropy instead of MSE for classification?
> 2. What is data augmentation, and why does it help?
> 3. What is the difference between `model.train()` and `model.eval()`?
> 4. How would you handle a heavily imbalanced image dataset?
> 5. Why is BatchNorm sensitive to batch size?
> 6. What is `torch.compile`, and when does it help?
> 7. How would you scale training to multiple GPUs?
> 8. What is the difference between `nn.CrossEntropyLoss` and `F.cross_entropy`?
