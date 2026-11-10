# Code Walkthrough — `04_object_detection_yolo.ipynb`

> Module: **07-computer-vision** | Notebook: **04_object_detection_yolo.ipynb**
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

> # 04 — Object Detection & YOLO
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
> ## 1. Concept Overview — Object detection and YOLO

> Image classification answers *what is in this image?* Object detection
> answers *what is in this image and where?* — outputting a set of bounding
> boxes with class labels. YOLO (You Only Look Once) is the dominant
> single-stage detector family, known for its speed-accuracy trade-off.
> 
> **Beginner track:** understand the YOLO output format
> (`(x_center, y_center, width, height, class, confidence)` per box).
> **Advanced track:** also learn the YOLO loss function (bbox + objectness +
> classification) and the IoU/NMS post-processing pipeline.

> ---
> 
> ## 2. Mathematical Foundation — Mathematical Foundation

> Key concepts:
> 
> - **Bounding box**: $(x_c, y_c, w, h)$ — centre coordinates + size.
> - **IoU** (Intersection over Union): $\frac{|A \cap B|}{|A \cup B|}$.
> - **Anchor boxes**: fixed prior shapes the model predicts offsets from.
> - **Non-Maximum Suppression (NMS)**: remove overlapping boxes, keep the
>   highest-confidence one per region.
> - **mAP** (mean Average Precision): area under the precision-recall curve
>   averaged over IoU thresholds and classes.

> ---
> 
> ## 3. Implementation from Scratch — Implementation from Scratch

> **What this does:** Implements IoU (Intersection over Union) and NMS
> (Non-Maximum Suppression) from scratch in NumPy. These are the two
> post-processing operations every detector relies on.
> 
> **Why it matters:** If you understand IoU and NMS, you understand the
> *output* of every object detector. The detector just predicts boxes and
> scores; IoU + NMS turn those into a final clean set.
> 
> **What to watch for:** NMS is *greedy* — sort by confidence, keep top,
> remove anyone overlapping with it above an IoU threshold, repeat. There
> are smarter variants (Soft-NMS, DIoU-NMS) but the principle is the same.


### Code cell #1
```python
import numpy as np

def iou(box1, box2):
    """IoU of two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def nms(boxes, scores, iou_threshold=0.5):
    """Greedy NMS. boxes: (N, 4) in xyxy, scores: (N,)."""
    order = scores.argsort()[::-1]   # highest confidence first
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        # Remove boxes that overlap too much with the kept one.
        order = rest[np.array([iou(boxes[i], boxes[j]) for j in rest]) < iou_threshold]
    return keep

# Sanity check.
b1 = (0, 0, 10, 10)
b2 = (5, 5, 15, 15)
print("IoU of overlapping boxes:", iou(b1, b2))   # 25 / 175 = 0.143

boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
scores = np.array([0.9, 0.8, 0.7])
kept = nms(boxes, scores, iou_threshold=0.5)
print("boxes kept after NMS:", kept)
```

**What this code does:**

- **Headline:** `import numpy as np`
- **Defines:** def `iou` — IoU of two boxes in (x1, y1, x2, y2) format.; def `nms` — Greedy NMS. boxes: (N, 4) in xyxy, scores: (N,).
- **Key imports:** `import numpy as np`

> **Takeaway:** Every object detector outputs boxes + scores; IoU + NMS turn
> them into a clean prediction set. The detector's job is to predict; NMS's
> job is to clean up.

> ---
> 
> ## 4. Library Implementation — Library Implementation

> **What this does:** Uses `torchvision.models.detection.fasterrcnn_resnet50_fpn`
> with pretrained COCO weights to detect objects in a single image. This is
> the simplest possible "use a pretrained detector" code.
> 
> **Why it matters:** modern CV engineering rarely involves training a
> detector from scratch. You load a pretrained one and either use it as-is
> or fine-tune on a small custom set.
> 
> **What to watch for:** Faster R-CNN outputs *raw* boxes (no NMS applied by
> default in some versions). We filter by confidence threshold ourselves.


### Code cell #2
```python
from torchvision import models, transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the pretrained detector.
detector = models.detection.fasterrcnn_resnet50_fpn(
    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
detector.eval()

# Try a sample image; skip gracefully if network unavailable.
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
try:
    img = Image.open(__import__("io").BytesIO(
        __import__("requests").get(url, timeout=10).content)).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = detector(input_tensor)[0]
    # Keep only confident detections.
    keep = outputs["scores"] > 0.7
    boxes = outputs["boxes"][keep].tolist()
    labels = outputs["labels"][keep].tolist()
    print(f"detected {len(boxes)} object(s) with confidence > 0.7")
except Exception as e:
    print(f"skipped pretrained-detection demo (no network): {e}")
```

**What this code does:**

- **Headline:** `from torchvision import models, transforms`
- **Key imports:** `from torchvision import models`, `from PIL import Image`, `import torch`, `import matplotlib.pyplot as plt`
- **Notable calls:** `ResNet`, `transforms`

> **What this does:** Visualises the detected boxes on top of the image.
> Each box is drawn as a rectangle with the predicted class label.
> 
> **Why it matters:** a detector is only useful if you can show its output
> to a human. The visualisation code below is the foundation of every CV
> demo app.
> 
> **What to watch for:** `patches.Rectangle((x1, y1), w, h)` takes the
> *top-left* corner plus width/height — not the centre. Convert from xyxy
> to xywh before drawing.


### Code cell #3
```python
if "img" in dir():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    for box in boxes:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
    ax.set_title("Faster R-CNN detections")
    ax.axis("off")
    plt.tight_layout(); plt.show()
else:
    print("no image to visualise (network was unavailable)")
```

**What this code does:**

- **Headline:** `if "img" in dir():`

> ---
> 
> ## 5. Dataset Exploration — Dataset Exploration

> **What this does:** Loads the COCO class names (80 categories) and shows
> what `torchvision`'s pretrained detector can predict.
> 
> **Why it matters:** COCO is the standard object-detection benchmark. If
> your target classes are in COCO, you can use a pretrained detector
> out-of-the-box.
> 
> **What to watch for:** COCO has 80 classes but the pretrained Faster R-CNN
> outputs labels up to 91 — the extra IDs are an artifact of how COCO was
> built.


### Code cell #4
```python
# COCO class names (the 80 categories Faster R-CNN was trained on).
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
print(f"COCO has {len(COCO_CLASSES)} categories (including background)")
print("first 10:", COCO_CLASSES[:10])
```

**What this code does:**

- **Headline:** `COCO_CLASSES = [`

> ---
> 
> ## 6. Visualization — Visualization

> **What this does:** Plots IoU vs NMS-keep rate for a synthetic set of
> overlapping boxes. As IoU threshold increases, more boxes survive NMS.
> 
> **Why it watch for:** IoU threshold is the key knob in detection post-
> processing. Lower = more aggressive deduplication (may miss real objects);
> higher = more duplicate boxes. 0.5 is the standard default.
> 
> **What to watch for:** a synthetic experiment like this lets you build
> intuition for the IoU threshold without needing a real dataset.


### Code cell #5
```python
# Generate 100 random boxes around 5 "true" objects and apply NMS at
# different IoU thresholds.
rng = np.random.default_rng(0)
true_centers = rng.uniform(0, 100, (5, 2))
# Each "true" object gets ~20 noisy boxes around it.
boxes, scores = [], []
for cx, cy in true_centers:
    for _ in range(20):
        dx, dy = rng.normal(0, 5, 2)
        w, h = rng.uniform(15, 25, 2)
        boxes.append([cx + dx - w/2, cy + dy - h/2, cx + dx + w/2, cy + dy + h/2])
        scores.append(rng.uniform(0.5, 1.0))
boxes = np.array(boxes); scores = np.array(scores)

thresholds = [0.3, 0.5, 0.7, 0.9]
keeps = [len(nms(boxes, scores, t)) for t in thresholds]
plt.figure(figsize=(7, 3))
plt.plot(thresholds, keeps, "o-")
plt.xlabel("IoU threshold for NMS"); plt.ylabel("# boxes kept")
plt.title("NMS sensitivity to IoU threshold")
plt.tight_layout(); plt.show()
```

**What this code does:**

- **Headline:** `rng = np.random.default_rng(0)`

> ---
> 
> ## 7. Model Evaluation — Model Evaluation

> Detection is evaluated with **mAP** (mean Average Precision):
> 
> - For each class, compute the precision-recall curve at various IoU thresholds.
> - Average over IoU thresholds (typically 0.5 to 0.95 in COCO).
> - Average over classes.
> 
> A mAP@[0.5:0.95] of 0.50 on COCO is a solid baseline; the best models reach ~0.65.

> **What this does:** Computes the Average Precision (AP) for one class
> given a set of predicted boxes and ground-truth boxes, using IoU=0.5.
> This is the building block of mAP.
> 
> **Why it matters:** AP is the standard detection metric. Once you can
> compute it for one class, you can compute mAP by averaging over classes.
> 
> **What to watch for:** predictions are matched to ground-truth *greedily*
> by IoU — each GT box can be claimed by at most one prediction.


### Code cell #6
```python
def average_precision(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """AP for a single class. pred_boxes: (N,4), gt_boxes: (M,4)."""
    order = pred_scores.argsort()[::-1]
    pred_boxes = pred_boxes[order]
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    used = np.zeros(len(gt_boxes), dtype=bool)
    for i, pb in enumerate(pred_boxes):
        ious = np.array([iou(pb, gb) for gb in gt_boxes])
        if len(ious) == 0:
            fp[i] = 1; continue
        best = ious.argmax()
        if ious[best] >= iou_thresh and not used[best]:
            tp[i] = 1; used[best] = True
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    recall = tp_cum / max(1, len(gt_boxes))
    precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
    # AP = area under precision-recall curve (step function).
    ap = np.trapz(precision, recall)
    return ap

# Toy example: 3 GT boxes, 5 predictions (some correct, some wrong).
gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]], dtype=float)
pred = np.array([
    [1, 1, 11, 11],      # matches gt[0]
    [49, 49, 61, 61],    # matches gt[1]
    [99, 99, 111, 111],  # matches gt[2]
    [200, 200, 210, 210],# wrong
    [5, 5, 15, 15],      # overlaps gt[0] but gt[0] already used
], dtype=float)
scores = np.array([0.9, 0.85, 0.8, 0.7, 0.6])
ap = average_precision(pred, scores, gt, iou_thresh=0.5)
print(f"AP@0.5 for this class: {ap:.3f}")
```

**What this code does:**

- **Headline:** `def average_precision(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):`
- **Defines:** def `average_precision` — AP for a single class. pred_boxes: (N,4), gt_boxes: (M,4).

> ---
> 
> ## 8. Experiments — Experiments

> 1. Run the pretrained Faster R-CNN on 10 different images and count the
>   average number of detected objects per image.
> 2. Replace Faster R-CNN with `retinanet_resnet50_fpn` — does accuracy / speed differ?
> 3. Lower the confidence threshold from 0.7 to 0.3 — what happens to the
>   precision-recall trade-off?
> 4. Train a YOLOv8 model on a small custom dataset using the `ultralytics`
>   package (Colab-recommended).

> ---
> 
> ## 9. Hands-on Challenges — Hands-on Challenges

> 1. Implement Soft-NMS and compare to standard NMS on a cluttered scene.
> 2. Implement the full COCO mAP@[0.5:0.95] metric from scratch.
> 3. Build a custom COCO-format dataset from a folder of labelled images.
> 4. Fine-tune a pretrained Faster R-CNN on your custom dataset for 5 epochs.
> 5. Implement mean Average Recall (mAR) and compare with mAP.

> ---
> 
> ## 10. Interview Questions — Interview Questions

> 1. What is the difference between one-stage (YOLO) and two-stage (Faster R-CNN) detectors?
> 2. What is IoU, and why is it used in NMS and mAP?
> 3. What are anchor boxes, and why do detectors use them?
> 4. How does NMS prevent duplicate detections?
> 5. What is the difference between AP and mAP?
> 6. Why is COCO mAP averaged over IoU thresholds 0.5 to 0.95?
> 7. How would you evaluate a detector on a heavily occluded scene?
> 8. What is the YOLO loss function composed of?
