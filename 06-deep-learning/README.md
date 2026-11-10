# Module 06 — Deep Learning

The bridge from classical ML to modern AI. Neural networks, backpropagation,
and PyTorch — the foundation of every later module.

| Notebook                                  | Topic                                                                  | Code walkthrough                                                            |
|-------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `01_neural_networks_from_scratch.ipynb`    | MLP forward pass, activations, softmax, train loop on `make_moons`.  | [README_01_neural_networks_from_scratch.md](README_01_neural_networks_from_scratch.md) |
| `02_backpropagation_from_scratch.ipynb`   | Hand-rolled backprop on a 1-hidden-layer MLP; verified with `gradcheck`.| [README_02_backpropagation_from_scratch.md](README_02_backpropagation_from_scratch.md) |
| `03_pytorch_fundamentals.ipynb`           | Tensors, autograd, `nn.Module`, `Dataset`/`DataLoader`, training loop. | [README_03_pytorch_fundamentals.md](README_03_pytorch_fundamentals.md)       |
| `04_training_deep_networks.ipynb`         | Initialisation, BatchNorm, Dropout, LR schedules, early stopping.     | [README_04_training_deep_networks.md](README_04_training_deep_networks.md)   |

Each notebook has a companion `README_<notebook-name>.md` that walks through
every code cell — open it alongside the notebook for a written explanation of
what each block does and why.
