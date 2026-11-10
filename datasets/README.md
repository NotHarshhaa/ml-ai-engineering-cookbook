# Datasets

Most notebooks in this cookbook fetch their data **live from public sources** at runtime — so you do not need to download anything in advance. The notebooks use these public URLs and library-bundled datasets:

| Dataset       | Where it is used                  | Source                                                                                              |
|---------------|-----------------------------------|-----------------------------------------------------------------------------------------------------|
| Iris          | Modules 01, 04, 05 | `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv`                            |
| Titanic       | Modules 01, 03, 04 | `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`                    |
| Diamonds      | Module 03 | `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv`                        |
| Boston Housing| Module 04 | `https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv`                       |
| Breast Cancer | Module 04 | `sklearn.datasets.load_breast_cancer` (bundled)                                                     |
| Wine          | Modules 03, 05 | `https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data` and `sklearn.datasets.load_wine` |
| California Housing | Modules 03, 04 | `sklearn.datasets.fetch_california_housing`                                                  |
| Digits (8x8)  | Module 05 | `sklearn.datasets.load_digits`                                                                      |
| MNIST         | Module 07 | `torchvision.datasets.MNIST` (auto-downloads on first use)                                          |
| CIFAR-10      | Module 07 | `torchvision.datasets.CIFAR10` (auto-downloads on first use)                                        |
| ImageNet sample | Module 07 | `https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg`                             |
| COCO class names | Module 07 | inline list in the notebook                                                                       |
| IMDB          | Module 08 | `datasets.load_dataset("imdb")` (auto-downloads on first use)                                       |
| GLUE SST-2    | Module 09 | `datasets.load_dataset("glue", "sst2")` (auto-downloads on first use)                               |
| Tiny Shakespeare | Module 09 | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare.txt`            |

All `torchvision.datasets.MNIST/CIFAR10`, `datasets.load_dataset`, and
`transformers.AutoTokenizer.from_pretrained` calls download to a local cache
on first use, then use the cached files for subsequent runs.

If you want to work fully offline, drop the CSV files in this directory and edit the notebook's `pd.read_csv(...)` calls to point at the local paths. For the torchvision/datasets/transformers calls, set `HF_DATASETS_OFFLINE=1` and pre-download once with internet access.
