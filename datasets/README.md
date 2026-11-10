# Datasets

Most notebooks in this cookbook fetch their data **live from public sources** at runtime — so you do not need to download anything in advance. The notebooks use these public URLs:

| Dataset       | Where it is used                  | Source                                                                                              |
|---------------|-----------------------------------|-----------------------------------------------------------------------------------------------------|
| Iris          | Module 01 (numpy, pandas, stats, viz), Module 04, Module 05 | `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv`                            |
| Titanic       | Module 01 (pandas), Module 03, Module 04 | `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`                    |
| Diamonds      | Module 03                         | `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv`                        |
| Boston Housing| Module 04                        | `https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv`                       |
| Breast Cancer | Module 04                         | `sklearn.datasets.load_breast_cancer` (bundled)                                                     |
| Wine          | Module 03, Module 05              | `https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data` and `sklearn.datasets.load_wine` |
| California Housing | Module 03, Module 04         | `sklearn.datasets.fetch_california_housing`                                                         |
| Digits (8x8)  | Module 05                         | `sklearn.datasets.load_digits`                                                                      |
| MNIST         | Module 05                         | `sklearn.datasets.fetch_openml("mnist_784")`                                                         |
| Mall Customers | Module 05                        | `https://raw.githubusercontent.com/AbhilipsaSenapati/Mall-Customer-Segmentation/main/Mall_Customers.csv` (with offline fallback) |

If you want to work fully offline, drop the CSV files in this directory and edit the notebook's `pd.read_csv(...)` calls to point at the local paths.
