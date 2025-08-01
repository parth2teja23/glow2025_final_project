# Wine Quality Classification

This project builds a classification model to predict red wine quality based on physicochemical properties. It transforms the original multi-class score into a binary classification problem: predicting whether a wine is **"good"** or **"bad"**.

## Dataset

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **File:** `winequality-red.csv`
- **Attributes:** 11 numerical input features (e.g., acidity, alcohol, pH), 1 quality score (0–10)
- **Labeling:**
  - **Bad (0):** Quality ≤ 6.5
  - **Good (1):** Quality > 6.5

## Project Structure

- `01_wine_data_prep_eval.ipynb`: Data preprocessing, outlier removal, normalization, label transformation
- `02_knn.ipynb`: K-Nearest Neighbors implementation and evaluation
- `03_naive_bayes.ipynb`: Naive Bayes model and performance report
- `04_decision_tree.ipynb `: Decision Tree implementation and evaluation

## Features

- Outlier removal using **IQR with 1.15 multiplier**
- Label binning using `pandas.cut`
- Feature scaling via `StandardScaler`
- Evaluation using:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
- Performance visualizations using Seaborn

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
