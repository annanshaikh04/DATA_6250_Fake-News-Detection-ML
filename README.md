# 📰 Fake News Detection using Machine Learning & Transformer Embeddings

> ML4DS | DATA 6250 | Wentworth Institute of Technology  
> Author: Annanahmed Furkanahmed Shaikh  

---

## 📖 Project Overview

This project focuses on detecting Fake News using advanced machine learning techniques, leveraging transformer embeddings, feature engineering, dimensionality reduction, clustering, model experiments, hyperparameter tuning, interpretability, and robustness testing.

---

## 🎯 Problem Statement

Fake News is a rising challenge in online platforms. The goal is to automatically classify news articles as Fake (1) or Real (0) based on their text content using ML models.

---

## 📁 Dataset Description

Dataset Source → Kaggle  
Used Files → `train.csv` & `test.csv`  

### Columns:
| Column | Description |
|--------|-------------|
| id     | Unique Identifier |
| title  | News Title |
| author | News Author |
| text   | Full News Content |
| label  | Target (1 = Fake, 0 = Real) |

---

## 🔨 Preprocessing Steps

File Used → `01_PreProcessing.ipynb`

- Imputed missing authors using **Sentence Transformers**.
- Generated `real_confidence` & `fake_confidence` using pre-trained models.
- Applied **UMAP** (reduced 384 transformer features → 80).
- Performed **HDBSCAN** clustering on text features (title, text, author).
- Removed high-cardinality & redundant features.
- Balanced numeric feature ranges using **MinMaxScaler**.
- Saved final datasets → `train.csv` & `test.csv`.

---

## 🧪 Experiments Performed (Experiment 1 - 8)

| Exp No. | Technique | Purpose | Result |
|---------|-----------|---------|--------|
| Exp 1 | MinMaxScaler | Scaling numeric features | Tree models overfitted, LR & SVM improved |
| Exp 2 | UMAP + HDBSCAN | Reduce text dimensionality | Faster training, minor accuracy drop |
| Exp 3 | PCA | Further compression | Small drop in linear models accuracy |
| Exp 4 | Noise Addition | Test model robustness | RF & GB survived, SVM failed |
| Exp 5 | SHAP & LIME | Explainability | Top features = UMAP Components & Real Confidence |
| Exp 6 | Efficiency Check | Compare training/inference time | GB balanced, SVM slowest |
| Exp 7 | Hyperparameter Tuning | Best parameter search | GB showed balanced performance |
| Exp 8 | CV vs Validation Accuracy | Overfitting check | RF overfit, GB generalizes well |

---

## 🤖 Models Trained

| Model                  | Notes |
|-----------------------|-------|
| Logistic Regression   | Baseline linear model |
| Support Vector Machine (SVM) | Multiple kernels tested |
| Decision Tree         | Simple tree learner |
| Random Forest         | Ensemble of Trees, Overfit in clean data |
| Gradient Boosting     | Final Best Performing Model |

---

## 🔍 Final Results

| Model | CV Accuracy | Validation Accuracy | Inference Time | Memory Usage |
|-------|-------------|---------------------|----------------|--------------|
| Logistic Regression | 78.68% | 78.79% | 1.3 sec | 84.9 MB |
| SVM | 79.64% | 80.11% | 32.6 sec | 107.6 MB |
| Decision Tree | 81.06% | 86.75% | 0.8 sec | 96.3 MB |
| Random Forest | 84.23% | 99.28% | 3.9 sec | 118.2 MB |
| Gradient Boosting | 82.67% | 86.87% | 2.4 sec | 103.9 MB |

---

## 🏆 Final Model Selection: Gradient Boosting Classifier (GB)

Reasons:
- Consistent performance across all experiments
- Generalized well without overfitting
- Efficient on both time and memory
- Explainable using SHAP & LIME

---

## 🧠 Interpretability Results

- SHAP Importance: Top features = `reduced_component_47`, `real_confidence`
- LIME Explanation: Clear reasoning for individual predictions

---

## 🚀 Future Improvements

If more time available:
- Try **XGBoost / LightGBM** for faster performance
- Perform SHAP-based Feature Selection
- Use RandomizedSearchCV for faster tuning
- Deploy via Streamlit / Flask
- Explore Ensemble Stacking

---

## 📦 Repository Files

| File | Purpose |
|------|---------|
| `01_PreProcessing.ipynb` | Complete preprocessing pipeline |
| `02_Model_Training_and_Experiments.ipynb` | Model Training + Experiments + Evaluation |
| `train.csv` | Final Cleaned Train Dataset |
| `test.csv` | Final Cleaned Test Dataset |
| `submit.csv` | Optional Submission Output |

---

## 👤 Author Information

- Name: Annanahmed Furkanahmed Shaikh  
- MS Data Science | Wentworth Institute of Technology  
- Course: DATA 6250 — Machine Learning for Data Science  
- Instructor: Dr. Memo Ergezer  
- Email: shaikha4@wit.edu  

---

⭐ *If you found this project helpful — please star the repository!*
