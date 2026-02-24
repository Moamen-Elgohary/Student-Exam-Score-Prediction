# Student Exam Score Prediction

A supervised machine learning project that explores and compares regression models to predict student exam scores based on academic, behavioral, and socioeconomic factors.

---

## Problem Statement

Predicting student academic performance is a critical challenge in educational data mining. Early and accurate prediction of exam scores can help educators identify at-risk students, allocate resources efficiently, and tailor interventions. This project aims to answer:

> **Can we accurately predict a student's exam score from factors such as study habits, attendance, parental involvement, and socioeconomic background — and which regression model best generalizes to unseen data?**

---

## Dataset

- **Source:** [Student Performance Factors — Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- **Size:** 6,607 records × 20 features (after cleaning: 6,275 records)
- **Target Variable:** `Exam_Score`

### Features Overview

| Type | Features |
|------|----------|
| Numerical | `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity` |
| Categorical | `Parental_Involvement`, `Access_to_Resources`, `Motivation_Level`, `Internet_Access`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`, `Extracurricular_Activities` |

> **Dataset Access:** The dataset is loaded directly from Kaggle via the `kagglehub` library. No manual download is required — simply run the notebook and it will be fetched automatically.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of the target variable (`Exam_Score`)
- Scatter plots to visualize feature–target relationships (e.g., `Hours_Studied` vs `Exam_Score`)
- Identification of missing values across columns

### 2. Data Preprocessing
- **Missing Value Handling:** Rows with null values in `Teacher_Quality` (78), `Parental_Education_Level` (90), and `Distance_from_Home` (67) were dropped → 6,378 rows retained
- **Outlier Removal:** IQR method applied on `Exam_Score` → 103 outliers removed, 6,275 rows retained
- **Encoding:** Categorical variables encoded using Label Encoding
- **Train/Test Split:** 80% training, 20% testing

### 3. Models Evaluated

Four regression approaches were compared, progressing from simple to complex:

| # | Model | Description |
|---|-------|-------------|
| 1 | **Simple Linear Regression** | Single feature (`Hours_Studied`) as predictor |
| 2 | **Multi-variable Linear Regression** | All numerical features only (before encoding) |
| 3 | **Linear Regression (All Features)** | All 19 features after full encoding — best balance of accuracy and generalization |
| 4 | **Polynomial Regression (degree=2)** | Non-linear extension applied on all encoded features |
| 5 | **Polynomial Regression (degree=5)** | High-degree polynomial — included to demonstrate overfitting |

---

## Results & Model Comparison

### 1. Simple Linear Regression (`Hours_Studied` only)
| Set | MSE | MAE | R² |
|-----|-----|-----|-----|
| Training | 8.0137 | 2.3336 | 0.2376 |
| Testing | 7.5133 | 2.2860 | 0.3082 |

> Hours studied alone is a weak predictor. The low R² (~0.24–0.31) indicates that a single feature captures very little of the variance in exam scores.

---

### 2. Multi-variable Linear Regression (Numerical Features Only)
| Set | MSE | MAE | R² |
|-----|-----|-----|-----|
| Training | 2.0391 | 1.1462 | 0.8060 |
| Testing | 1.8306 | 1.0895 | 0.8314 |

> Adding all numerical features significantly boosts performance. R² improves to ~0.83, showing that student behavior and background together are strong predictors.

---

### 3. Linear Regression (All Features — After Encoding)  Best Model
| Set | MSE | MAE | R² |
|-----|-----|-----|-----|
| Training | 0.0987 | 0.2657 | 0.9906 |
| Testing | **0.1024** | **0.2710** | **0.9906** |

> Including encoded categorical features dramatically improves the model. It achieves an R² of **0.9906** on both training and testing sets, with virtually no gap — indicating excellent generalization and no overfitting.

---

### 4. Polynomial Regression (degree=2)
| Set | MSE | MAE | R² |
|-----|-----|-----|-----|
| Training | 0.0952 | 0.2600 | 0.9909 |
| Testing | 0.1042 | 0.2725 | 0.9904 |

> Marginally lower training error than linear, but slightly higher test error. The small train–test gap confirms minor overfitting, and the added complexity provides no real benefit over plain Linear Regression.

---

### 5. Polynomial Regression (degree=5)  Overfitting
| Set | MSE | MAE | R² |
|-----|-----|-----|-----|
| Training | 0.0000 | 0.0000 | 1.0000 |
| Testing | 4.0850 | 1.2764 | 0.6239 |

> This model perfectly memorizes the training data (R² = 1.0, MSE = 0.0), but completely fails to generalize. Test MSE jumps to **4.0850** and R² collapses to **0.6239** — a textbook case of **overfitting**. The model learns noise rather than true patterns, making it useless for real-world prediction.

---

## Model Comparison Summary

| Model | Train MSE | Test MSE | Train R² | Test R² |
|-------|-----------|----------|----------|---------|
| Simple Linear Regression | 8.0137 | 7.5133 | 0.2376 | 0.3082 |
| Multi-variable Linear Regression | 2.0391 | 1.8306 | 0.8060 | 0.8314 |
| Linear Regression (All Features) | 0.0987 | 0.1024 | 0.9906 | 0.9906 |
| Polynomial Regression (degree=2) | 0.0952 | 0.1042 | 0.9909 | 0.9904 |
| Polynomial Regression (degree=5) | 0.0000 | 4.0850 | 1.0000 | 0.6239 |

---

## Conclusion

The **Linear Regression model trained on all encoded features** is the clear winner. It achieves an R² of **0.9906** on both train and test sets, with an almost negligible gap — demonstrating strong predictive power and excellent generalization.

Key takeaways from the model comparison:
- **Simple Linear Regression** significantly underfits. One feature is not enough to capture the complexity of exam score prediction.
- **Multi-variable Linear Regression** (numerical only) shows a solid jump to R² ~0.83, proving that behavioral and socioeconomic factors carry real predictive weight.
- **Polynomial degree=2** achieves nearly identical results to Linear Regression (All Features) — Train R² of 0.9909 vs 0.9906, Test R² of 0.9904 vs 0.9906 — but at the cost of significantly higher model complexity. When two models perform this closely, the simpler one is always preferred: it is easier to interpret, faster to train, and less prone to overfitting on new data. Linear Regression wins here on the principle of parsimony.
- **Polynomial degree=5** is a clear case of **overfitting** — perfect training performance but poor generalization, with test R² dropping from 0.99 to 0.62 and test MSE exploding to 4.09.

**Final recommendation:** Linear Regression with all encoded features is the optimal model. Not only does it match or outperform the polynomial alternatives on test data, it does so with the simplest possible form — making it the most reliable and practical choice for predicting student exam scores.

---

## Technologies Used

- Python 3.x
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — modeling & evaluation
- `kagglehub` — dataset loading

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Moamen-Elgohary/Student-Exam-Score-Prediction.git
   cd student-exam-score-prediction
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook Student_Exam_Score_Prediction.ipynb
   ```
   > The dataset will be downloaded automatically from Kaggle on first run.

---

## Repository Structure

```
student-exam-score-prediction/
│
├── Student_Exam_Score_Prediction.ipynb
└── README.md
```

---

## License

This project is intended for educational purposes. Dataset credit: [lainguyn123 on Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors).

This work was submitted as part of a machine learning internship at **Elevvo Pathways**.
