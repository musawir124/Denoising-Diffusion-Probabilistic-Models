# Wine Quality Assessment — MSc Data Science FYP (Topic 54)

**Classification of Wine Quality Tiers Using Physicochemical Tests**

Author: **Musawir Hussain** — MSc Data Science, University of Hertfordshire
Module: 7PAM2002-0206-2025 — Individual Project (Semester B, 2025/26)
Dataset: UCI Wine Quality (DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T))

---

## 1. Project summary

This project treats wine quality grading as a binary supervised-learning problem:

> Given eleven standardised physicochemical measurements, predict whether a Portuguese *vinho verde* sample will receive a sensory panel score of **6 or higher (Good)** or below 6 **(Poor)**.

Three classical classifiers (Logistic Regression, Random Forest, Support Vector Machine with an RBF kernel) are compared under three experimental regimes (baseline, SMOTE-balanced, GridSearchCV-tuned) and the winning model is opened up with feature-importance, Partial Dependence and SHAP analyses so the decision rule is interpretable.

**Best result:** tuned SVM — test accuracy 0.775, F1 = 0.825, AUC = 0.826. Tuned Random Forest is a close second (F1 = 0.821, AUC = 0.834) and is the most stable under 5-fold CV (F1 std = 0.0082).

---

## 2. Repository layout

```
.
├── 1. Topic Form/                Project selection form
├── 2. Coding part/
│   ├── 1. EDA/                   Initial exploratory notebook
│   ├── 2. Models Training/       Baseline + SMOTE training
│   ├── 3. Hyper Tuning/          Grid search + CV + SHAP
│   └── 4. Final Code/
│       └── Topic_54_jan_26_hyper_final.ipynb   ← canonical notebook
├── 3. Presentation Part/         Slide deck and handout
├── 4. Complete Report/
│   ├── FYP_Wine_Quality_Musawir_Hussain.docx   ← final report
│   ├── Project Handbook Sem B MSc Data Science 2025.docx
│   └── README.md                 ← this file
└── testing/                      Scratch experiments
```

The canonical, marked version of the code is `2. Coding part/4. Final Code/Topic_54_jan_26_hyper_final.ipynb`. Every number, table and figure in the Final Project Report is produced directly by this notebook.

---

## 3. Environment and dependencies

Python 3.12 is required. The following third-party packages are used:

| Package            | Version tested | Purpose                               |
|--------------------|----------------|---------------------------------------|
| `ucimlrepo`        | 0.0.7          | Programmatic fetch of UCI dataset 186 |
| `pandas`           | 2.2            | Data handling                         |
| `numpy`            | 2.0            | Numerical arrays                      |
| `scikit-learn`     | 1.4+           | Models, metrics, GridSearchCV         |
| `imbalanced-learn` | 0.12           | SMOTE                                 |
| `shap`             | 0.44+          | SHAP TreeExplainer                    |
| `matplotlib`       | 3.8+           | Plots                                 |
| `seaborn`          | 0.13+          | Styled plots                          |

Install in one line:

```bash
pip install ucimlrepo pandas numpy scikit-learn imbalanced-learn shap matplotlib seaborn
```

No GPU, internet access during training, or proprietary software is required. The dataset is fetched by `ucimlrepo` at runtime, so an internet connection is needed on the first cell only.

---

## 4. How to reproduce every number in the report

1. Clone the repository and open `2. Coding part/4. Final Code/Topic_54_jan_26_hyper_final.ipynb` in Jupyter or VS Code.
2. Run the cells top-to-bottom. Every `random_state` is fixed at **42** where a pseudo-random choice is made (`train_test_split`, `SMOTE`, `RandomForestClassifier`, `SVC`), so results are deterministic.
3. The expected outputs for the key cells are:

| Stage                    | Expected output                                                                 |
|--------------------------|----------------------------------------------------------------------------------|
| After deduplication      | `Dataset shape: (5318, 12)` (1,179 duplicates removed from 6,497 originals)      |
| Binary target            | Good = 62.6 %, Poor = 37.4 %                                                     |
| Train/test split         | (4254, 11) / (1064, 11)                                                          |
| SMOTE                    | `Counter({1: 2665, 0: 1589})` → `Counter({1: 2665, 0: 2665})`                    |
| Best LR hyperparameters  | `{'C': 10, 'solver': 'lbfgs'}`                                                   |
| Best RF hyperparameters  | `{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}`               |
| Best SVM hyperparameters | `{'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}`                                    |
| Most stable CV model     | `Random Forest` (avg std lowest on every metric)                                 |

---

## 5. Verified numerical results

### 5.1 Baseline

| Model               | Accuracy | Balanced Acc | Precision | Recall | F1      | AUC    |
|---------------------|----------|--------------|-----------|--------|---------|--------|
| Logistic Regression | 0.7528   | 0.7171       | 0.7719    | 0.8589 | 0.8131  | 0.8047 |
| Random Forest       | 0.7726   | 0.7455       | 0.7978    | 0.8529 | 0.8244  | 0.8300 |
| SVM (RBF)           | 0.7744   | 0.7465       | 0.7975    | 0.8574 | 0.8263  | 0.8315 |

### 5.2 SMOTE-balanced

| Model               | Accuracy | Balanced Acc | Precision | Recall | F1      | AUC    |
|---------------------|----------|--------------|-----------|--------|---------|--------|
| Logistic Regression | 0.7397   | 0.7390       | 0.8247    | 0.7417 | 0.7810  | 0.8040 |
| Random Forest       | 0.7575   | 0.7467       | 0.8168    | 0.7898 | 0.8031  | 0.8253 |
| SVM (RBF)           | 0.7519   | 0.7573       | 0.8478    | 0.7357 | 0.7878  | 0.8298 |

### 5.3 GridSearchCV-tuned

| Model               | Accuracy | Balanced Acc | Precision | Recall | F1      | AUC    |
|---------------------|----------|--------------|-----------|--------|---------|--------|
| Logistic Regression | 0.7528   | 0.7171       | 0.7719    | 0.8589 | 0.8131  | 0.8047 |
| Random Forest       | 0.7688   | 0.7420       | 0.7958    | 0.8483 | 0.8212  | 0.8337 |
| SVM (RBF)           | 0.7754   | 0.7518       | 0.8054    | 0.8453 | 0.8249  | 0.8259 |

### 5.4 Five-fold cross-validation (training fold)

| Model               | F1 mean ± std         | Accuracy mean ± std    |
|---------------------|-----------------------|------------------------|
| Logistic Regression | 0.8041 ± 0.0136       | 0.7461 ± 0.0169        |
| Random Forest       | **0.8214 ± 0.0082**   | **0.7708 ± 0.0100**    |
| SVM (RBF)           | 0.8174 ± 0.0104       | 0.7654 ± 0.0128        |

### 5.5 Feature importance (tuned Random Forest)

| Rank | Feature                | Importance |
|------|------------------------|------------|
| 1    | `alcohol`              | 0.1659     |
| 2    | `volatile_acidity`     | 0.1104     |
| 3    | `density`              | 0.1067     |
| 4    | `total_sulfur_dioxide` | 0.0845     |
| 5    | `free_sulfur_dioxide`  | 0.0842     |
| 6    | `chlorides`            | 0.0822     |
| 7    | `sulphates`            | 0.0798     |
| 8    | `citric_acid`          | 0.0767     |
| 9    | `residual_sugar`       | 0.0730     |
| 10   | `pH`                   | 0.0702     |
| 11   | `fixed_acidity`        | 0.0663     |

### 5.6 Pearson correlation with binary target

| Feature              | r vs `binary_quality` |
|----------------------|-----------------------|
| alcohol              | +0.414                |
| citric_acid          | +0.087                |
| free_sulfur_dioxide  | +0.047                |
| pH                   | +0.036                |
| sulphates            | +0.036                |
| residual_sugar       | −0.048                |
| total_sulfur_dioxide | −0.049                |
| fixed_acidity        | −0.072                |
| chlorides            | −0.187                |
| volatile_acidity     | −0.270                |
| density              | −0.287                |

---

## 6. Notebook structure (cell map)

| Cells | Purpose |
|-------|---------|
| 0–3   | Dataset acquisition via `ucimlrepo.fetch_ucirepo(id=186)` |
| 4–11  | Inspection, missing-value check, correlation heatmap, alcohol-vs-quality boxplot |
| 12–17 | Binary transformation (`quality >= 6`) and class-balance bar / pie charts |
| 18–21 | Stratified 80/20 split, `StandardScaler` fitted on train only |
| 22–27 | Baseline training of LR, RF, SVM; classification reports, confusion matrices, ROC |
| 28–33 | SMOTE resampling and re-training |
| 34–39 | `GridSearchCV` tuning and evaluation on held-out test set |
| 40–43 | Consolidated comparison table and F1 bar chart |
| 44–47 | 5-fold cross-validation stability analysis |
| 48–52 | Feature importance, Partial Dependence Plots, SHAP summary |

---

## 7. Ethical considerations

The UCI Wine Quality dataset contains no personal data — only laboratory measurements and aggregated anonymous sensory scores from three unnamed oenology experts. It therefore falls outside the scope of UK GDPR and did not require UH Research Ethics Committee approval. The dataset is published under a permissive licence by a reputable academic source (University of Minho, 2009) and is widely reused in the machine-learning literature. The classifier is framed strictly as a decision aid, never as a replacement for sensory evaluation, and its regional bias (Portuguese *vinho verde* only) is stated explicitly in the report.

---

## 8. Declaration on AI tools

All code in this repository and all writing in the Final Project Report were produced by Musawir Hussain. No sections of the code or the report were generated by ChatGPT or any other generative AI tool. Libraries used are all open-source and cited in the report.

---

## 9. Contact

Musawir Hussain — MSc Data Science — University of Hertfordshire.
Supervisor and marker queries should be directed through the Canvas assignment page for module 7PAM2002-0206-2025.




