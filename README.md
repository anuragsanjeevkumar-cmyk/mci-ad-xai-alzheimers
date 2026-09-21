# 🧠 Predicting MCI-to-Alzheimer’s Conversion with Explainable Machine Learning

### A Cross-Model Study of Explainable AI Attribution Consistency and Stability

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)](https://xgboost.readthedocs.io/)
[![Random Forest](https://img.shields.io/badge/Random%20Forest-ML-green)](https://scikit-learn.org/)
[![Explainable Boosting Machine](https://img.shields.io/badge/EBM-Explainable%20AI-purple)](https://interpret.ml/)
[![SHAP](https://img.shields.io/badge/SHAP-XAI-red)](https://shap.readthedocs.io/)
[![ADNI](https://img.shields.io/badge/Data-ADNI-blueviolet)](https://adni.loni.usc.edu/)

---

https://mci-ad-xai-alzheimers-aj7r6bsxqgvnjbs7y8eloa.streamlit.app/

## 📌 Overview

This project develops and evaluates machine learning models for predicting **two-year conversion from Mild Cognitive Impairment (MCI) to an Alzheimer’s-related diagnosis** using data derived from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)**.

The project goes beyond conventional predictive performance evaluation.

The central research focus is:

> **How consistently and stably do different machine learning models attribute features when predicting the conversion of patients with Mild Cognitive Impairment to Alzheimer’s disease?**

Three different machine-learning model families were evaluated:

* **XGBoost**
* **Random Forest**
* **Explainable Boosting Machine (EBM)**

Explainable AI techniques were then used to investigate whether these models identify similar important features.

The project evaluates:

1. Predictive performance
2. Cross-model feature-attribution consistency
3. Top-feature agreement
4. Attribution stability across cross-validation folds

---

# 🎯 Research Question

> **How consistently and stably do different machine learning models attribute features when predicting the conversion of patients with Mild Cognitive Impairment to Alzheimer’s disease?**

The goal is not simply to find a model with good predictive performance.

Instead, the project asks whether the **explanations produced by different models are themselves reliable and consistent**.

---

# 🔬 Research Contribution

The main contribution of this project is a systematic evaluation of **cross-model feature-attribution consistency and resampling stability** using:

* A common ADNI-derived cohort
* A common two-year conversion outcome
* A common baseline feature set
* Three different model families
* SHAP-based explanations for tree models
* Native explanations from EBM
* Rank-based consistency metrics
* Five-fold stratified cross-validation

This allows the project to distinguish between:

> **"A model predicts well"**

and

> **"Different models explain their predictions in similar ways."**

---

# 🧬 Dataset

The project uses data from the:

**Alzheimer’s Disease Neuroimaging Initiative (ADNI)**

ADNI is a longitudinal research dataset containing clinical, cognitive, demographic, genetic, and neuroimaging information collected from participants across multiple study phases.

🔗 [ADNI Official Website](https://adni.loni.usc.edu/)

### Important Data Note

The raw ADNI data are **not included in this repository**.

Access to ADNI data requires appropriate authorization through ADNI.

Therefore, this repository contains the analysis code, notebooks, results, and project documentation rather than redistributing restricted/raw ADNI data.

---

# 👥 Cohort Construction

The cohort was constructed from longitudinal diagnostic information.

### Baseline

For each participant, the **first recorded MCI visit with a valid examination date** was used as the prediction baseline.

This provides a consistent starting point for determining whether conversion occurs during the subsequent two-year period.

### Outcome Definition

The binary outcome was defined as:

| Outcome | Definition                                                                          |
| ------- | ----------------------------------------------------------------------------------- |
| `1`     | AD-related conversion observed within 2 years after the MCI baseline                |
| `0`     | No AD-related conversion observed within 2 years AND sufficient follow-up available |

### Final Cohort

| Statistic                                     |      Value |
| --------------------------------------------- | ---------: |
| Participants with valid dated first MCI visit |      1,753 |
| Final modeling cohort                         |    **997** |
| Converters                                    |    **119** |
| Non-converters                                |    **878** |
| Positive-class proportion                     | **11.94%** |

The resulting dataset therefore represents a relatively imbalanced binary classification problem.

---

# 🧩 Features

The primary model uses nine baseline feature groups.

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `AGE`      | Participant age                       |
| `PTGENDER` | Gender/sex variable                   |
| `PTEDUCAT` | Education                             |
| `PTETHCAT` | Ethnicity category                    |
| `PTRACCAT` | Race category                         |
| `APOE`     | APOE genotype                         |
| `CDGLOBAL` | Clinical Dementia Rating Global Score |
| `CDRSB`    | Clinical Dementia Rating Sum of Boxes |
| `MMSCORE`  | Mini-Mental State Examination         |

### Why these features?

The feature set combines:

* Demographic information
* Genetic information
* Cognitive assessment
* Clinical dementia severity

This provides a compact tabular representation suitable for comparing different ML model families.

---

# 🧹 Data Preprocessing

The preprocessing pipeline separates numerical and categorical variables.

### Numerical features

The following preprocessing was applied:

1. Missing values → **median imputation**
2. Features → **standardization**

### Categorical features

The following preprocessing was applied:

1. Missing values → **most-frequent imputation**
2. Categories → **one-hot encoding**

After preprocessing, the nine original feature groups resulted in **22 encoded variables**.

### APOE Handling

APOE genotype was treated as a **categorical variable**.

It was not ordinal encoded because genotypes do not naturally form a simple numerical ordering.

An APOE-E4 carrier variable was also prepared for secondary analysis but was not combined with APOE genotype in the primary model.

---

# 🤖 Machine Learning Models

## 1. XGBoost

**XGBoost** is a gradient-boosted decision-tree algorithm.

It builds trees sequentially, with later trees attempting to correct errors from earlier trees.

Main configuration:

```text
n_estimators = 300
max_depth = 3
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

Class imbalance was addressed using `scale_pos_weight`.

---

## 2. Random Forest

Random Forest is an ensemble learning method that combines predictions from multiple decision trees.

Main configuration:

```text
n_estimators = 300
min_samples_split = 5
min_samples_leaf = 2
class_weight = "balanced"
```

---

## 3. Explainable Boosting Machine

The **Explainable Boosting Machine (EBM)** is an interpretable boosting model designed to provide transparent feature-level effects.

Main configuration:

```text
interactions = 0
```

EBM was included because it provides a different modeling approach from the tree ensembles while also offering native interpretability.

---

# 📊 Model Evaluation

An initial stratified **80/20 train-test split** was used for baseline evaluation.

The main analysis used:

> **5-fold stratified cross-validation**

The following metrics were evaluated:

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

Because the dataset is imbalanced, **ROC-AUC and PR-AUC**, together with class-sensitive metrics, are particularly important.

---

# 📈 Predictive Performance

Five-fold cross-validation produced the following average results:

| Model             |      Accuracy | Balanced Accuracy |            F1 |       ROC-AUC |        PR-AUC |
| ----------------- | ------------: | ----------------: | ------------: | ------------: | ------------: |
| **XGBoost**       | 0.754 ± 0.019 |     0.606 ± 0.033 | 0.285 ± 0.039 | 0.684 ± 0.038 | 0.209 ± 0.035 |
| **Random Forest** | 0.763 ± 0.021 |     0.637 ± 0.042 | 0.322 ± 0.050 | 0.733 ± 0.039 | 0.256 ± 0.055 |
| **EBM**           | 0.881 ± 0.002 |     0.500 ± 0.000 | 0.000 ± 0.000 | 0.741 ± 0.056 | 0.291 ± 0.078 |

### Important interpretation

EBM's default classification threshold of `0.5` resulted in essentially all-negative class predictions, which produced:

```text
Recall = 0
F1 = 0
```

However, EBM still produced meaningful probability ranking information, reflected in its ROC-AUC and PR-AUC values.

Therefore:

> **Accuracy alone should not be used to interpret the EBM result.**

A future analysis should perform threshold selection within the training data or use a pre-specified thresholding strategy.

---

# 🔍 Explainable AI

Explainable AI was the central part of this project.

## SHAP

**SHAP (SHapley Additive exPlanations)** was used for:

* XGBoost
* Random Forest

SHAP estimates the contribution of each feature toward an individual model prediction.

For example:

```text
Feature → Contribution toward prediction
```

The absolute SHAP values were aggregated to estimate global feature importance.

---

# 🧠 EBM Explainability

EBM provides its own feature-level explanation mechanism.

Instead of applying SHAP to EBM, the model's native global explanation was used.

This gives the project three explanation sources:

```text
XGBoost      → SHAP
Random Forest → SHAP
EBM           → Native EBM explanation
```

The resulting feature contributions were grouped back from one-hot encoded variables into the original feature groups.

---

# 🔗 Cross-Model Attribution Consistency

The key research question is whether different models identify similar important features.

Two complementary approaches were used.

## 1. Spearman Rank Correlation

Spearman correlation compares the **ranking** of feature importance between models.

This was chosen because raw attribution magnitudes are not necessarily directly comparable between different model families.

### Results

| Model Pair              | Spearman ρ |
| ----------------------- | ---------: |
| XGBoost ↔ Random Forest |  **0.950** |
| XGBoost ↔ EBM           |  **0.867** |
| Random Forest ↔ EBM     |  **0.917** |

These values indicate strong agreement in feature rankings within this experiment.

---

# 🎯 Top-K Feature Consistency

The project also compared the sets of the most important features using **Jaccard similarity**.

Jaccard similarity measures the overlap between two sets.

A value of:

```text
1.0 = identical feature sets
0.0 = no overlap
```

### Top-3

All three models identified:

1. `AGE`
2. `CDRSB`
3. `MMSCORE`

as their common top three feature groups.

**Top-3 Jaccard similarity = 1.0**

### Top-5

The common top five feature groups were:

1. `AGE`
2. `APOE`
3. `CDRSB`
4. `MMSCORE`
5. `PTEDUCAT`

**Top-5 Jaccard similarity = 1.0**

---

# 🔄 Attribution Stability

Cross-model agreement is only one part of explanation reliability.

The project also asked:

> **Do the feature-attribution rankings remain similar when the training data changes?**

To answer this, the models were evaluated using **five-fold stratified cross-validation**.

Feature attribution was calculated independently for each fold.

The resulting rankings were then compared across folds.

### Within-Fold Rank Stability

| Model         | Rank Stability |
| ------------- | -------------: |
| XGBoost       |      **0.942** |
| Random Forest |      **0.980** |
| EBM           |      **0.888** |

The results indicate that the broad feature-attribution patterns were relatively stable across folds in this dataset.

---

# ⭐ Main Findings

The most important findings of the project are:

### 1. Strong cross-model agreement

The three models showed strong feature-ranking agreement.

```text
Spearman ρ = 0.867 – 0.950
```

### 2. Identical top-feature sets

All three models shared:

```text
Top-3:
AGE
CDRSB
MMSCORE
```

and:

```text
Top-5:
AGE
APOE
CDRSB
MMSCORE
PTEDUCAT
```

### 3. Stable explanations

Feature-attribution rankings remained relatively stable across five-fold cross-validation.

### 4. Explanation is not the same as prediction

The project demonstrates why predictive performance and explanation reliability should be considered separately.

### 5. XAI does not establish causality

A feature receiving a high attribution means that the model relies on that feature.

It does **not** prove that the feature causes Alzheimer’s progression.

---

# 🧠 Key Research Insight

The central insight can be summarized as:

> **Different machine-learning models trained on the same ADNI-derived MCI cohort produced substantially overlapping feature-attribution rankings, and these broad attribution patterns were relatively stable across cross-validation folds.**

This suggests that the major explanatory patterns observed in this experiment were not completely dependent on one particular model architecture or one particular train/validation split.

However, these findings are limited to the dataset and experimental design used in this study.

---

# 🗂️ Project Structure

```text
MCI_AD_XAI_Project/
│
├── data/
│   ├── final_cohort.csv
│   ├── primary_features.csv
│   ├── mri_enhanced_features.csv
│   └── final_modeling_dataset.csv
│
├── notebooks/
│   ├── 01_Cohort_Definition.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 04_XAI_Analysis.ipynb
│   └── 05_Results_Figures.ipynb
│
├── models/
│
├── results/
│   ├── 5fold_performance_summary.csv
│   ├── xai_model_stability_summary.csv
│   ├── three_model_pairwise_consistency.csv
│   ├── three_model_top_k_consistency.csv
│   ├── final_model_comparison.csv
│   ├── feature_ranks_for_paper.csv
│   └── *.png
│
├── .gitignore
└── README.md
```

> **Note:** Raw/restricted ADNI data should remain outside the public repository.

---

# 🛠️ Technologies Used

### Programming

* Python
* Jupyter Notebook
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* InterpretML / EBM

### Explainable AI

* SHAP
* EBM native explanations

### Visualization

* Matplotlib
* Seaborn

### Data

* ADNI
* Clinical and demographic variables
* APOE genotype
* Cognitive assessments
* MRI-derived hippocampal volumes for sensitivity analysis

---

# 📓 Notebooks

## `01_Cohort_Definition.ipynb`

Responsible for:

* Loading diagnostic data
* Identifying MCI participants
* Defining first MCI baseline
* Defining two-year conversion outcome
* Building the modeling cohort
* Constructing the primary feature dataset
* Preparing MRI sensitivity features

---

## `02_Model_Training.ipynb`

Responsible for:

* Loading the final modeling dataset
* Splitting data
* Preprocessing
* Training XGBoost
* Training Random Forest
* Training EBM
* Evaluating model performance

---

## `04_XAI_Analysis.ipynb`

Responsible for:

* SHAP analysis
* EBM explanations
* Feature grouping
* Cross-model consistency
* Spearman correlation
* Top-k Jaccard similarity
* Five-fold attribution stability

---

## `05_Results_Figures.ipynb`

Responsible for:

* Final performance figures
* Attribution figures
* Consistency figures
* Stability figures
* Paper-ready result tables

---

# 📊 Main Outputs

The analysis produces:

```text
Model performance
        ↓
Feature attribution
        ↓
Cross-model consistency
        ↓
Top-k agreement
        ↓
Cross-validation stability
        ↓
Research conclusions
```

Important result files include:

```text
results/5fold_performance_summary.csv
results/xai_model_stability_summary.csv
results/three_model_pairwise_consistency.csv
results/three_model_top_k_consistency.csv
results/final_model_comparison.csv
results/feature_ranks_for_paper.csv
```

---

# ⚠️ Limitations

This project should be interpreted as a research analysis rather than a clinically deployable system.

### Dataset limitations

* ADNI is a research cohort and may not represent routine clinical populations.
* External validation was not performed.
* The number of converters was relatively small.
* Several variables had substantial missingness.

### Modeling limitations

* Hyperparameter optimization was not the main focus.
* Classification threshold optimization was not performed for EBM.
* MRI was not included as a mandatory primary feature.

### Explainability limitations

* SHAP explains model behavior rather than biological causality.
* Different explanation methods have different assumptions.
* Strong agreement in one dataset does not guarantee agreement in another dataset.

---

# 🚀 Future Work

Potential extensions include:

* External validation on independent datasets
* Nested cross-validation
* Hyperparameter optimization
* Probability calibration
* Decision-curve analysis
* Optimized classification thresholds
* Bootstrap confidence intervals
* MRI-enhanced models
* Additional biomarkers
* Subgroup robustness analysis
* Explanation perturbation testing
* Prospective clinical evaluation

---

# 📚 References

1. **Alzheimer’s Disease Neuroimaging Initiative (ADNI).** ADNI Diagnostic Documentation.
   https://adni.loni.usc.edu/

2. Yi F, Yang H, Chen D, et al. **XGBoost-SHAP-based interpretable diagnostic framework for Alzheimer’s disease.** *BMC Medical Informatics and Decision Making*. 2023;23:137.
   https://doi.org/10.1186/s12911-023-02238-9

3. Sarica A, Aracri F, Bianco MG, et al. **Explainability of random survival forests in predicting conversion risk from mild cognitive impairment to Alzheimer’s disease.** *Brain Informatics*. 2023;10:31.
   https://doi.org/10.1186/s40708-023-00211-w

4. Guillén P, Frias-Martinez E. **Enhancing SHAP Explainability for Diagnostic and Prognostic ML Models in Alzheimer’s Disease.** *Computers, Materials & Continua*. 2026;87(2):95.
   https://doi.org/10.32604/cmc.2026.076400

5. Lundberg SM, Lee SI. **A Unified Approach to Interpreting Model Predictions.** *Advances in Neural Information Processing Systems*. 2017;30:4765–4774.

6. Chen T, Guestrin C. **XGBoost: A Scalable Tree Boosting System.** *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016:785–794.
   https://doi.org/10.1145/2939672.2939785

7. Breiman L. **Random Forests.** *Machine Learning*. 2001;45:5–32.
   https://doi.org/10.1023/A:1010933404324

---

# 👨‍💻 Author

**MCI-to-AD XAI Research Project**

Developed as an academic research project in machine learning, healthcare AI, and explainable artificial intelligence.

---

# ⚖️ Disclaimer

This repository is intended for **research and educational purposes only**.

The models and explanations presented here are not medical advice and should not be used for diagnosis, treatment decisions, or clinical decision-making.

The results should not be interpreted as evidence of causal relationships or clinical readiness.

---

## ⭐ Project in One Sentence

> **An ADNI-based machine-learning study that predicts two-year MCI-to-Alzheimer’s conversion while systematically evaluating whether XGBoost, Random Forest, and Explainable Boosting Machine produce consistent and stable feature explanations.**
