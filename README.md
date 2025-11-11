# Lightweight Intrusion Detection on CSE-CIC-IDS2018  
## Random Forest vs. Ensemble Learning (RF + LR + SVM + DT)

This repository contains the implementation, and testing for a lightweight Intrusion Detection System (IDS) built using the **CSE-CIC-IDS2018** dataset.  
The main goal is to **compare a tuned Random Forest baseline** against a **soft-voting style ensemble** of:

- Random Forest (RF)
- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Decision Tree (DT)

under realistic, resource-aware conditions.

The project is structured like a mini research pipeline:
1. Data selection and preprocessing
2. Class imbalance and feature behavior analysis
3. Correlation and redundancy checks
4. Model training (RF baseline + Ensemble)
5. Evaluation on:
   - All attacks (Benign vs Any Attack)
   - Bot attack scenario (Benign vs Bot)

---

## 1. Key Ideas

- Use **all** attack types for the main testing.
- Apply a **transparent preprocessing pipeline**:
  - remove irrelevant identifiers,
  - handle missing and infinite values explicitly,
  - ensure numeric, clean, and stable features.
- Comparing a **strong classical baseline (Random Forest)** against a **heterogeneous ensemble**.

---

## 2. Repository Structure

Suggested structure (match this to your actual files):

```text
LIGHTWEIGHT_IDS_ASSIGNMENT/
├── notebooks/
│   ├── Data Preparation and Preprocessing/
│   │   ├── 1. Data Loading.ipynb
│   │   ├── 2. Data Cleaning.ipynb
│   │   ├── 3. Label Encoding.ipynb
│   │   ├── 4. Feature Encoding.ipynb
│   │   ├── 5. Feature Scaling.ipynb
│   │   ├── 6.1 Dataset Splitting_all_attacks.ipynb
│   │   └── 6.2 Dataset Splitting_Bot.ipynb
│   │
│   ├── Exploratory Data Analysis (EDA)/
│   │   ├── 1. Class Imbalance Check.ipynb
│   │   ├── 2. Feature Distribution Analysis.ipynb
│   │   └── 3. Correlation Analysis.ipynb
│   │
│   └── Model Development/
│       ├── RandomForestClassifier_allAttacks_after_research_20%subset.ipynb
│       ├── Ensemble_Learning_allAttacks_after_research_20%subset.ipynb
│       ├── RandomForestClassifier_after_research_Bot.ipynb
│       └── Ensemble_Learning_after_research_Bot.ipynb
│
├── results/
│   ├── Class Imbalance Check.png
│   ├── Confusion Matrix Ensemble Model All Attacks.png
│   ├── Confusion Matrix Random Forest Classifier All Attacks.png
│   ├── Correlation Analysis.png
│   └── Feature Distribution Analysis - histogram.png
│
├── requirements.txt
└── README.md

```

Each notebook is focused and documented as:

* **1.*** — Label distribution / imbalance.
* **2.*** — Feature distribution analysis.
* **3.*** — Correlation heatmap and redundancy check.
* **6.1 / 6.2** — Subset construction and splitting (all attacks / Bot).
* **RandomForest*** — Baseline model training + evaluation.
* **Ensemble*** — Ensemble training + evaluation (all attacks / Bot).

---

## 3. Dataset

**Dataset:** CSE-CIC-IDS2018
**Provider:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick.

You must download the raw dataset yourself from the official page:

```text
https://www.unb.ca/cic/datasets/ids-2018.html
```

Command to download the CSV Files required for the machine learning to train it on:

```sh
aws s3 sync --no-sign-request --region us-east-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" "./Processed_Traffic_Data" --exclude "*" --include "*.csv"
```

---

## 4. Preprocessing Pipeline

### 4.1 All Attacks in the dataset:

Implemented mainly in `6.1 Dataset Splitting_all_attacks.ipynb`:

* Load multiple CSV files in chunks.
* Normalize the `Label` column (strip spaces, unify names).
* Collect:
  * all rows with known/selected **attack labels** into an attack buffer,
  * a controlled number of **Benign** rows.
* Concatenate into a single DataFrame.
* Map:
  * `Benign → 0`
  * `Any attack → 1`  (binary classification: realistic, lightweight IDS)
* Save as `preprocessed_ids2018_subset.csv` for reuse.

### 4.2 Bot vs Benign Subset:

Implemented in `6.2 Dataset Splitting_Bot.ipynb`:

* Same idea, but the attack class is restricted to only the `Bot` attacks to evaluate on different attacks.
* Used to evaluate whether behavior changes for a **single attack type**.

### 4.3 Cleaning Steps

* Drop clearly irrelevant columns like the IDs and timestamp etc...
* Convert all feature columns to numeric.
* Replace:
  1. `+inf` with **max finite value** of that column,
  2. `-inf` with **min finite value**,
* Impute remaining `NaN` values with **median** per feature.
* Ensure the entire dataset is numeric and stable for:
  1. tree models (RF, DT),
  2. linear / kernel models (LR, SVM).

---

## 5. Exploratory Data Analysis:

### 5.1 Class Imbalance Check:

Notebook: `1. Class Imbalance Check.ipynb`

* Plots and counts for:

  1. `Benign` vs `Attack` (all types of attacks)
  2. Bot vs Benign (Bot attacks only)

* Purpose:

  1. avoid misleading “99% accuracy” on skewed data,
  2. confirm that the sampled subset has a more reasonable ratio.

### 5.2 Feature Distribution Analysis:

Notebook: `2. Feature Distribution Analysis.ipynb`

* Visualizes distributions (histograms/boxplots) of key features for:

  1. Benign vs Attack.

* Used to:

  1. confirm that several flow features are discriminative,
  2. spot heavy tails / skew suggesting the use of robust stats.

### 5.3 Correlation Analysis:

Notebook: `3. Correlation Analysis.ipynb`

* Computes correlation matrix on numeric features.
* Confirms:

  1. no extreme redundancy across the majority of the features.
  2. aligns with prior findings on CIC-IDS feature redundancy.

---

## 6. Train/Test Split

For each scenario:

* Use **80/20 split** with:

  1. `stratify=y` (preserve class ratios),
  2. `random_state=42` (reproducible).

* The **same split** is used for:

  1. Random Forest baseline,
  2. Ensemble model.

This ensures a **Fair Comparison**.

---

## 7. Models

### 7.1 Random Forest Baseline

Notebook: `RandomForestClassifier_allAttacks_after_research_20%Subset.ipynb`

Key settings:

* `RandomForestClassifier`

  1. `n_jobs = -1`
  2. `max_features = 'sqrt'`
  3. `bootstrap = True`

* Hyperparameter search:

  1. `n_estimators ∈ {50,100,200}`
  2. `max_depth ∈ {None,10,20}`
  3. `min_samples_split ∈ {2,5,10}`

* Search via `RandomizedSearchCV` (3-fold, scoring=`f1`).
* Train best RF on full training set.
* Evaluate on test set.

Same idea is mirrored for the **Bot vs Benign** RF notebook.

### 7.2 Ensemble Model (Soft Voting Style)

Notebook: `Ensemble_Learning_allAttacks_after_research_20%subset.ipynb`

Base learners:

* Random Forest (tuned / strong baseline)
* Logistic Regression (`max_iter` increased)
* SVM (RBF, `probability=True`)
* Decision Tree (bounded depth)

Pipeline:

* Train each base model on the training set.
* For evaluation:

  1. obtain predicted probabilities for the attack class from each model,
  2. **average them** (soft voting),
  3. apply threshold 0.5 for final prediction.

A similar setup is used in `Ensemble_Learning_after_research_Bot.ipynb` for the Bot scenario.

---

## 8. Results Summary

### 8.1 All Attacks (Benign vs Any Attack)

Using the 20% sampled, preprocessed all-attacks subset:

**Random Forest (Baseline)**

* Accuracy: **0.99994**
* Precision: **0.99994**
* Recall: **0.99992**
* F1-score: **0.99993**
* ROC-AUC: **0.99994**
* Confusion matrix:

  1. ![Alt text](results/Confusion%20Matrix%20Random%20Forest%20Classifier%20All%20Attacks.png)


**Ensemble (RF + LR + SVM + DT, soft voting)**

* Accuracy: **0.99981**
* Precision: **0.99994**
* Recall: **0.99961**
* F1-score: **0.99977**
* ROC-AUC: **0.999996**
* Confusion matrix:

  1. [[48410, 2],
    [14, 35497]]

**Interpretation:**

* Both models are extremely strong.
* Random Forest:

  1. fewer missed attacks,
  2. slightly better recall and F1,
  3. faster and simpler.

* Ensemble:

  1. slightly higher ROC-AUC (probability ranking),
  2. **but** worse discrete detection (more false negatives).

* Conclusion:

  1. In this setup, the **tuned RF baseline is strictly more practical**.

### 8.2 Bot vs Benign

Using the targeted Bot subset:

* Both Random Forest and the ensemble achieve **very high performance**.

* The Random Forest:

  1. matches or slightly outperforms the ensemble,
  2. remains cheaper to train and easier to deploy.

* No additional benefit from the more complex ensemble model.

(Exact metrics are shown inside the results folder.)

---

## 9. How to Run

### Google Colab (recommended)

1. Upload this repo (or notebooks) to your Drive.
2. Make sure you have:

   1. raw CSE-CIC-IDS2018 CSVs in Drive, **or**
   2. `preprocessed_ids2018_subset.csv` already generated.

3. In each notebook:

   1. mount Drive,
   2. update file paths,
   3. run all cells in order.


---

## 10. Conclusions:

* A carefully tuned **Random Forest** on a well-preprocessed subset of CSE-CIC-IDS2018:

  1. achieves **near-perfect detection**,
  2. outperforms or matches the ensemble,
  3. is more efficient and easier to deploy.

* The **heterogeneous soft-voting ensemble**:

  1. increases complexity,
  2. does **not** deliver a clear, consistent improvement in this lightweight IDS environment.

This supports the idea: **start with strong classical baselines + solid preprocessing before jumping into heavier ensembles or deep learning.**

---
