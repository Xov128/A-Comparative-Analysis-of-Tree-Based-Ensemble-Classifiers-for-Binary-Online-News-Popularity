# 📰 Binary Online News Popularity Prediction

> A comparative analysis of tree-based ensemble classifiers for predicting whether a Mashable article will go viral — before it's published.

**Authors:** Cornelius Christian Setiadi · Josua Edward Budianto · Jovan Melsyah  
**Institution:** Computer Science & Statistics, Bina Nusantara University

---

## Overview

With thousands of articles published daily, only a fraction achieve significant social media engagement. This project builds a binary classification pipeline to predict whether an online news article will be **Popular** or **Not Popular** based on its content and metadata — before publication.

Three tree-based ensemble classifiers are trained, tuned, and compared:

- **Random Forest** — bagging-based ensemble
- **XGBoost** — sequential gradient boosting
- **LightGBM** — histogram-based gradient boosting *(best performer: 67.27% accuracy)*

---

## Dataset

**Source:** [UCI Machine Learning Repository — Online News Popularity](https://archive.ics.uci.edu/ml/datasets/online+news+popularity)  
**File:** `OnlineNewsPopularity.csv`  
**Articles:** 39,644 from Mashable (Jan 2013 – Jan 2015)  
**Features:** 60 predictive attributes + 1 target variable (`shares`)

### Target construction

The raw `shares` column is transformed into a binary label using the **median (1,400 shares)** as threshold:

| Label | Class | Samples |
|---|---|---|
| 0 | Not Popular (shares < 1,400) | 18,490 (46.6%) |
| 1 | Popular (shares ≥ 1,400) | 21,154 (53.4%) |

The near-balanced distribution eliminates the need for resampling techniques like SMOTE.

### Feature groups

| Group | Examples |
|---|---|
| Text content | `n_tokens_title`, `n_tokens_content`, `n_unique_tokens` |
| Multimedia | `num_imgs`, `num_videos`, `num_hrefs` |
| Keyword stats | `kw_avg_avg`, `kw_max_max`, `kw_min_avg` |
| Data channel | `data_channel_is_tech`, `_lifestyle`, `_world`, etc. |
| Publication day | `weekday_is_monday` … `weekday_is_sunday` |
| LDA topics | `LDA_00` … `LDA_04` |
| Sentiment/NLP | `global_sentiment_polarity`, `avg_positive_polarity` |

---

## Methodology

### Pipeline

```
Dataset
  ↓
Feature Decoding (one-hot → categorical)
  ↓
EDA (correlation, distributions, topic analysis)
  ↓
Train-Test Split 80:20 (stratified)
  ↓
Label Encoding (fit on train only — no leakage)
  ↓
Baseline Modeling (RF · XGBoost · LightGBM)
  ↓
Hyperparameter Tuning (RandomizedSearchCV, 3-fold CV, 30 iterations)
  ↓
Evaluation + Feature Importance Analysis
```

### Feature decoding

One-hot encoded columns were collapsed into interpretable categoricals:

- `data_channel_is_*` → `channel` (Lifestyle, Tech, World, etc.)
- `weekday_is_*` → `weekday`
- `LDA_00–04` → `dominant_topic` (via `idxmax()`)

### Models & baseline config

| Model | Key Parameters |
|---|---|
| Random Forest | `n_estimators=200`, `max_depth=15`, `min_samples_leaf=5` |
| XGBoost | `n_estimators=500`, `lr=0.05`, `max_depth=6`, `subsample=0.8` |
| LightGBM | Default binary classification objective |

### Tuning

`RandomizedSearchCV` with 3-fold cross-validation, 30 iterations per model, scored by accuracy. Split-before-encode enforced throughout to prevent data leakage.

---

## Results

### Model accuracy comparison

| Model | Baseline | Tuned |
|---|---|---|
| Random Forest | 65.59% | 65.68% |
| XGBoost | 67.16% | 67.23% |
| **LightGBM** | 66.53% | **67.27% 🏆** |

### Best model — LightGBM (tuned)

| Metric | Value |
|---|---|
| Accuracy | **67.27%** |
| Precision | 0.68 |
| Recall | 0.72 |
| F1-Score | 0.70 |

Boosting-based models (XGBoost, LightGBM) consistently outperform the bagging-based approach (Random Forest), confirming that sequential error-correction is better suited to the non-linear feature interactions in this dataset.

### Top predictors (consistent across all 3 models)

1. `kw_avg_avg` — average keyword shareability
2. `dominant_topic` — latent LDA topic
3. `channel` — content category
4. `self_reference_min_shares` — author's historical shareability
5. `timedelta` — days since publication

---

## Getting Started

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### Running the notebook

1. Place `OnlineNewsPopularity.csv` in the working directory.
2. Open `Code.ipynb` in Jupyter or VS Code.
3. Run all cells. The notebook covers:
   - Data loading and missing value inspection
   - Feature decoding and target construction
   - EDA visualizations
   - Train-test split + label encoding
   - Baseline and tuned model training
   - Confusion matrix + classification report
   - Feature importance plots

---

## Project Structure

```
.
├── Code.ipynb                      # Full analysis and modeling notebook
├── OnlineNewsPopularity.csv        # Dataset (UCI)
├── PaperMachineLearning.pdf        # Full research paper
├── Poster.png                      # Research poster
└── README.md                       # Project documentation
```

---

## Key Findings

- **Keyword metrics** (`kw_avg_avg`) are the strongest single predictor of article virality across all three models.
- **Tech and Lifestyle** channels produce a disproportionately high share of popular articles; **World and Entertainment** channels skew toward not popular.
- **Image count** positively correlates with shareability; **video count** shows no meaningful discriminating power.
- **Article length and sentiment polarity** are weak standalone predictors — content quality and topic matter more than quantity of text.
- **Boosting > Bagging** for this task: XGBoost and LightGBM both consistently beat Random Forest, and hyperparameter tuning provides the largest gain for LightGBM (+0.74 pp).

---

## Future Work

- Apply SHAP values for more interpretable feature attribution
- Explore deep learning (BERT-based) on article text beyond metadata
- Test time-aware train-test splits to better reflect real publication scenarios
- Incorporate social graph features (author followers, platform reach)

---

## References

Key references used in this study:

- Fernandes et al. (2015) — Online News Popularity Dataset, UCI ML Repository
- Chen & Guestrin (2016) — XGBoost
- Ke et al. (2017) — LightGBM
- Breiman (2001) — Random Forests
- Bergstra & Bengio (2012) — RandomizedSearchCV

---

## License

This project is for academic purposes only.
