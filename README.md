## Fake Job Posting Detection Using ML & Deep Learning

##  Problem Description

Online recruitment fraud has become a significant threat, with scammers creating fake job postings to steal personal information and money from job seekers. This project develops an automated system to detect fraudulent job postings using Natural Language Processing and machine learning. We implement and compare **8 models** (5 ML + 3 DL) to find the most effective approach for identifying fraud patterns in job descriptions and metadata.

**Why It Matters:**
- Many of the online job postings are fraudulent  
- Manual review is impractical at scale (millions of postings daily)  
- Victims lose money, personal data, and trust in legitimate platforms  

**Key Results:** Linear SVC achieved the best performance with **F1-Score: 0.8357** and **ROC-AUC: 0.9148**, successfully detecting 84% of fraudulent posts while maintaining 83% precision.

---

## Dataset

**Source:** [Kaggle - Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

**Statistics:**
- **Size:** 17,880 job postings × 18 features  
- **Target:** `fraudulent` (0=Real, 1=Fake)  
- **Class Distribution:** 95.2% Real (17,014) | 4.8% Fake (866) → **Severe imbalance (19.7:1)**  

**Key Features:**
- **Textual:** `title`, `company_profile`, `description`, `requirements`, `benefits` (primary signals)  
- **Categorical:** `employment_type`, `experience`, `education`, `industry`, `function`  
- **Binary:** `telecommuting`, `has_company_logo`, `has_questions`  

**Preprocessing:**
1. **Missing Values:** Filled text fields with "Not Provided" (40% missing in `company_profile`, 60% in `benefits`)  
2. **Text Combination:** Merged `title + description + requirements + company_profile` into `combined_text` for richer context  
3. **Feature Engineering:**  
   - **ML:** TF-IDF vectorization (5000 features, uni+bigrams, min_df=2)  
   - **DL:** Tokenization (vocab=10,000, max_len=200, padding)  
4. **Class Rebalancing:**  
   - **ML:** SMOTE oversampling to 1:1 ratio (27,222 balanced samples)  
   - **DL:** Class weights {0: 1.0, 1: 9.89} to penalize minority errors  
5. **Train-Test Split:** 80-20 stratified (14,304 train | 3,576 test)  

**Data Challenges:** Severe imbalance, high dimensionality (5000+ TF-IDF features), noisy text, subtle fraud indicators [web:81][web:83]

---

## Methods

### Approach

We adopt a **comparative experimental design** evaluating traditional ML vs. deep learning to determine which approach better handles text-based fraud detection under severe class imbalance.

**Why This Approach?**
- **ML Models:** Proven effectiveness on text with TF-IDF, fast training, interpretable, work well on moderate datasets [web:80]  
- **DL Models:** Automatically learn representations, capture sequential dependencies, state-of-the-art on many NLP tasks [web:82]  
- **Comparison Goal:** Assess whether DL's complexity provides advantages over simpler ML + feature engineering  

### Models Implemented

| Category | Model | Architecture/Algorithm |
|----------|-------|------------------------|
| **ML (5)** | Logistic Regression | Linear classifier, L2 regularization |
| | Naive Bayes | Multinomial probabilistic classifier |
| | **Linear SVC** ⭐ | Linear SVM with hinge loss |
| | Random Forest | Ensemble of 100 decision trees |
| | Gradient Boosting | Sequential boosting of weak learners |
| **DL (3)** | LSTM | Embedding(128) → LSTM(64) → LSTM(32) → Dense |
| | Bidirectional LSTM | Embedding(128) → BiLSTM(64) → BiLSTM(32) → Dense |
| | CNN (1D) | Embedding(128) → Conv1D(128) → Conv1D(64) → Dense |

**Training Details:**
- **ML:** Scikit-learn, trained on SMOTE-balanced TF-IDF features  
- **DL:** TensorFlow/Keras, Adam optimizer, early stopping (patience=3), dropout (0.3–0.5)  
- **Evaluation:** F1-Score (primary), ROC-AUC, Precision, Recall, Confusion Matrices  

**Why F1-Score?** Accuracy is misleading on 95:5 imbalanced data (predicting all "real" = 95% accuracy but 0% fraud detection). F1 balances precision (minimize false alarms) and recall (catch frauds) [web:26][web:83].

---

## Experiments & Results

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Linear SVC** ⭐ | **98.41%** | **83.33%** | **83.82%** | **0.8357** | 0.9148 |
| Logistic Regression | 97.60% | 70.05% | 87.86% | 0.7795 | 0.9298 |
| Random Forest | 98.18% | 95.76% | 65.32% | 0.7766 | 0.8259 |
| CNN | 97.48% | 77.48% | 67.63% | 0.7222 | 0.9693 |
| BiLSTM | 96.62% | 62.87% | 73.41% | 0.6773 | 0.9464 |
| LSTM | 95.73% | 65.11% | 65.32% | 0.6521 | 0.8753 |
| Gradient Boosting | 96.00% | 77.81% | 53.18% | 0.6521 | 0.8720 |
| Naive Bayes | 91.53% | 70.67% | 37.57% | 0.5152 | 0.9226 |

### Key Findings

 **Linear SVC Best Overall:** Achieves optimal precision-recall balance (83.3% precision, 83.8% recall)  
 **ML Outperforms DL:** Traditional ML + TF-IDF beats neural networks on this moderate-sized, imbalanced dataset  
 **Random Forest: Highest Precision:** 95.76% precision but lower recall (65.32%) → good for minimizing false alarms  
 **Logistic Regression: Highest Recall:** 87.86% recall but lower precision → catches more frauds, more false alarms  
 **DL Models Underperform:** Limited training data (14k samples) + extreme imbalance hinders neural network advantage  

### Visualizations

The notebook includes:
-  Class distribution (bar/pie charts)  
-  Model comparison (F1, Precision, Recall bar plots)  
-  ROC curves (all 8 models overlaid)  
-  Confusion matrices (8 heatmaps showing TP/FP/TN/FN)  
-  DL training history (loss/accuracy over epochs)  

---
##  How to Run

### Prerequisites
- Python 3.8+  
- pip or conda package manager  

---

### Setup & Execution

1. **Clone Repository**
   ```bash
   git clone https://github.com/deepakkk05/FAKE_JOB_POSTING_PREDICTION.git
   cd FAKE_JOB_POSTING_PREDICTION
2. **Install Dependencies**

```bash

pip install -r requirements.txt
requirements.txt:
```
**requirements.txt**
```shell

numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
tensorflow>=2.19.0
jupyter>=1.0.0
Run Notebook
```

3. **Run Notebook**
```bash
jupyter notebook fake_job_posting.ipynb
```
4. **Expected Output:**
All cells execute sequentially, training 8 models and generating comparison visualizations (~10–15 min on CPU).

---

##  Conclusion

### What We Learned

1. **Traditional ML Wins:** Linear SVC with TF-IDF features outperforms complex DL architectures (F1: 0.8357 vs. 0.6773 for BiLSTM) on this moderately-sized, highly imbalanced text classification task.  
2. **Feature Engineering Matters:** Manual TF-IDF with uni+bigrams captures fraud-indicative phrases (e.g., "no experience required", "work from home") more effectively than learned embeddings with limited training data.  
3. **Class Imbalance is Critical:** SMOTE for ML and class weighting for DL significantly improve minority class detection. Without rebalancing, models achieve 95%+ accuracy by predicting all "real" but 0% fraud detection.  
4. **Precision-Recall Tradeoff:**  
   - **High Precision (Random Forest):** Fewer false alarms but misses 35% of frauds  
   - **High Recall (Logistic Regression):** Catches 88% of frauds but 30% false alarm rate  
   - **Balanced (Linear SVC):** Best F1-score with 83% precision and 84% recall  
5. **Deep Learning Underperforms:** With only 14k training samples and 19:1 imbalance, neural networks fail to leverage their representation learning advantage. They require more data or advanced techniques (pre-trained transformers).  

### Real-World Impact

Our Linear SVC model can:
- Screen 1,000 job postings in <1 second  
- Detect 84% of frauds with 83% precision (only 17% false positives)  
- Be deployed in production job platforms to flag suspicious postings for review  
- Save manual reviewers time by prioritizing high-risk postings  

---

##  Future Work

- [ ] **Transformers (BERT/RoBERTa):** Pre-trained language models may improve DL performance [web:40]  
- [ ] **Explainability (SHAP/LIME):** Identify which keywords/phrases trigger fraud predictions [web:47]  
- [ ] **Ensemble Methods:** Combine Linear SVC + Random Forest for robustness  
- [ ] **Web Application:** Flask/Streamlit app for real-time fraud detection  
- [ ] **Additional Features:** Extract URLs, email patterns, phone numbers as metadata features  

---

##  References

1. [Kaggle Dataset - Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
2. Vidros, S., et al. (2023). "Detecting fake job postings using machine learning." *IEEE Conference on Artificial Intelligence*.  
3. Chawla, N.V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*, 16, 321–357.  
4. Zhang, Y., & Wallace, B. (2017). "A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification." *IJCNLP*.  
