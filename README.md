# 🧠 Fake Job Posting Detection Using Machine Learning & Deep Learning

## 🚀 Overview
This project focuses on detecting fraudulent job postings using **Natural Language Processing (NLP)** and **Machine Learning/Deep Learning** models.

The goal is to build a system that can automatically identify fake job listings based on their descriptions, requirements, and metadata. It uses both traditional ML algorithms and deep neural architectures to find the best-performing model in terms of accuracy, F1-score, and ROC-AUC.

---

## 📊 Dataset

- **Name:** Fake Job Postings Dataset
- **Source:** [Kaggle — Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Shape:** 17,880 rows × 18 columns
- **Target Variable:** `fraudulent`
  - `0` → Real Job
  - `1` → Fake Job
- **Class Distribution:** 
  - Real Jobs: 95.2% (17,014 posts)
  - Fake Jobs: 4.8% (866 posts)

### Key Features Include:
- `title` - Job title
- `location` - Job location
- `company_profile` - Company description
- `description` - Detailed job description
- `requirements` - Job requirements
- `benefits` - Benefits offered
- `employment_type`, `required_experience`, `required_education`, `industry`, `function`, etc.

---

## 🧹 Data Preprocessing

1. **Handled missing values** - Filled missing text fields with "Not Provided"
2. **Combined multiple textual columns** (`title`, `description`, `requirements`, `company_profile`) for richer context
3. **Text vectorization:**
   - **TF-IDF Vectorization** for ML models (5000 features, uni+bigrams)
   - **Tokenization & Padding** for DL models (MAX_WORDS=10,000, MAX_LEN=200)
4. **Applied SMOTE** to balance the dataset for ML models (handle class imbalance)
5. **Class weights** for DL models to penalize minority class errors

---

## 🧮 Models Implemented

### 🧩 Machine Learning Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Simple linear baseline model |
| **Naive Bayes** | Probabilistic classifier suitable for text data |
| **Linear SVC** | High-dimensional linear classifier |
| **Random Forest** | Ensemble-based decision trees |
| **Gradient Boosting** | Boosted ensemble improving accuracy |

### 🤖 Deep Learning Models

| Model | Architecture |
|-------|--------------|
| **LSTM** | Sequential model using word embeddings and LSTM layers |
| **Bidirectional LSTM** | Bidirectional LSTM for contextual understanding |
| **CNN** | 1D convolutional network capturing local text features |

---

## ⚙️ Training & Evaluation

- Each model was trained on the same preprocessed dataset
- Train-Test Split: **80-20** with stratification
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score (Primary metric for imbalanced data)
  - ROC-AUC

---

## 📈 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 97.60% | 70.05% | 87.86% | 0.7795 | 0.9298 |
| **Naive Bayes** | 91.53% | 70.67% | 37.57% | 0.5152 | 0.9226 |
| **Linear SVC** | **98.41%** | **83.33%** | **83.82%** | **0.8357** | **0.9148** |
| **Random Forest** | 98.18% | 95.76% | 65.32% | 0.7766 | 0.8259 |
| **Gradient Boosting** | 96.00% | 77.81% | 53.18% | 0.6521 | 0.8720 |
| **LSTM** | 95.73% | 65.11% | 65.32% | 0.6521 | 0.8753 |
| **Bidirectional LSTM** | 96.62% | 62.87% | 73.41% | 0.6773 | 0.9464 |
| **CNN** | 97.48% | 77.48% | 67.63% | 0.7222 | 0.9693 |

---

## 📊 Visualizations

The notebook includes comprehensive visualizations:

- 📊 **Class Distribution** - Bar and pie charts showing the imbalance
- 📈 **Performance Metrics** - Comparative bar plots for all models
- 🎯 **ROC Curves** - Model discrimination ability comparison across all 8 models
- 🔢 **Confusion Matrices** - Error analysis for each model (8 heatmaps)
- 📉 **Training History** - Loss and accuracy curves for DL models (LSTM, BiLSTM, CNN)

---

## 🏆 Conclusion

### Best Performing Model: **Linear SVC**

Among all tested models, **Linear SVC** achieved the highest performance with:
- **F1-Score:** 0.8357 (Best balance between precision and recall)
- **Accuracy:** 98.41%
- **ROC-AUC:** 0.9148

#### Why Linear SVC Won?
- Best balance between **precision (83.33%)** and **recall (83.82%)**
- Successfully detects **84% of fraud cases** while minimizing false alarms
- Performs exceptionally well on high-dimensional TF-IDF features

#### Key Insights:
1. **Traditional ML models** (especially Linear SVC and Random Forest) performed very well on this dataset
2. **Deep Learning models** showed competitive but slightly lower F1-scores in this implementation
3. **Class imbalance handling** (SMOTE + class weights) was critical for performance
4. **F1-Score is more important than accuracy** for this highly imbalanced problem

---

## 🧩 Tech Stack

- **Language:** Python 3.8+
- **Libraries:**
  - **Data Handling:** `pandas`, `numpy`
  - **Visualization:** `matplotlib`, `seaborn`
  - **ML Models:** `scikit-learn`, `imblearn` (SMOTE)
  - **DL Models:** `TensorFlow 2.19.0`, `Keras`
  - **Text Processing:** `TfidfVectorizer`, `Tokenizer`

---


---

## 🌟 Future Enhancements

-  Implement **Transformer models** (BERT, RoBERTa, DistilBERT) for better contextual understanding
-  Add **Explainability** using SHAP or LIME to interpret model predictions
-  Build a **Flask/Streamlit web app** for live job post fraud detection
