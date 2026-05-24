# Student Placement Prediction using Machine Learning

## Project Overview

This project predicts whether a student will be placed or not based on academic performance, internships, aptitude scores, soft skills, and placement-related activities.

The objective of this project is to build and evaluate a machine learning model capable of predicting placement outcomes with an accuracy greater than 60%.

---

## Problem Statement

Using a student placement dataset, predict whether a student will be:

- Placed
- Not Placed

using supervised machine learning algorithms.

---

## Project Objectives

- Perform Exploratory Data Analysis (EDA) to identify patterns and trends influencing placements.
- Preprocess the dataset for machine learning.
- Train and compare multiple supervised learning models.
- Evaluate model performance.
- Select the best-performing predictive model.

---

## Dataset Information

The dataset contains **10,000 student records** and includes features such as:

### Numerical Features
- CGPA
- Internships
- Projects
- Workshops/Certifications
- Aptitude Test Score
- Soft Skills Rating
- SSC Marks
- HSC Marks

### Categorical Features
- Extracurricular Activities
- Placement Training

### Target Variable
- Placement Status

---

## Exploratory Data Analysis (EDA)

Key findings from EDA:

- Students with higher **CGPA** generally had better placement outcomes.
- Higher **Aptitude Test Scores** positively influenced placement success.
- **Placement Training** improved employability.
- Students with more **Internships** demonstrated higher placement rates.
- **Soft Skills** moderately influenced placement outcomes.
- **Extracurricular Activities** showed relatively smaller influence.

---

## Machine Learning Models Used

The following supervised learning classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)

---

## Model Performance

| Model | Accuracy |
|--------|-----------|
| Logistic Regression | **79.45%** |
| Random Forest | 78.30% |
| KNN | 75.80% |
| Decision Tree | 72.65% |

### Best Model

🏆 **Logistic Regression** achieved the highest accuracy (**79.45%**) and was selected as the final model.

---

## Project Structure

```text
student-placement-predictor/
│
├── data/
│   └── raw/
│       └── placement.csv
│
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
├── requirements.txt
└── .gitignore