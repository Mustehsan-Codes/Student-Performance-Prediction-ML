# 🎓 Student Performance Prediction (ML Project)

This machine learning project aims to predict student performance levels — such as Fail, Pass, Good, and Excellent — based on demographic and academic features. The model helps identify patterns and potential performance gaps in students, enabling early intervention and academic planning.

---

## 📁 Dataset

- **Source:** [Students Performance in Exams Dataset - Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Features:**
  - Gender
  - Race/Ethnicity
  - Parental level of education
  - Lunch type
  - Test preparation course
  - Math, Reading, and Writing scores

---

## 🔍 Project Overview

- **Goal:** Predict student performance levels using ML classification.
- **ML Model Used:** K-Nearest Neighbors (KNN)
- **Platform:** Google Colab (Python)
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn

---

## 🛠️ Steps Performed

1. **Data Preprocessing**
   - Handled missing values
   - Encoded categorical features
   - Normalized numerical data
   - Engineered labels into 4 performance categories:
     - `Fail` (below 40)
     - `Pass` (40–60)
     - `Good` (60–80)
     - `Excellent` (above 80)

2. **Model Implementation**
   - Built a custom **K-Nearest Neighbors (KNN)** model from scratch
   - Splitted data into training and testing sets
   - Tuned the number of neighbors (K value)

3. **Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report (Precision, Recall, F1-score)

4. **Visualization**
   - Score distributions
   - Class predictions vs. actual values
   - Heatmaps and bar plots

---

## 📊 Results

- Achieved good classification accuracy on test data
- Effective at identifying students at risk of failure
- Visual insights helped in understanding feature impact

---

## 📎 Project Structure

