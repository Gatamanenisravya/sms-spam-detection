# SMS Spam Detection App

An end-to-end machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and a deployed web application.

---

## Features

- Text classification using TF-IDF + Logistic Regression
- Real-time prediction via Streamlit web app
- Adjustable spam threshold
- Model training + evaluation pipeline

---

## Machine Learning Workflow

1. Data Cleaning & Preprocessing
2. Text Vectorization (TF-IDF)
3. Model Training (Logistic Regression)
4. Model Evaluation (Precision, Recall, F1-score)
5. Model Deployment using Streamlit

---

## Model Performance

- Accuracy: ~98%
- Spam Precision: ~0.94
- Spam Recall: ~0.93
- F1 Score: ~0.94

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## Project Structure
SMS_spam_detection/
data/
models/
 sms_spam_model.joblib
src/
 train.py
 predict.py
app.py
requirements.txt
README.md

---
## How to Run

### 1. Create environment
```bash
conda create -n sms_spam python=3.11 -y
conda activate sms_spam

```
---

### 2. Install dependencies

```bash
pip install -r requirements.txt

```
---

### 3. Train model

```bash
python src/train.py

```
---

### 4. Run app

```bash
streamlit run app.py

```
---

## Example

---

### Input
Congratulations! You have won a free prize.

---

### Output
Prediction: SPAM
Spam Probability: 0.98

---

## Key Learnings

Text preprocessing and vectorization
Classification using Logistic Regression
Handling probability thresholds
Building and deploying ML apps
Debugging environment issues

---

## Author


---

## Expected result

Your README will now look:
- professional
- complete
- interview-ready
- GitHub-ready

---

# What I want you to do now

Do ONLY these:

1. Delete `.vscode` folder  
2. Run:
```bash
pip freeze > requirements.txt
```

---

## App Demo
![App Screenshot](appdemo.jpeg)
