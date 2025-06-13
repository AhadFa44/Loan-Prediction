# 📊 Loan Prediction with Machine Learning

This project aims to automate and improve the loan approval process using machine learning. By analyzing applicant data, the system helps financial institutions predict whether an individual is eligible for a loan.

---

## 📘 Introduction

A loan is a debt between two parties: a financial institution and a borrower. Applying for a loan often involves complex steps. With the rise of financial activity and growing demand, banks now receive hundreds of loan applications daily, making automation essential.

---

## 💡 Motivation

- Banks and financial institutions receive many loan requests every day.
- Granting loans depends on the applicant’s repayment capability, work status, salary, and credit history.
- Traditionally, bank staff manually evaluate loan applications using set criteria.

This project helps reduce manual effort by automating the classification and evaluation of loan applications.

---

## 🎯 Project Objectives

This system assists banks in:

1. Classifying clients based on personal data and past repayment history.
2. Suggesting eligible clients for loans.
3. Saving time and effort in analyzing loan requests.
4. Automating data entry and decision-making for loan approvals.

---

## 🛠️ Proposed Solution

A machine learning-based system that:

- Predicts whether a customer is eligible for a loan.
- Trains a model on real-world loan data using Support Vector Machine (SVM).
- Automates the prediction and validation process with minimal human intervention.

---

## 🚀 Project Contribution

✅ Organized and prioritized customer loan requests.  
✅ Automated the analysis of applications based on personal and financial data.  
✅ Classified clients using loan history and demographic information.  
✅ Provided loan recommendations for bank decision-makers.  
✅ Made the loan application process more accessible for clients.

---

## 🧠 About Machine Learning (ML)

Machine learning is a subset of artificial intelligence that builds predictive models based on historical data. ML helps automate decision-making in various fields such as:

- Data Security  
- Finance  
- Healthcare  
- Fraud Detection  
- Retail  

---

## 📂 Dataset & Tools

**Dataset:**  
[Loan Prediction Problem Dataset - Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

- Contains 13 features (Gender, Income, Loan Amount, Credit History, etc.)
- Real loan application data

**Technologies Used:**
- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib & Seaborn

---

## 🧪 Model & Techniques

- **Preprocessing**: Null value removal, encoding categorical values, normalization with MinMaxScaler.
- **Model Used**: Support Vector Machine (SVM)
- **Evaluation**: Confusion Matrix, Accuracy, Precision, Recall
- **Validation**: K-Fold Cross Validation
- **Experimentation**: Logistic Regression + PCA + GridSearchCV for parameter tuning

---

## 📌 Final Output

✅ **Accuracy**: **83.81%**  
🎯 **Recall**: **100%**  
🎯 **Precision**: **81.11%**
**Confusion Matrix**: [[15 17]
                      [ 0 73]]



---

### 🔍 PCA + Logistic Regression Experiment

An additional experiment using PCA and Logistic Regression showed:

- **Best PCA Components**: 7  
- **Best C Value**: ≈ 0.046  
- **Best Cross-Validation Score**: **75.7%**

> ⚠️ Note: A `DeprecationWarning` was triggered from `DataFrameGroupBy.apply`. It doesn't affect results but may need future update.

---

## 🙋‍♀️ About Me

**👩‍💻 Name:** Ahad Almufareh  
**🎓 Project Type:** University Graduation Project  
**📚 Title:** Loan Prediction with Machine Learning  
**🔍 Specialty:** Data Science & Machine Learning

---

## 📥 How to Use

> This project runs in a Jupyter notebook. You’ll need:
- The dataset (`BankData.csv`)
- Python with libraries: `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`
- Run the notebook and follow the input prompts for live prediction!

---

## 📎 License

This project is for educational purposes only and may not be suitable for production use.

---
