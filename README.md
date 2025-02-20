# Credit-Card-Default-Prediction
A machine learning project predicting credit card default using Logistic Regression, Random Forest, and XGBoost.

# Credit Card Default Prediction 🚀

## 📌 Project Overview
This project predicts **credit card defaults** using **Logistic Regression, Random Forest, and XGBoost**. It includes **exploratory data analysis (EDA), feature selection, model evaluation, and hyperparameter tuning**. The goal is to build a robust machine learning model to identify high-risk customers.

## 📊 Dataset
- The dataset contains **credit history, payment behavior, and bill amounts**.
- Target variable: `default.payment.next.month` (0 = No Default, 1 = Default).
- Features include:
  - **LIMIT_BAL**: Credit limit amount.
  - **PAY_X**: Payment history for previous months.
  - **BILL_AMT_X**: Past bill statement amounts.
  - **PAY_AMT_X**: Payment amounts in past months.

## 🔧 How to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/nancy-precious/Credit-Card-Default-Prediction.git
   cd Credit-Card-Default-Prediction
   ```
2. **Set up the environment**:
   ```bash
   conda env create -f credit_card_default.yml
   conda activate credit_env
   ```
3. **Run Jupyter Notebook for EDA and Model Training**:
   ```bash
   jupyter notebook
   ```
4. **To train the model using Python script**:
   ```bash
   python train_model.py
   ```

## ⚡ Model Performance
- **Logistic Regression**: 71% accuracy
- **Random Forest**: 
- **XGBoost**: 

## 📂 Project Structure
```
📂 Credit-Card-Default-Prediction/
│── 📂 data/               # data (Not uploaded to GitHub)
│── 📂 notebooks/          # Jupyter Notebooks for EDA, model training
│── 📂 src/                # Python scripts for data processing & training
│── 📂 models/             # Saved models (optional)
│── 📂 reports/            # Figures, charts, and results
│── .gitignore             # Ignore unnecessary files
│── credit_card_default.yml # Conda environment file
│── README.md              # Project documentation
│── requirements.txt       # List of dependencies
│── train_model.py         # Main script to train models
```

## 🚀 Future Work
- Deploy model using **Flask or Streamlit**.
- Improve accuracy with **hyperparameter tuning and feature engineering**.


## 📬 Contact
- **GitHub**: [Nancy-Precious](https://github.com/nancy-precious)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/precious-nwaokenneya)


---
🌟 If you find this project helpful, **consider giving it a star ⭐ on GitHub!**

