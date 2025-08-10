# 🏃‍♂️ Exercise Duration Predictor Web App

A **machine learning-powered** web application built with **Streamlit** that predicts the recommended daily **exercise duration (in minutes)** needed to burn a specified number of calories based on personal attributes and exercise type.

---

## 📖 Project Overview

This project leverages a fitness-inspired dataset to build regression models predicting the exercise duration required to burn a target calorie amount.

**Features considered:**

- Age  
- Gender  
- Weight & Height  
- Activity Level  
- Exercise Type  
- Calorie Goal  

The app provides:  
- Interactive data exploration & visualizations  
- Real-time personalized exercise duration predictions  
- Model performance insights for transparency  

**Goal:** Empower healthier lifestyles through accessible, personalized, and data-driven exercise planning.

---

## ✨ Features

- ✅ **Interactive Data Exploration:** Filter dataset samples, view summary stats and data types  
- ✅ **Rich Visualizations:** Histograms, scatter plots, boxplots showing distributions & relationships  
- ✅ **Real-time Predictions:** Enter personal data & get instant exercise duration recommendations  
- ✅ **Model Performance Insights:** Compare models with MSE, RMSE, and R² metrics & visual charts  
- ✅ **User-Friendly Interface:** Clean multi-page layout with sidebar navigation and responsive widgets  

---

## 🛠️ Technologies

- Python 3.x  
- [Streamlit](https://streamlit.io) — Interactive web app framework  
- Pandas & NumPy — Data manipulation & analysis  
- Scikit-learn — Model training and evaluation  
- [XGBoost](https://xgboost.ai/) & [LightGBM](https://lightgbm.readthedocs.io/) — Gradient boosting models  
- Matplotlib, Seaborn, Plotly — Data visualizations  
- Pickle — Model saving and loading  

---

## 🧪 Model Training Workflow

Located in: `notebooks/model_training.ipynb`

- Exploratory Data Analysis (EDA) with visualizations & data quality checks  
- Data Cleaning: missing values, duplicates, BMI & MET calculations  
- Model Training using:  
  - Random Forest  
  - MLP (Neural Network)  
  - SVM  
  - XGBoost  
  - LightGBM  
- Model Evaluation: MSE, RMSE, R² metrics  
- Model Saving: Best model serialized for deployment  

---

## 🧾 Input Features for Prediction

| Feature        | Description                          | Example           |
| -------------- | ---------------------------------- | ----------------- |
| **Age**        | User’s age (18–60 years)            | 29                |
| **Gender**     | Male / Female                      | Male              |
| **Weight (kg)**| Body weight in kilograms           | 68                |
| **Height (cm)**| Height in centimeters              | 172               |
| **Activity Level** | Sedentary, Lightly Active, Moderately Active, Very Active | Moderately Active |
| **Exercise Type** | Walking, Running, Cycling, Swimming, Yoga | Running           |
| **Target Calories** | Calories to burn in a day       | 500               |

---

## 🚀 How to Run the App Locally

```bash
# Clone the repo
git clone https://github.com/your-username/exercise-duration-predictor.git

# Change directory
cd exercise-duration-predictor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
