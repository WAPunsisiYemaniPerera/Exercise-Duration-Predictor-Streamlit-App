🏃‍♂️ Exercise Duration Predictor Web App
A machine learning–powered Streamlit web application that predicts the recommended daily exercise duration (minutes) needed for a person to burn a specified number of calories, based on personal attributes and exercise type.

📖 Project Overview
This project uses a fitness-inspired dataset to build regression models that predict the exercise duration required to reach a calorie target. The model considers factors such as age, gender, weight, height, activity level, exercise type, and calorie goal.

The web app offers an interactive interface where users can explore the dataset, view visualizations, and enter their own data to receive a personalized exercise duration recommendation. The goal is to make exercise planning more accessible and data-driven.

✨ Features
✅ Interactive Data Exploration — Filter and inspect dataset samples, view summary statistics and data types.

✅ Rich Visualizations — Histograms, scatter plots, and boxplots to explore distributions and relationships.

✅ Real-time Prediction — Enter personal details and exercise preferences to get an instant duration recommendation.

✅ Model Performance Insights — Compare regression models using MSE, RMSE, and R² with visual comparisons.

✅ User-Friendly Interface — Clean multi-page app using Streamlit’s sidebar navigation and responsive widgets.

🛠️ Technologies Used
Python 3.x

Streamlit — interactive web app framework

Pandas & NumPy — data manipulation

Scikit-learn — model training & evaluation

XGBoost & LightGBM — gradient boosting regressors

Matplotlib, Seaborn & Plotly — visualizations

Pickle — model serialization

🧪 Model Training Notebook
See notebooks/model_training.ipynb for:

EDA and data quality checks

Cleaning and feature engineering (e.g., BMI, MET calculations)

Training and comparing models (Random Forest, MLP, SVM, XGBoost, LightGBM)

Evaluation (MSE, RMSE, R²) and model selection

Model serialization for deployment

🧾 Input Features for Prediction
Age: 18–60 years

Gender: Male / Female

Weight (kg)

Height (cm)

Activity Level: Sedentary, Lightly Active, Moderately Active, Very Active

Exercise Type: Walking, Running, Cycling, Swimming, Yoga

Target Calories: Desired calories to burn in a day
