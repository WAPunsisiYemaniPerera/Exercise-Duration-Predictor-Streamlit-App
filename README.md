🏃‍♂️ Exercise Duration Predictor Web App
A machine learning-powered web application built with Streamlit that predicts the recommended daily exercise duration (in minutes) needed for a person to burn a specified number of calories based on personal attributes and exercise type.

📖 Project Overview
This project uses a dataset inspired by the fitness data to build regression models that predict exercise duration required to burn target calories. The model considers factors such as age, gender, weight, height, activity level, exercise type, and calorie goal.

The web app provides an interactive interface where users can explore the dataset, visualize relationships, and enter their own data to receive a personalized exercise duration recommendation.

The aim is to promote healthier lifestyles by making exercise planning accessible, personalized, and data-driven.

✨ Features
✅ Interactive Data Exploration: Filter and explore dataset samples with summary statistics and data types.

✅ Rich Visualizations: Includes histograms, scatter plots, and boxplots showcasing data distributions and relationships.

✅ Real-time Prediction: Input your personal details and exercise preferences to instantly predict the required exercise duration.

✅ Model Performance Insights: Compare multiple regression models using metrics like MSE, RMSE, and R² with visual comparisons.

✅ User-Friendly Interface: Clean, organized multi-page app using Streamlit’s sidebar navigation and responsive widgets.

🛠️ Technologies Used
Python 3.x

Streamlit: Web app framework for interactive dashboards.

Pandas & NumPy: Data manipulation and analysis.

Scikit-learn: Machine learning model training and evaluation.

XGBoost & LightGBM: Gradient boosting regression models.

Matplotlib, Seaborn & Plotly: Data visualization libraries.

Pickle: Model serialization and loading.

🧪 Model Training Notebook
The notebook notebooks/model_training.ipynb contains:

Exploratory Data Analysis (EDA): Insightful visualizations and data quality checks.

Data Cleaning: Handling missing values, duplicates, and feature engineering (e.g., BMI and MET calculation).

Model Training: Training and comparing multiple regression algorithms including Random Forest, MLP, SVM, XGBoost, and LightGBM.

Model Evaluation: Metrics such as MSE, RMSE, and R² used to select the best performing model.

Model Saving: Serialization of the trained model for deployment.

🧾 Input Features for Prediction
Age: User’s age (18–60 years)

Gender: Male or Female

Weight (kg): Body weight in kilograms

Height (cm): Height in centimeters

Activity Level: Sedentary, Lightly Active, Moderately Active, or Very Active

Exercise Type: Walking, Running, Cycling, Swimming, Yoga

Target Calories: Number of calories the user wants to burn in a day