🏃‍♂️ Exercise Duration Predictor Web App
A machine learning-powered web application built with Streamlit that predicts the recommended daily exercise duration (in minutes) needed for a person to burn a specified number of calories based on personal attributes and exercise type.

📖 Project Overview
This project leverages a fitness-inspired dataset to build regression models that predict the exercise duration required to burn a target calorie amount.

The model considers:

Age

Gender

Weight & Height

Activity Level

Exercise Type

Calorie Goal

The web app provides:

An interactive interface for dataset exploration and visualizations

A real-time prediction tool for personalized recommendations

Insights into model performance for transparency and trust

💡 Goal: Promote healthier lifestyles by making exercise planning accessible, personalized, and data-driven.

✨ Features
✅ Interactive Data Exploration — Filter and explore dataset samples with summary statistics and data types.

✅ Rich Visualizations — Histograms, scatter plots, and boxplots showcasing data distributions & relationships.

✅ Real-time Prediction — Input your personal details & preferences to instantly predict exercise duration.

✅ Model Performance Insights — Compare multiple regression models using MSE, RMSE, and R², with visual comparisons.

✅ User-Friendly Interface — Clean, organized multi-page app with sidebar navigation & responsive widgets.

🛠️ Technologies Used
Python 3.x

Streamlit — Interactive web app framework

Pandas & NumPy — Data manipulation & analysis

Scikit-learn — Machine learning model training & evaluation

XGBoost & LightGBM — Gradient boosting regression models

Matplotlib, Seaborn, Plotly — Data visualizations

Pickle — Model serialization & loading

🧪 Model Training Workflow
📂 File: notebooks/model_training.ipynb

Steps Included:

Exploratory Data Analysis (EDA) — Insightful visualizations & data quality checks

Data Cleaning — Handling missing values, duplicates & feature engineering

BMI calculation

MET (Metabolic Equivalent) calculation

Model Training — Multiple regression algorithms:

Random Forest

MLP (Neural Network)

SVM

XGBoost

LightGBM

Model Evaluation — Metrics:

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

Model Saving — Best performing model serialized for deployment

🧾 Input Features for Prediction
Feature	Description	Example
Age	User’s age (18–60 years)	29
Gender	Male / Female	Male
Weight (kg)	Body weight in kilograms	68
Height (cm)	Height in centimeters	172
Activity Level	Sedentary, Lightly Active, Moderately Active, Very Active	Moderately Active
Exercise Type	Walking, Running, Cycling, Swimming, Yoga	Running
Target Calories	Calories to burn in a day	500

🚀 How to Run the App Locally
bash
Copy
Edit
# Clone this repository
git clone https://github.com/your-username/exercise-duration-predictor.git

# Navigate into the project directory
cd exercise-duration-predictor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
