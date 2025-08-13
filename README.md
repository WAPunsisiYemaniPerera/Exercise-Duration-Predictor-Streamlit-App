# 🏋️‍♂️ Exercise Duration Prediction App

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)


🔗 GitHub Repository: https://github.com/WAPunsisiYemaniPerera/Exercise-Duration-Predictor-Streamlit-App.git

🌐 Live App: https://exercise-duration-predictor-app.streamlit.app/

---

## 📖 Project Overview

This project predicts the exercise duration required for an individual to burn a specific number of calories based on personal attributes and activity type. It leverages machine learning regression models and an interactive Streamlit web app to make predictions in real-time.

The aim is to encourage healthier lifestyles by providing personalized exercise time recommendations.

---

## ✨ Features

- ✅ **Interactive Data Exploration** – View dataset shape, columns, sample data, and filter records  
- ✅ **Rich Visualizations** – Interactive charts to explore personal attributes and exercise-related data  
- ✅ **Live Predictions** – Input your data and get instant predictions  
- ✅ **Model Performance Insights** – View metrics like RMSE and R², plus model comparison results  
- ✅ **User-Friendly Web UI** – Sidebar navigation, clean layouts, tooltips, and responsive design  

---

## 🗂️ Dataset Description

**Attributes:**  
- Age  
- Gender  
- Weight (kg)  
- Height (cm)  
- BMI  
- Activity Level  
- Exercise Type  
- MET value  
- Target Calories  
- Duration Minutes (target)

**Source:** Adapted from FitLife: Health & Fitness Tracking Dataset (Kaggle) + synthetic data reflecting Sri Lankan community (ages 18–60).

### Preprocessing Steps:
1. Missing values imputed (median for numerical, mode for categorical)  
2. Duplicate entries removed  
3. Label encoding for categorical features  
4. Min-max normalization for scaling  
5. Train-validation-test split: 70% / 15% / 15%  

---

## 🧠 Model Training & Evaluation

**Models Trained:**
- Multi-Layer Perceptron (MLP) Regressor  
- **Random Forest Regressor** ✅ (Best Performer)  
- Support Vector Regressor (SVR)  

**Evaluation Metrics:**
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

**Best Model Performance:**
- **Random Forest Regressor**  
  - RMSE: ~0.041  
  - R²: ~0.977  

---

## 🛠️ Technologies Used

- Python 3.x  
- Streamlit – Web application framework  
- scikit-learn – Machine learning models and metrics  
- Pandas & NumPy – Data handling and preprocessing  
- Matplotlib & Seaborn – Data visualization  
- Pickle – Model persistence for deployment  

---

## 🚀 How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/WAPunsisiYemaniPerera/ExerciseDuration-Predictor-Streamlit-App.git
   cd ExerciseDuration-Predictor-Streamlit-App

2. **Create Virtual Environment (Optional but Recommended)**
   ```bash
   # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
   
3. **Install Dependencies**
    ```bash
   pip install -r requirements.txt

4. **Run the Streamlit App**
   ```bash
   streamlit run app.py


## 🌐 Deployment
**Deployment Steps:**
- Connected GitHub repo to Streamlit Cloud
- Used relative file paths instead of absolute
- Ensured all dependencies in requirements.txt
- Tested all sections post-deployment

## 🎯 Learning Outcomes
- Built a complete ML pipeline from data preprocessing to deployment
- Gained experience handling realistic data scenarios
- Implemented and compared multiple regression models
- Developed an interactive Streamlit app with good UX principles
- Overcame cloud deployment challenges
- Applied ML to solve real-world health & fitness problems
