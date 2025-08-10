import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle

from sklearn.metrics import mean_squared_error, r2_score

# Load dataset and model (cache for performance)
@st.cache_data
def load_data():
    df = pd.read_csv("E:/HORIZON CAMPUS/AASEMESTER6/Intelligent Systems/Exercise_Recommendation_System/data/health_fitness_dataset.csv")
    return df

@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Preprocessing function matching training pipeline
def preprocess_input(df):
    # Handle missing values (impute median for numericals, mode for categoricals)
    numerical_cols = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'MET', 'Target_Calories']
    categorical_cols = ['Gender', 'Activity_Level', 'Exercise_Type']

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns using known encoding from training
    gender_map = {'Male': 1, 'Female': 0}
    activity_map = {'Sedentary': 3, 'Lightly Active': 2, 'Moderately Active': 1, 'Very Active': 0}
    exercise_map = {'Walking': 3, 'Running': 1, 'Cycling': 2, 'Swimming': 0, 'Yoga': 4}

    df['Gender'] = df['Gender'].map(gender_map).fillna(0).astype(int)
    df['Activity_Level'] = df['Activity_Level'].map(activity_map).fillna(3).astype(int)
    df['Exercise_Type'] = df['Exercise_Type'].map(exercise_map).fillna(3).astype(int)

    # Calculate BMI if missing (optional)
    df['BMI'] = df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)

    # Calculate MET based on exercise type
    met_values = {'Walking': 3.5, 'Running': 8.0, 'Cycling': 6.0, 'Swimming': 7.0, 'Yoga': 2.5}
    df['MET'] = df['Exercise_Type'].map({v:k for k,v in exercise_map.items()})
    df['MET'] = df['MET'].map(met_values).fillna(3.5)

    # Select columns in order expected by model
    features = ['Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI', 'Activity_Level', 'Exercise_Type', 'MET', 'Target_Calories']
    return df[features]

# Main app function
def main():
    st.set_page_config(page_title="Exercise Duration Predictor", layout="wide")
    
    st.title("Exercise Duration Predictor")
    st.markdown("""
    This app predicts the **exercise duration (minutes)** needed to burn your target calories,
    based on your personal attributes and exercise choice.
    """)

    # Load data and model
    df = load_data()
    model = load_model()

    # Sidebar menu
    menu = ["Data Exploration", "Visualizations", "Model Prediction", "Model Performance"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # --------------------------
    # Data Exploration Section
    # --------------------------
    if choice == "Data Exploration":
        st.header("Dataset Overview")

        st.write(f"**Dataset shape:** {df.shape}")
        st.write("**Columns and data types:**")
        st.write(df.dtypes)

        st.write("**Sample data:**")
        st.dataframe(df.sample(10))

        # Interactive filtering
        st.subheader("Filter Data")
        age_filter = st.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (18, 60))
        gender_filter = st.multiselect("Select Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
        activity_filter = st.multiselect("Select Activity Level", options=df['Activity_Level'].unique(), default=list(df['Activity_Level'].unique()))

        filtered_df = df[
            (df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1]) &
            (df['Gender'].isin(gender_filter)) &
            (df['Activity_Level'].isin(activity_filter))
        ]
        st.write(f"Filtered dataset size: {filtered_df.shape[0]}")
        st.dataframe(filtered_df.sample(min(10, filtered_df.shape[0])))

    # --------------------------
    # Visualizations Section
    # --------------------------
    elif choice == "Visualizations":
        st.header("Data Visualizations")

        # Plot 1: Distribution of Age
        st.subheader("Age Distribution")
        fig1 = px.histogram(df, x='Age', nbins=20, color='Gender', barmode='group',
                            labels={'Age':'Age'}, title="Age Distribution by Gender")
        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: Scatter plot of Weight vs Duration colored by Exercise Type
        st.subheader("Weight vs Exercise Duration")
        fig2 = px.scatter(df, x='Weight_kg', y='Duration_minutes', color='Exercise_Type',
                          labels={'Weight_kg':'Weight (kg)', 'Duration_minutes':'Duration (minutes)'},
                          title="Weight vs Exercise Duration by Exercise Type")
        st.plotly_chart(fig2, use_container_width=True)

        # Plot 3: Boxplot of Duration by Activity Level
        st.subheader("Exercise Duration by Activity Level")
        fig3 = px.box(df, x='Activity_Level', y='Duration_minutes', color='Activity_Level',
                      labels={'Activity_Level':'Activity Level', 'Duration_minutes':'Duration (minutes)'},
                      title="Exercise Duration Distribution by Activity Level")
        st.plotly_chart(fig3, use_container_width=True)

    # --------------------------
    # Model Prediction Section
    # --------------------------
    elif choice == "Model Prediction":
        st.header("Make a Prediction")

        with st.form("prediction_form"):
            age = st.slider("Age", 18, 60, 25, help="Select your age")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
            weight = st.number_input("Weight (kg)", min_value=40.0, max_value=120.0, value=65.0, step=0.1, help="Enter your weight")
            height = st.number_input("Height (cm)", min_value=140.0, max_value=200.0, value=170.0, step=0.1, help="Enter your height")
            activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"], help="Select your daily activity level")
            exercise_type = st.selectbox("Exercise Type", ["Walking", "Running", "Cycling", "Swimming", "Yoga"], help="Select the exercise type")
            target_calories = st.slider("Target Calories to Burn", 150, 700, 300, help="Calories you want to burn")

            submitted = st.form_submit_button("Predict Duration")

        if submitted:
            # Prepare input dataframe
            input_dict = {
                'Age': [age],
                'Gender': [gender],
                'Weight_kg': [weight],
                'Height_cm': [height],
                'Activity_Level': [activity_level],
                'Exercise_Type': [exercise_type],
                'Target_Calories': [target_calories]
            }
            input_df = pd.DataFrame(input_dict)

            # Add BMI and MET for prediction (same formula as training)
            input_df['BMI'] = input_df['Weight_kg'] / ((input_df['Height_cm'] / 100) ** 2)
            met_map = {'Walking': 3.5, 'Running': 8.0, 'Cycling': 6.0, 'Swimming': 7.0, 'Yoga': 2.5}
            input_df['MET'] = input_df['Exercise_Type'].map(met_map)

            # Encode categorical inputs to model format
            gender_map = {'Male': 1, 'Female': 0}
            activity_map = {'Sedentary': 3, 'Lightly Active': 2, 'Moderately Active': 1, 'Very Active': 0}
            exercise_map = {'Walking': 3, 'Running': 1, 'Cycling': 2, 'Swimming': 0, 'Yoga': 4}

            input_df['Gender'] = input_df['Gender'].map(gender_map)
            input_df['Activity_Level'] = input_df['Activity_Level'].map(activity_map)
            input_df['Exercise_Type'] = input_df['Exercise_Type'].map(exercise_map)

            features = ['Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI', 'Activity_Level', 'Exercise_Type', 'MET', 'Target_Calories']

            # Scale inputs to [0,1] using min-max scaler fitted on original dataset
            # For simplicity here, just clip and normalize approx based on training ranges
            # You can improve this by saving scaler during training
            input_df = input_df[features]
            input_df_scaled = input_df.copy()
            # Manual scaling approximation:
            input_df_scaled['Age'] = (input_df_scaled['Age'] - 18) / (60 - 18)
            input_df_scaled['Weight_kg'] = (input_df_scaled['Weight_kg'] - 40) / (120 - 40)
            input_df_scaled['Height_cm'] = (input_df_scaled['Height_cm'] - 140) / (200 - 140)
            input_df_scaled['BMI'] = (input_df_scaled['BMI'] - 15) / (40 - 15)
            input_df_scaled['Activity_Level'] = input_df_scaled['Activity_Level'] / 3
            input_df_scaled['Exercise_Type'] = input_df_scaled['Exercise_Type'] / 4
            input_df_scaled['MET'] = (input_df_scaled['MET'] - 2.5) / (8 - 2.5)
            input_df_scaled['Target_Calories'] = (input_df_scaled['Target_Calories'] - 150) / (700 - 150)

            # Prediction with loading spinner
            with st.spinner("Predicting..."):
                prediction_scaled = model.predict(input_df_scaled)
                # Reverse scale Duration_minutes (target) assuming min=5, max=180 during training scaling
                duration_pred = prediction_scaled[0] * (180 - 5) + 5

            st.success(f"Estimated Exercise Duration: {duration_pred:.1f} minutes")

    # --------------------------
    # Model Performance Section
    # --------------------------
    elif choice == "Model Performance":
        st.header("Model Performance Metrics")

        # Hard-coded model results from training (you can load from file or DB)
        results = {
            "Model": ["MLP", "Random Forest", "SVM", "XGBoost", "LightGBM"],
            "MSE": [0.0041, 0.0017, 0.0029, 0.0019, 0.0018],
            "RMSE": [0.0639, 0.0411, 0.0541, 0.0436, 0.0424],
            "R2": [0.9445, 0.9770, 0.9602, 0.9750, 0.9760]
        }
        perf_df = pd.DataFrame(results)
        st.table(perf_df)

        # Plot comparison
        fig_perf = px.bar(perf_df, x='Model', y='RMSE', title='Model RMSE Comparison', color='RMSE')
        st.plotly_chart(fig_perf, use_container_width=True)

        # Note: Confusion matrix is for classification; here you can show predicted vs actual plot instead
        st.subheader("Predicted vs Actual Duration Scatter Plot")

        # Load test dataset predictions for visualization (simulate)
        test_df = load_data()
        test_df = test_df.sample(500, random_state=42)  # sample for speed

        # Preprocess for prediction
        gender_map = {'Male': 1, 'Female': 0}
        activity_map = {'Sedentary': 3, 'Lightly Active': 2, 'Moderately Active': 1, 'Very Active': 0}
        exercise_map = {'Walking': 3, 'Running': 1, 'Cycling': 2, 'Swimming': 0, 'Yoga': 4}

        test_df['Gender'] = test_df['Gender'].map(gender_map)
        test_df['Activity_Level'] = test_df['Activity_Level'].map(activity_map)
        test_df['Exercise_Type'] = test_df['Exercise_Type'].map(exercise_map)
        test_df['BMI'] = test_df['Weight_kg'] / ((test_df['Height_cm'] / 100) ** 2)
        met_values = {'Walking': 3.5, 'Running': 8.0, 'Cycling': 6.0, 'Swimming': 7.0, 'Yoga': 2.5}
        test_df['MET'] = test_df['Exercise_Type'].map({v:k for k,v in exercise_map.items()})
        test_df['MET'] = test_df['MET'].map(met_values)

        features = ['Age', 'Gender', 'Weight_kg', 'Height_cm', 'BMI', 'Activity_Level', 'Exercise_Type', 'MET', 'Target_Calories']

        # Approximate scaling
        for col in ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Activity_Level', 'Exercise_Type', 'MET', 'Target_Calories']:
            if col in ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Target_Calories']:
                test_df[col] = (test_df[col] - test_df[col].min()) / (test_df[col].max() - test_df[col].min())
            else:
                test_df[col] = test_df[col] / test_df[col].max()

        X_test = test_df[features]
        y_test = test_df['Duration_minutes']

        y_pred = model.predict(X_test)
        # Reverse scale duration approx
        y_pred = y_pred * (180 - 5) + 5

        fig_scatter = px.scatter(x=y_test, y=y_pred,
                                 labels={'x':'Actual Duration (scaled)', 'y':'Predicted Duration (scaled)'},
                                 title='Actual vs Predicted Exercise Duration')
        st.plotly_chart(fig_scatter, use_container_width=True)


if __name__ == '__main__':
    main()
