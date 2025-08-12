import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

from sklearn.metrics import mean_squared_error, r2_score

# Load dataset and model (cache for performance)
@st.cache_data
def load_data():
    df = pd.read_csv("data/health_fitness_dataset.csv")
    # Ensure proper data types to avoid PyArrow serialization issues
    df['Age'] = df['Age'].astype('int64')
    df['Weight_kg'] = df['Weight_kg'].astype('float64')
    df['Height_cm'] = df['Height_cm'].astype('float64')
    df['Duration_minutes'] = df['Duration_minutes'].astype('float64')
    df['Target_Calories'] = df['Target_Calories'].astype('int64')
    return df

@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main app function
def main():
    st.set_page_config(
        page_title="🔥 Exercise Duration Predictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # --- Simple CSS for basic styling ---
    st.markdown("""
        <style>
        .block-container {
            padding: 1.5rem 3rem 3rem 3rem;
            max-width: 1100px;
        }
        .css-18e3th9 {
            padding-top: 1rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5em 1.5em;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            color: #fff;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
            color: #000;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔥 Exercise Duration Predictor")
    st.markdown(
        """
        <div style='font-size:18px;'>
        Welcome! This app predicts the <b>exercise duration (minutes)</b> you need to burn your target calories,
        based on your personal details and exercise choice.
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("---")

    # Load data and model
    df = load_data()
    model = load_model()
    
    # Check if model loaded successfully
    if model is None:
        st.error("❌ Failed to load the trained model. Please check if 'best_model.pkl' exists.")
        st.stop()

    # Sidebar menu with emojis and descriptions
    menu = {
        "🏠 Home": "Project overview and app introduction",
        "📊 Data Exploration": "Explore dataset shape, columns, sample data & filters",
        "📈 Visualizations": "View interactive charts about exercise & personal data",
        "🏃‍♂️ Model Prediction": "Input your data and get exercise duration prediction",
        "📉 Model Performance": "See model metrics and performance comparison"
    }
    choice = st.sidebar.radio("Navigate through the app", list(menu.keys()), format_func=lambda x: x)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{menu[choice]}**")
    
    # ----------- Home Section -----------
    if choice == "🏠 Home":
        st.header("Welcome to the Exercise Duration Predictor Project! 🔥")
        st.markdown("""
        ### Project Overview
        This application predicts the **exercise duration (in minutes)** needed to burn your target calories
        based on your personal attributes and exercise choice.
        
        ---
        
        ### Dataset Description
        - Contains personal data like age, gender, weight, height, BMI, activity level, exercise type, and target calories.
        - Target variable is the actual exercise duration needed to burn calories.
        - Dataset cleaned and preprocessed with missing value imputation, encoding, and scaling.
        
        ---
        
        ### Model Selection & Evaluation
        - Models trained include Multi-Layer Perceptron (MLP), Random Forest, Support Vector Machine (SVM), XGBoost, and LightGBM.
        - Random Forest performed best with highest R² (~0.977) and lowest RMSE.
        - Model saved and deployed for real-time predictions.
        
        ---
        
        ### Application Features
        - **Data Exploration:** View and filter the dataset.
        - **Visualizations:** Interactive charts to understand data distributions and relationships.
        - **Model Prediction:** Input your details and get instant exercise duration estimation.
        - **Model Performance:** Compare model metrics and visualize predictions vs actual durations.
        
        ---
        
        ### Deployment
        - The app is deployed on Streamlit Cloud connected to GitHub repository.
        - Handles real-time inputs and model predictions with user-friendly interface and error handling.
        
        ---
        
        ### How to Use
        Use the sidebar to navigate between sections and explore or predict based on your needs.
        
        ---
        
        **Thank you for using the Exercise Duration Predictor! Stay active and healthy! 💪**
        """)

    # ----------- Data Exploration -----------
    elif choice == "📊 Data Exploration":
        st.header("📋 Dataset Overview")
        st.markdown(f"**Shape:** {df.shape[0]} rows and {df.shape[1]} columns")

        with st.expander("▶ View Columns and Data Types"):
            st.write(df.dtypes)

        with st.expander("▶ Sample Data (Random 10 Rows)"):
            st.dataframe(df.sample(10))

        st.markdown("### 🔍 Filter Dataset")
        age_filter = st.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (18, 60))
        gender_filter = st.multiselect("Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
        activity_filter = st.multiselect("Activity Level", options=df['Activity_Level'].unique(), default=list(df['Activity_Level'].unique()))

        filtered_df = df[
            (df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1]) &
            (df['Gender'].isin(gender_filter)) &
            (df['Activity_Level'].isin(activity_filter))
        ]
        st.success(f"Filtered dataset has {filtered_df.shape[0]} records")
        st.dataframe(filtered_df.sample(min(10, filtered_df.shape[0])))

    # ----------- Visualizations -----------
    elif choice == "📈 Visualizations":
        st.header("📊 Interactive Visualizations")

        st.subheader("👥 Age Distribution by Gender")
        fig1 = px.histogram(df, x='Age', nbins=20, color='Gender', barmode='group',
                            labels={'Age':'Age (years)'}, 
                            title="Age Distribution by Gender",
                            color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'})
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("⚖️ Weight vs Exercise Duration by Exercise Type")
        fig2 = px.scatter(df, x='Weight_kg', y='Duration_minutes', color='Exercise_Type',
                          labels={'Weight_kg':'Weight (kg)', 'Duration_minutes':'Duration (min)'},
                          title="Weight vs Exercise Duration",
                          color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("⏱️ Exercise Duration by Activity Level")
        fig3 = px.box(df, x='Activity_Level', y='Duration_minutes', color='Activity_Level',
                      labels={'Activity_Level':'Activity Level', 'Duration_minutes':'Duration (min)'},
                      title="Duration Distribution by Activity Level",
                      color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig3, use_container_width=True)

    # ----------- Model Prediction -----------
    elif choice == "🏃‍♂️ Model Prediction":
        st.header("🏋️‍♂️ Predict Your Exercise Duration")
        st.markdown("Fill in your details below and get an estimated exercise duration to burn your target calories!")

        with st.form("prediction_form"):
            age = st.slider("Age (years)", 18, 60, 25, help="Select your age")
            gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
            weight = st.number_input("Weight (kg)", min_value=40.0, max_value=120.0, value=65.0, step=0.1, help="Your body weight in kilograms")
            height = st.number_input("Height (cm)", min_value=140.0, max_value=200.0, value=170.0, step=0.1, help="Your height in centimeters")
            activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"], help="Select your daily activity level")
            exercise_type = st.selectbox("Exercise Type", ["Walking", "Running", "Cycling", "Swimming", "Yoga"], help="Choose your preferred exercise")
            target_calories = st.slider("Target Calories to Burn", 150, 700, 300, help="How many calories do you want to burn?")

            submitted = st.form_submit_button("🔥 Predict Duration")

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

            # Manual min-max scaling approximation (you can improve by saving scaler from training)
            input_df = input_df[features]
            input_df_scaled = input_df.copy()
            input_df_scaled['Age'] = (input_df_scaled['Age'] - 18) / (60 - 18)
            input_df_scaled['Weight_kg'] = (input_df_scaled['Weight_kg'] - 40) / (120 - 40)
            input_df_scaled['Height_cm'] = (input_df_scaled['Height_cm'] - 140) / (200 - 140)
            input_df_scaled['BMI'] = (input_df_scaled['BMI'] - 15) / (40 - 15)
            input_df_scaled['Activity_Level'] = input_df_scaled['Activity_Level'] / 3
            input_df_scaled['Exercise_Type'] = input_df_scaled['Exercise_Type'] / 4
            input_df_scaled['MET'] = (input_df_scaled['MET'] - 2.5) / (8 - 2.5)
            input_df_scaled['Target_Calories'] = (input_df_scaled['Target_Calories'] - 150) / (700 - 150)

            with st.spinner("⏳ Predicting..."):
                try:
                    prediction_scaled = model.predict(input_df_scaled)
                    duration_pred = prediction_scaled[0] * (180 - 5) + 5  # reverse scale
                    st.success(f"✅ **Estimated Exercise Duration:** {duration_pred:.1f} minutes")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

    # ----------- Model Performance -----------
    elif choice == "📉 Model Performance":
        st.header("📊 Model Performance Summary")

        results = {
            "Model": ["MLP", "Random Forest", "SVM", "XGBoost", "LightGBM"],
            "MSE": [0.0041, 0.0017, 0.0029, 0.0019, 0.0018],
            "RMSE": [0.0639, 0.0411, 0.0541, 0.0436, 0.0424],
            "R2": [0.9445, 0.9770, 0.9602, 0.9750, 0.9760]
        }
        perf_df = pd.DataFrame(results)
        st.table(perf_df)

        fig_perf = px.bar(perf_df, x='Model', y='RMSE', title='Model RMSE Comparison', color='RMSE',
                          color_continuous_scale='Viridis')
        st.plotly_chart(fig_perf, use_container_width=True)

        st.subheader("Predicted vs Actual Duration Scatter Plot")
        
        try:
            # Load fresh data for testing
            test_df = load_data().sample(500, random_state=42)

            # Preprocessing same as training
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

            # Normalize approx
            for col in ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Activity_Level', 'Exercise_Type', 'MET', 'Target_Calories']:
                if col in ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Target_Calories']:
                    test_df[col] = (test_df[col] - test_df[col].min()) / (test_df[col].max() - test_df[col].min())
                else:
                    test_df[col] = test_df[col] / test_df[col].max()

            X_test = test_df[features]
            y_test = test_df['Duration_minutes']

            # Ensure model is available
            if model is not None:
                y_pred = model.predict(X_test)
                y_pred = y_pred * (180 - 5) + 5  # reverse scale

                fig_scatter = px.scatter(x=y_test, y=y_pred,
                                         labels={'x': 'Actual Duration (minutes)', 'y': 'Predicted Duration (minutes)'},
                                         title='Actual vs Predicted Exercise Duration',
                                         color_discrete_sequence=['#ff6361'])
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Calculate and display metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MSE", f"{mse:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    st.metric("R²", f"{r2:.4f}")
            else:
                st.error("❌ Model not available for performance evaluation")
                
        except Exception as e:
            st.error(f"❌ Error generating performance plot: {e}")
            st.info("This might be due to data preprocessing issues or model compatibility.")

if __name__ == '__main__':
    main()