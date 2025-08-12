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
    
    # Simplified CSS styling without boxes
    st.markdown("""
        <style>
        .block-container {
            padding: 1rem 2rem;
            max-width: 1200px;
        }
        .charm-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        .charm-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
        }
        .charm-header p {
            font-size: 1rem;
            opacity: 0.9;
            margin: 0.5rem 0;
        }
        .charm-section {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1rem;
            margin: 2rem 0 1rem 0;
            color: white;
            text-align: center;
        }
        .charm-section h2 {
            margin: 0;
            font-size: 1.8rem;
            font-weight: 600;
        }
        .charm-text {
            color: #333;
            margin: 1rem 0;
            padding: 1rem;
            background: #f8f9fa;
        }
        .charm-text h3 {
            color: #667eea;
            margin-bottom: 0.5rem;
            font-size: 1.2rem;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 0.6em 1.5em;
            border: none;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .sidebar .sidebar-content .stRadio > label {
            color: white;
            font-weight: 500;
        }
        .charm-success {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 1.5rem;
            text-align: center;
            margin: 2rem 0;
        }
        .charm-success h2 {
            margin-bottom: 0.5rem;
            font-size: 1.8rem;
        }
        .charm-success h1 {
            font-size: 2.5rem;
            margin: 0.5rem 0;
        }
        .stats-row {
            display: flex;
            justify-content: space-around;
            margin: 2rem 0;
            gap: 1rem;
        }
        .simple-metric {
            text-align: center;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .simple-metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.3rem;
        }
        .simple-metric-label {
            color: #666;
            font-size: 0.9rem;
            font-weight: 500;
        }
        @media (max-width: 768px) {
            .stats-row {
                flex-direction: column;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Charming Header ---
    st.markdown("""
        <div class="charm-header">
            <h1>🔥 Exercise Duration Predictor</h1>
            <p>Transform your fitness goals into actionable workout plans with AI-powered predictions</p>
        </div>
    """, unsafe_allow_html=True)

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
        st.markdown('<div class="charm-section"><h2>🚀 Welcome to the Future of Fitness Planning!</h2></div>', unsafe_allow_html=True)
        
        # Stats overview
        st.markdown("""
            <div class="stats-row">
                <div class="simple-metric">
                    <div class="simple-metric-value">{:,}</div>
                    <div class="simple-metric-label">📊 Dataset Records</div>
                </div>
                <div class="simple-metric">
                    <div class="simple-metric-value">97.7%</div>
                    <div class="simple-metric-label">🎯 Model Accuracy</div>
                </div>
                <div class="simple-metric">
                    <div class="simple-metric-value">&lt;1s</div>
                    <div class="simple-metric-label">⚡ Prediction Speed</div>
                </div>
            </div>
        """.format(df.shape[0]), unsafe_allow_html=True)
        
        # Project overview
        st.markdown('<div class="charm-text"><h3>🎯 Project Overview</h3></div>', unsafe_allow_html=True)
        st.write("""
            This cutting-edge application leverages **machine learning** to predict the **exercise duration (in minutes)** 
            needed to burn your target calories based on your personal attributes and exercise choice. 
            Say goodbye to guesswork and hello to data-driven fitness planning!
        """)
        
        # Features in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="charm-text"><h3>🧠 AI-Powered Predictions</h3><p>Advanced machine learning algorithms trained on comprehensive fitness data</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="charm-text"><h3>📊 Comprehensive Analytics</h3><p>Explore data distributions, correlations, and insights about fitness patterns</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="charm-text"><h3>🎨 Interactive Visualizations</h3><p>Beautiful charts and graphs to understand your fitness journey</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="charm-text"><h3>⚡ Real-Time Results</h3><p>Instant predictions with professional-grade accuracy</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="charm-text"><h3>🔬 Model Performance</h3><p>Transparent metrics showing model accuracy and reliability</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="charm-text"><h3>📱 User-Friendly Interface</h3><p>Intuitive design that makes fitness planning enjoyable</p></div>', unsafe_allow_html=True)
        
        # Dataset description
        st.markdown('<div class="charm-section"><h2>📋 Dataset & Technology</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="charm-text"><h3>📊 Rich Data Collection</h3><p>Personal attributes: age, gender, weight, height, BMI, activity level, exercise type, and target calories</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="charm-text"><h3>🧹 Data Quality</h3><p>Comprehensive preprocessing with missing value imputation, encoding, and scaling</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="charm-text"><h3>🎯 Target Variable</h3><p>Actual exercise duration needed to burn calories - the core prediction target</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="charm-text"><h3>🔬 Advanced Models</h3><p>MLP, Random Forest, SVM, XGBoost, and LightGBM with Random Forest achieving 97.7% R²</p></div>', unsafe_allow_html=True)
        
        # How to use
        st.markdown('<div class="charm-section"><h2>🚀 Getting Started</h2></div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="charm-text">
                <h3>📱 Navigation Guide</h3>
                <p><strong>🏠 Home:</strong> You're here! Overview and project information</p>
                <p><strong>📊 Data Exploration:</strong> Dive deep into the dataset with filters and analysis</p>
                <p><strong>📈 Visualizations:</strong> Interactive charts and data insights</p>
                <p><strong>🏃‍♂️ Model Prediction:</strong> Input your details and get personalized predictions</p>
                <p><strong>📉 Model Performance:</strong> Evaluate model accuracy and reliability</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
            <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 5px;">
                <h3>💪 Ready to Transform Your Fitness Journey?</h3>
                <p>Use the sidebar to explore different sections and start making data-driven fitness decisions today!</p>
                <p><strong>Stay active, stay healthy, and let AI guide your path to fitness success! 🎯</strong></p>
            </div>
        """, unsafe_allow_html=True)

    # ----------- Data Exploration -----------
    elif choice == "📊 Data Exploration":
        st.markdown('<div class="charm-section"><h2>📋 Dataset Overview & Exploration</h2></div>', unsafe_allow_html=True)
        
        # Dataset stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="simple-metric">
                    <div class="simple-metric-value">{df.shape[0]:,}</div>
                    <div class="simple-metric-label">Total Records</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="simple-metric">
                    <div class="simple-metric-value">{df.shape[1]}</div>
                    <div class="simple-metric-label">Features</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="simple-metric">
                    <div class="simple-metric-value">{df['Gender'].nunique()}</div>
                    <div class="simple-metric-label">Gender Categories</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="simple-metric">
                    <div class="simple-metric-value">{df['Exercise_Type'].nunique()}</div>
                    <div class="simple-metric-label">Exercise Types</div>
                </div>
            """, unsafe_allow_html=True)

        with st.expander("🔍 View Columns and Data Types", expanded=False):
            st.write(df.dtypes)

        with st.expander("📊 Sample Data (Random 10 Rows)", expanded=False):
            st.dataframe(df.sample(10), use_container_width=True)

        st.markdown('<div class="charm-section"><h2>🔍 Interactive Data Filtering</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            age_filter = st.slider("🎂 Age Range", int(df['Age'].min()), int(df['Age'].max()), (18, 60))
            gender_filter = st.multiselect("👥 Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
        
        with col2:
            activity_filter = st.multiselect("🏃‍♂️ Activity Level", options=df['Activity_Level'].unique(), default=list(df['Activity_Level'].unique()))
            exercise_filter = st.multiselect("💪 Exercise Type", options=df['Exercise_Type'].unique(), default=list(df['Exercise_Type'].unique()))

        filtered_df = df[
            (df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1]) &
            (df['Gender'].isin(gender_filter)) &
            (df['Activity_Level'].isin(activity_filter)) &
            (df['Exercise_Type'].isin(exercise_filter))
        ]
        
        st.success(f"🎯 **Filtered dataset has {filtered_df.shape[0]} records**")
        
        if filtered_df.shape[0] > 0:
            st.markdown('<div class="charm-text"><h3>📊 Filtered Data Preview</h3></div>', unsafe_allow_html=True)
            st.dataframe(filtered_df.sample(min(10, filtered_df.shape[0])), use_container_width=True)
        else:
            st.warning("⚠️ No data matches your current filters. Try adjusting the criteria.")

    # ----------- Visualizations -----------
    elif choice == "📈 Visualizations":
        st.markdown('<div class="charm-section"><h2>📊 Interactive Data Visualizations</h2></div>', unsafe_allow_html=True)
        st.markdown("Explore the relationships and patterns in your fitness data through these interactive charts!")

        st.markdown('<div class="charm-text"><h3>👥 Age Distribution by Gender</h3></div>', unsafe_allow_html=True)
        fig1 = px.histogram(df, x='Age', nbins=20, color='Gender', barmode='group',
                            labels={'Age':'Age (years)'}, 
                            title="Age Distribution by Gender",
                            color_discrete_map={'Male': '#1f77b4', 'Female': '#ff7f0e'})
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown('<div class="charm-text"><h3>⚖️ Weight vs Exercise Duration by Exercise Type</h3></div>', unsafe_allow_html=True)
        fig2 = px.scatter(df, x='Weight_kg', y='Duration_minutes', color='Exercise_Type',
                          labels={'Weight_kg':'Weight (kg)', 'Duration_minutes':'Duration (min)'},
                          title="Weight vs Exercise Duration",
                          color_discrete_sequence=px.colors.qualitative.Safe)
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="charm-text"><h3>⏱️ Exercise Duration by Activity Level</h3></div>', unsafe_allow_html=True)
        fig3 = px.box(df, x='Activity_Level', y='Duration_minutes', color='Activity_Level',
                      labels={'Activity_Level':'Activity Level', 'Duration_minutes':'Duration (min)'},
                      title="Duration Distribution by Activity Level",
                      color_discrete_sequence=px.colors.sequential.Teal)
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ----------- Model Prediction -----------
    elif choice == "🏃‍♂️ Model Prediction":
        st.markdown('<div class="charm-section"><h2>🏋️‍♂️ AI-Powered Exercise Duration Prediction</h2></div>', unsafe_allow_html=True)
        st.markdown("Fill in your details below and get an instant, AI-powered estimate of your exercise duration!")

        st.markdown('<div class="charm-text"><h3>🎯 Enter Your Fitness Profile</h3></div>', unsafe_allow_html=True)

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("🎂 Age (years)", 18, 60, 25, help="Select your age")
                weight = st.number_input("⚖️ Weight (kg)", min_value=40.0, max_value=120.0, value=65.0, step=0.1, help="Your body weight in kilograms")
                activity_level = st.selectbox("🏃‍♂️ Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"], help="Select your daily activity level")
                target_calories = st.slider("🔥 Target Calories to Burn", 150, 700, 300, help="How many calories do you want to burn?")
            
            with col2:
                gender = st.selectbox("👥 Gender", ["Male", "Female"], help="Select your gender")
                height = st.number_input("📏 Height (cm)", min_value=140.0, max_value=200.0, value=170.0, step=0.1, help="Your height in centimeters")
                exercise_type = st.selectbox("💪 Exercise Type", ["Walking", "Running", "Cycling", "Swimming", "Yoga"], help="Choose your preferred exercise")

            submitted = st.form_submit_button("🚀 Get AI Prediction!")

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

            with st.spinner("🤖 AI is analyzing your profile..."):
                try:
                    prediction_scaled = model.predict(input_df_scaled)
                    duration_pred = prediction_scaled[0] * (180 - 5) + 5  # reverse scale
                    
                    st.markdown(f"""
                        <div class="charm-success">
                            <h2>🎯 AI Prediction Complete!</h2>
                            <h1>{duration_pred:.1f} minutes</h1>
                            <p>Estimated exercise duration to burn {target_calories} calories</p>
                            <p><strong>Exercise:</strong> {exercise_type} | <strong>Activity Level:</strong> {activity_level}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
                    
                    # Additional insights
                    st.markdown('<div class="charm-text"><h3>💡 Fitness Insights</h3></div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("BMI", f"{input_df['BMI'].iloc[0]:.1f}")
                    with col2:
                        st.metric("MET Value", f"{input_df['MET'].iloc[0]:.1f}")
                    with col3:
                        st.metric("Calories per Minute", f"{target_calories/duration_pred:.1f}")
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

    # ----------- Model Performance -----------
    elif choice == "📉 Model Performance":
        st.markdown('<div class="charm-section"><h2>📊 Model Performance & Evaluation</h2></div>', unsafe_allow_html=True)
        st.markdown("Discover how our AI models perform and compare their accuracy!")

        st.markdown('<div class="charm-text"><h3>🏆 Model Comparison Results</h3></div>', unsafe_allow_html=True)
        
        results = {
            "Model": ["MLP", "Random Forest", "SVM", "XGBoost", "LightGBM"],
            "MSE": [0.0041, 0.0017, 0.0029, 0.0019, 0.0018],
            "RMSE": [0.0639, 0.0411, 0.0541, 0.0436, 0.0424],
            "R2": [0.9445, 0.9770, 0.9602, 0.9750, 0.9760]
        }
        perf_df = pd.DataFrame(results)
        
        # Highlight best model
        best_model = perf_df.loc[perf_df['R2'].idxmax()]
        st.success(f"🥇 **Best Model:** {best_model['Model']} with R² = {best_model['R2']:.4f}")

        st.dataframe(perf_df, use_container_width=True)

        # Performance visualization
        st.markdown('<div class="charm-text"><h3>📈 RMSE Comparison Chart</h3></div>', unsafe_allow_html=True)
        fig_perf = px.bar(perf_df, x='Model', y='RMSE', title='Model RMSE Comparison (Lower is Better)', 
                          color='RMSE', color_continuous_scale='Viridis')
        fig_perf.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        st.markdown('<div class="charm-section"><h2>🔬 Real-Time Model Testing</h2></div>', unsafe_allow_html=True)
        st.markdown("See how our model performs on actual data with predictions vs actual values!")
        
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

                st.markdown('<div class="charm-text"><h3>📊 Predicted vs Actual Duration Scatter Plot</h3></div>', unsafe_allow_html=True)
                fig_scatter = px.scatter(x=y_test, y=y_pred,
                                         labels={'x': 'Actual Duration (minutes)', 'y': 'Predicted Duration (minutes)'},
                                         title='Actual vs Predicted Exercise Duration',
                                         color_discrete_sequence=['#ff6361'])
                fig_scatter.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=14)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Calculate and display metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                st.markdown('<div class="charm-text"><h3>📈 Performance Metrics on Test Data</h3></div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                        <div class="simple-metric">
                            <div class="simple-metric-value">{mse:.4f}</div>
                            <div class="simple-metric-label">Mean Squared Error</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="simple-metric">
                            <div class="simple-metric-value">{rmse:.4f}</div>
                            <div class="simple-metric-label">Root Mean Squared Error</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                        <div class="simple-metric">
                            <div class="simple-metric-value">{r2:.4f}</div>
                            <div class="simple-metric-label">R² Score</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ Model not available for performance evaluation")
                
        except Exception as e:
            st.error(f"❌ Error generating performance plot: {e}")
            st.info("This might be due to data preprocessing issues or model compatibility.")

if __name__ == '__main__':
    main()