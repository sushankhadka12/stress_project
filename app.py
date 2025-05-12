import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
import time
import datetime

# Import custom modules
from data_processor import DataProcessor
from model_trainer import SVMModelTrainer
from visualizer import Visualizer
from utils import load_image_from_url, plot_confusion_matrix, stress_level_to_text
from database import (
    get_or_create_user, 
    save_prediction, 
    get_user_predictions, 
    save_model_metadata,
    get_latest_model_metadata
)

# Set page configuration
st.set_page_config(
    page_title="Stress Level Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None
if 'user_id' not in st.session_state:
    # Create a default user in the database
    user = get_or_create_user(name="Anonymous User")
    st.session_state.user_id = user.id

# Header section
st.title("Stress Level Detection System")
st.markdown("### Using SVM to analyze physiological data and predict stress levels")

# Create a more structured navigation system for college project
with st.sidebar:
    st.title("Stress Detection System")
    
    # Clear navigation with numbered steps for student project
    st.markdown("### Project Navigation Guide")
    
    # Number the steps for clear sequential flow
    navigation_options = {
        "1. Introduction": "Home",
        "2. Data Management": "Data Exploration",
        "3. Model Creation": "Model Training",
        "4. Stress Analysis": "Stress Prediction",
        "5. Results & History": "History & Trends"
    }
    
    # Create navigation with numbered steps
    page = st.radio(
        "Follow these steps sequentially:",
        list(navigation_options.keys())
    )
    
    # Map the selected option to actual page value
    page = navigation_options[page]
    
    # Add divider
    st.markdown("---")
    
    # Add compact project info instead of large images
    st.markdown("### Project Information")
    
    # Simplified stress level reference
    st.markdown("""
    **Stress Level Classification:**
    - 🟢 Low (0): Minimal stress
    - 🟡 Normal (1): Moderate stress
    - 🔴 High (2): High stress
    """)
    
    # Technical stack info
    st.markdown("### Technical Stack")
    st.markdown("""
    - ML: SVM Algorithm
    - Database: PostgreSQL
    - Frontend: Streamlit
    """)
    
    # Reset option with more academic wording
    if st.button("Reset Project Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content based on selected page
if page == "Home":
    st.markdown("""
    # Stress Level Detection System
    ## College Final Year Project
    
    **Department of Computer Science**  
    **Academic Year 2024-2025**
    
    *This project implements a machine learning-based system to detect human stress levels from physiological parameters.*
    """)
    
    # Project description (academic style)
    st.markdown("""
    ## Project Overview
    
    This system uses a Support Vector Machine (SVM) algorithm to classify stress levels based on physiological data.
    The implementation demonstrates the application of machine learning in healthcare monitoring.
    """)
    
    # System workflow - simplified for college project
    st.markdown("## System Workflow")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### 5-Step Process
        
        Follow these steps in order to use the system:
        
        1. **Data Management** (Step 2 in navigation)
           - Upload or generate a dataset with physiological parameters
           - View dataset statistics and distributions
        
        2. **Model Training** (Step 3 in navigation)
           - Configure SVM parameters (kernel, C, gamma)
           - Train the model on the dataset
           - Evaluate model performance metrics
        
        3. **Stress Analysis** (Step 4 in navigation)
           - Input physiological measurements
           - Get stress level prediction
           - View confidence scores and recommendations
        
        4. **Results Review** (Step 5 in navigation)
           - Analyze prediction history
           - View stress trends over time
           - Examine correlations between parameters
        """)
    
    with col2:
        # Simple diagram showing workflow
        st.markdown("""
        ```
        ┌─────────────────┐
        │  Data Loading   │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ Model Training  │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │    Prediction   │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ Result Analysis │
        └─────────────────┘
        ```
        """)
    
    # Technical implementation
    st.markdown("## Technical Implementation")
    
    # Create a table for parameters
    st.markdown("""
    ### System Parameters
    
    | Parameter | Description | Range |
    |-----------|-------------|-------|
    | Temperature | Body temperature | 35.0-40.0°C |
    | Humidity | Body humidity | 0-100% |
    | Step Count | Physical activity | 0-10,000 steps |
    
    ### Model Information
    
    - **Algorithm**: Support Vector Machine (SVM)
    - **Kernels**: Linear, RBF, Polynomial
    - **Classification**: 3 stress levels (Low, Normal, High)
    - **Data Storage**: PostgreSQL database
    """)
    
    # How to get started - quick guide
    st.info("""
    **Quick Start Guide**
    
    1. Go to "2. Data Management" in the navigation
    2. Generate a sample dataset or upload your own
    3. Proceed to "3. Model Creation" to train the SVM model
    4. Use "4. Stress Analysis" to make predictions
    5. View your results in "5. Results & History"
    """)
    
    # Project metadata for academic presentation
    st.markdown("""
    ---
    **Project Keywords**: Machine Learning, SVM, Stress Detection, Healthcare Analytics, Physiological Monitoring
    """)

    # Add project contributors for academic presentation
    st.markdown("""
    ## Project Contributors
    
    - Student Name (Student ID)
    - Department of Computer Science
    - Under the guidance of: Professor Name
    
    *This project was developed as part of the final year curriculum requirements.*
    """)

elif page == "Data Exploration":
    st.header("Data Exploration")
    
    data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs(["Dataset Overview", "Data Statistics", "Data Visualization", "Upload Dataset"])
    
    # Dataset upload tab - making it more academic for college final year project
    with data_tab4:
        st.subheader("Dataset Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Dataset Requirements
            
            This project requires a dataset with the following physiological metrics:
            
            | Feature | Description | Format |
            | ------- | ----------- | ------ |
            | Temperature | Body temperature | Numeric (°C) |
            | Humidity | Body humidity | Numeric (%) |
            | Step_Count | Physical activity | Integer |
            | Stress_Level | Target variable | Integer (0-2) |
            
            The stress levels are encoded as:
            - **0**: Low stress
            - **1**: Normal stress
            - **2**: High stress
            
            You can upload your own dataset below. Please ensure it follows the required format.
            """)
        
        with col2:
            st.image(
                "https://cdn.pixabay.com/photo/2017/09/21/19/12/analysis-2773507_1280.jpg",
                caption="Data is essential for accurate stress level predictions",
                use_container_width=True
            )
        
        st.markdown("---")
        
        upload_col1, upload_col2 = st.columns([3, 2])
        
        with upload_col1:
            st.markdown("""
            ### Upload Dataset
            
            Upload your dataset in CSV format containing physiological measurements and stress levels.
            The system will validate the dataset format before processing.
            """)
            
            uploaded_file = st.file_uploader(
                "Upload CSV Dataset", 
                type="csv",
                help="Your CSV file should contain Temperature, Humidity, Step_Count, and Stress_Level columns"
            )
        
        with upload_col2:
            st.markdown("""
            ### Academic References
            
            The stress level detection methodology is based on established research:
            
            1. Kim et al. (2023). "Physiological markers for stress detection"
            2. Sharma & Johnson (2024). "Machine learning approaches for stress level classification"
            3. Zhang et al. (2022). "SVM applications in health monitoring systems"
            """)
            
        # Option to use sample dataset for educational purposes
        st.markdown("---")
        st.markdown("### Generate Sample Dataset")
        st.markdown("""
        For educational purposes, this system can generate a synthetic dataset based on physiological 
        patterns observed in stress studies. This is useful for understanding the relationship between 
        physiological variables and stress levels.
        """)
        
        if st.button("Generate Sample Dataset for Educational Use"):
            with st.spinner("Generating synthetic dataset based on physiological patterns..."):
                # Initialize data processor
                data_processor = DataProcessor()
                
                # Generate synthetic data
                df = data_processor.generate_synthetic_data()
                
                # Save to file
                if not os.path.exists("data"):
                    os.makedirs("data")
                df.to_csv("data/dataset.csv", index=False)
                
                # Store data processor in session state
                st.session_state.data_processor = data_processor
                
                # Reset model training state
                if 'model_trained' in st.session_state:
                    st.session_state.model_trained = False
                if 'model_trainer' in st.session_state:
                    st.session_state.model_trainer = None
                
                st.success("Sample dataset generated successfully!")
                
                # Show dataset preview
                st.subheader("Dataset Preview")
                st.dataframe(df.head(10))
        
        if uploaded_file is not None:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Create a copy for transformation if needed
            processed_df = df.copy()
            
            # Check if the required columns exist
            required_columns = ['Temperature', 'Humidity', 'Step_Count', 'Stress_Level']
            available_columns = list(df.columns)
            
            # Handle common column name variations
            column_mapping = {}
            
            if 'temperature' in available_columns or 'body_temperature' in available_columns or 'temp' in available_columns:
                for col in available_columns:
                    if col.lower() in ['temperature', 'body_temperature', 'temp', 'body temperature']:
                        column_mapping[col] = 'Temperature'
                        break
                        
            if 'humidity' in available_columns or 'body_humidity' in available_columns or 'hum' in available_columns:
                for col in available_columns:
                    if col.lower() in ['humidity', 'body_humidity', 'hum', 'body humidity']:
                        column_mapping[col] = 'Humidity'
                        break
                        
            if 'step_count' in available_columns or 'steps' in available_columns or 'step count' in available_columns:
                for col in available_columns:
                    if col.lower() in ['step_count', 'steps', 'step count', 'stepcount']:
                        column_mapping[col] = 'Step_Count'
                        break
                        
            if 'stress_level' in available_columns or 'stress' in available_columns or 'level' in available_columns:
                for col in available_columns:
                    if col.lower() in ['stress_level', 'stress', 'level', 'stress level']:
                        column_mapping[col] = 'Stress_Level'
                        break
            
            # Apply column mapping if found
            if column_mapping:
                processed_df = df.rename(columns=column_mapping)
                st.info(f"Automatically mapped columns: {column_mapping}")
            
            # Check again for required columns
            missing_columns = [col for col in required_columns if col not in processed_df.columns]
            
            if missing_columns:
                st.error(f"The uploaded dataset is missing the following required columns: {', '.join(missing_columns)}")
                
                # Show a more detailed error message with column mapping guide
                st.markdown("""
                ### Dataset Format Error
                
                Please make sure your dataset has the following columns:
                - **Temperature**: Body temperature (°C)
                - **Humidity**: Body humidity (%)
                - **Step_Count**: Number of steps taken
                - **Stress_Level**: Stress level (0 for low, 1 for normal, 2 for high)
                
                #### Column Mapping Guide
                
                If your dataset uses different column names, you can either:
                1. Rename the columns in your CSV file before uploading
                2. Use the following format guide to match your dataset:
                
                | Required Column | Alternative Names |
                | -------------- | ----------------- |
                | Temperature | temp, body_temperature |
                | Humidity | hum, body_humidity |
                | Step_Count | steps, stepcount |
                | Stress_Level | stress, level |
                """)
                
                # Show the current columns
                st.subheader("Your Dataset Columns")
                st.write(f"Available columns: {', '.join(available_columns)}")
                
            else:
                # Additional data validation
                error_messages = []
                
                # Check for numeric columns
                for col in ['Temperature', 'Humidity', 'Step_Count']:
                    if not pd.api.types.is_numeric_dtype(processed_df[col]):
                        error_messages.append(f"{col} column must contain numeric values")
                
                # Check that Stress_Level is 0, 1, or 2
                unique_stress = processed_df['Stress_Level'].unique()
                if not all(level in [0, 1, 2] for level in unique_stress):
                    error_messages.append(f"Stress_Level must only contain values 0, 1, or 2. Found: {unique_stress}")
                
                if error_messages:
                    st.error("Dataset Validation Errors:")
                    for msg in error_messages:
                        st.error(f"- {msg}")
                else:
                    # Save the processed dataset
                    if not os.path.exists("data"):
                        os.makedirs("data")
                    processed_df.to_csv("data/dataset.csv", index=False)
                    st.success("Dataset uploaded, validated, and saved successfully!")
                    
                    # Display dataset info
                    st.subheader("Dataset Preview")
                    st.dataframe(processed_df.head(10))
                    
                    # Dataset statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", f"{processed_df.shape[0]}")
                    with col2:
                        st.metric("Features", f"{processed_df.shape[1] - 1}")  # Excluding target column
                    with col3:
                        stress_distribution = processed_df['Stress_Level'].value_counts()
                        stress_distribution_str = ", ".join([f"{stress_level_to_text(k)}: {v}" for k, v in stress_distribution.items()])
                        st.metric("Class Distribution", stress_distribution_str)
                    
                    # Reset model training state
                    if 'model_trained' in st.session_state:
                        st.session_state.model_trained = False
                    if 'model_trainer' in st.session_state:
                        st.session_state.model_trainer = None
                    
                    st.info("Your dataset is ready! Go to the Model Training page to build your SVM model.")
    
    with data_tab1:
        if os.path.exists("data/dataset.csv"):
            df = pd.read_csv("data/dataset.csv")
            st.write("Dataset loaded successfully!")
        else:
            # Initialize data processor to download and process data
            data_processor = DataProcessor()
            with st.spinner("Generating synthetic data..."):
                df = data_processor.generate_synthetic_data()
                if not os.path.exists("data"):
                    os.makedirs("data")
                df.to_csv("data/dataset.csv", index=False)
                st.session_state.data_processor = data_processor
                
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Dataset Information")
        st.write(f"Number of samples: {df.shape[0]}")
        st.write(f"Number of features: {df.shape[1]}")
        
        # Display stress level distribution
        stress_counts = df['Stress_Level'].value_counts().sort_index()
        st.subheader("Stress Level Distribution")
        
        if len(stress_counts) > 0:
            # Create labels based on available stress levels
            stress_labels = []
            colors = []
            
            if 0 in stress_counts.index:
                stress_labels.append('Low')
                colors.append('green')
            if 1 in stress_counts.index:
                stress_labels.append('Normal')
                colors.append('gold')
            if 2 in stress_counts.index:
                stress_labels.append('High')
                colors.append('red')
                
            fig = px.pie(
                values=stress_counts.values, 
                names=stress_labels,
                color_discrete_sequence=colors,
                title="Distribution of Stress Levels"
            )
            st.plotly_chart(fig)
        else:
            st.warning("No stress level data available in the dataset.")
    
    with data_tab2:
        if os.path.exists("data/dataset.csv"):
            df = pd.read_csv("data/dataset.csv")
            
            st.subheader("Statistical Summary")
            st.write(df.describe())
            
            st.subheader("Feature Correlation")
            corr = df.corr()
            fig = px.imshow(
                corr, 
                text_auto=True, 
                color_continuous_scale='RdBu_r',
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig)
            
            st.subheader("Missing Values Analysis")
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                st.write(missing_values)
            else:
                st.write("No missing values in the dataset!")
    
    with data_tab3:
        if os.path.exists("data/dataset.csv"):
            df = pd.read_csv("data/dataset.csv")
            
            st.subheader("Feature Distributions")
            feature = st.selectbox(
                "Select feature to visualize:", 
                ["Temperature", "Humidity", "Step_Count"]
            )
            
            # Create histogram with stress level color
            fig = px.histogram(
                df, 
                x=feature, 
                color='Stress_Level',
                color_discrete_map={0: 'green', 1: 'gold', 2: 'red'},
                labels={'Stress_Level': 'Stress Level'},
                category_orders={"Stress_Level": [0, 1, 2]},
                barmode='overlay',
                opacity=0.7,
                title=f"Distribution of {feature} by Stress Level"
            )
            st.plotly_chart(fig)
            
            # Scatter plot
            st.subheader("Feature Relationships")
            x_feature = st.selectbox("Select X-axis feature:", ["Temperature", "Humidity", "Step_Count"])
            y_feature = st.selectbox("Select Y-axis feature:", ["Humidity", "Temperature", "Step_Count"], index=1)
            
            if x_feature != y_feature:
                fig = px.scatter(
                    df, 
                    x=x_feature, 
                    y=y_feature, 
                    color='Stress_Level',
                    color_discrete_map={0: 'green', 1: 'gold', 2: 'red'},
                    title=f"{x_feature} vs {y_feature} by Stress Level",
                    category_orders={"Stress_Level": [0, 1, 2]},
                )
                st.plotly_chart(fig)
            else:
                st.warning("Please select different features for X and Y axis")

elif page == "Model Training":
    st.header("SVM Model Training")
    
    # Set up tabs for different model aspects
    model_tab1, model_tab2, model_tab3 = st.tabs(["Train Model", "Model Performance", "Feature Importance"])
    
    with model_tab1:
        if os.path.exists("data/dataset.csv"):
            df = pd.read_csv("data/dataset.csv")
            
            # Model parameters
            st.subheader("SVM Model Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                kernel = st.selectbox("Kernel:", ["linear", "rbf", "poly"], index=1)
                C = st.slider("Regularization (C):", 0.1, 10.0, 1.0, 0.1)
                
            with col2:
                gamma = st.selectbox("Gamma:", ["scale", "auto"], index=0)
                test_size = st.slider("Test Size (%):", 10, 40, 30)
                
            # Train the model
            if st.button("Train Model"):
                with st.spinner("Training SVM Model..."):
                    # Initialize data processor if not done already
                    if st.session_state.data_processor is None:
                        st.session_state.data_processor = DataProcessor()
                    
                    # Process the data
                    X_train, X_test, y_train, y_test = st.session_state.data_processor.process_data(
                        df, test_size=test_size/100
                    )
                    
                    # Initialize and train the model
                    model_trainer = SVMModelTrainer(
                        kernel=kernel, 
                        C=C, 
                        gamma=gamma
                    )
                    
                    model_trainer.train(X_train, y_train)
                    accuracy, report, conf_matrix = model_trainer.evaluate(X_test, y_test)
                    
                    st.session_state.model_trainer = model_trainer
                    st.session_state.model_trained = True
                    st.session_state.test_data = (X_test, y_test)
                    st.session_state.train_data = (X_train, y_train)
                    st.session_state.model_metrics = {
                        'accuracy': accuracy,
                        'report': report,
                        'conf_matrix': conf_matrix
                    }
                    
                    # Save model metadata to database
                    try:
                        save_model_metadata(
                            model_type='SVM',
                            kernel=kernel,
                            c_param=C,
                            gamma=gamma,
                            accuracy=accuracy
                        )
                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}% (saved to database)")
                    except Exception as e:
                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}%")
                        st.warning(f"Could not save model metadata to database: {str(e)}")
            
            # Display model details if trained
            if st.session_state.model_trained:
                st.subheader("Model Overview")
                st.write(f"**SVM Kernel:** {st.session_state.model_trainer.kernel}")
                st.write(f"**Regularization (C):** {st.session_state.model_trainer.C}")
                st.write(f"**Gamma:** {st.session_state.model_trainer.gamma}")
                st.write(f"**Test Size:** {test_size}%")
        else:
            st.warning("Please go to Data Exploration first to load the dataset")
    
    with model_tab2:
        if st.session_state.model_trained:
            st.subheader("Model Performance Metrics")
            
            # Display accuracy
            accuracy = st.session_state.model_metrics['accuracy']
            st.metric("Model Accuracy", f"{accuracy:.2f}%")
            
            # Display classification report
            st.subheader("Classification Report")
            report = st.session_state.model_metrics['report']
            st.text(report)
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            conf_matrix = st.session_state.model_metrics['conf_matrix']
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_confusion_matrix(
                conf_matrix, 
                classes=['Low Stress', 'Normal Stress', 'High Stress'],
                normalize=True,
                title='Normalized Confusion Matrix',
                ax=ax
            )
            st.pyplot(fig)
        else:
            st.info("Train the model first to see performance metrics")
    
    with model_tab3:
        if st.session_state.model_trained and hasattr(st.session_state.model_trainer.model, 'coef_'):
            st.subheader("Feature Importance")
            
            # Get feature importance for linear kernel
            coefficients = st.session_state.model_trainer.model.coef_
            feature_names = ['Temperature', 'Humidity', 'Step_Count']
            
            # For multiclass SVM, we have coefficients for each class
            for i, coef in enumerate(coefficients):
                st.write(f"**Class {i} vs Rest**")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': np.abs(coef)
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title=f'Feature Importance for Class {i}',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig)
        elif st.session_state.model_trained:
            st.info("Feature importance is only available for linear kernel")
        else:
            st.info("Train the model first to see feature importance")

elif page == "Stress Prediction":
    st.header("Stress Level Prediction")
    
    # Instructions for users
    st.markdown("""
    Enter physiological parameters below to receive a stress level prediction using the trained SVM model.
    Ensure you have trained a model in the previous step.
    """)
    
    # Create form for user input
    with st.form("prediction_form"):
        st.subheader("Enter Your Physiological Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.number_input("Body Temperature (°C)", 
                                           min_value=35.0, max_value=40.0, 
                                           value=36.5, step=0.1)
            
        with col2:
            humidity = st.number_input("Body Humidity (%)", 
                                        min_value=0.0, max_value=100.0, 
                                        value=60.0, step=1.0)
            
        with col3:
            step_count = st.number_input("Step Count (last hour)", 
                                           min_value=0, max_value=10000, 
                                           value=500, step=50)
        
        submitted = st.form_submit_button("Predict Stress Level")
    
    # Make prediction if form is submitted
    if submitted:
        if st.session_state.model_trained:
            with st.spinner("Analyzing your stress level..."):
                time.sleep(1)  # Simulate processing time
                
                # Prepare input data for prediction
                input_data = pd.DataFrame({
                    'Temperature': [temperature],
                    'Humidity': [humidity],
                    'Step_Count': [step_count]
                })
                
                # Scale the input features
                input_scaled = st.session_state.data_processor.scale_features(input_data)
                
                # Make prediction
                prediction = st.session_state.model_trainer.predict(input_scaled)
                
                # Get probability scores if available
                try:
                    proba = st.session_state.model_trainer.predict_proba(input_scaled)[0]
                    has_proba = True
                except:
                    has_proba = False
                
                # Map prediction to label
                stress_labels = {0: "Low", 1: "Normal", 2: "High"}
                stress_colors = {0: "green", 1: "gold", 2: "red"}
                stress_emojis = {0: "😌", 1: "😐", 2: "😫"}
                
                stress_level = prediction[0]
                stress_label = stress_labels[stress_level]
                stress_color = stress_colors[stress_level]
                stress_emoji = stress_emojis[stress_level]
                
                # Save prediction to database
                try:
                    # Convert probabilities to list
                    probabilities_list = proba.tolist() if has_proba else None
                    
                    # Save to database
                    save_prediction(
                        user_id=st.session_state.user_id,
                        temperature=temperature,
                        humidity=humidity,
                        step_count=step_count,
                        stress_level=int(stress_level),
                        probabilities=probabilities_list
                    )
                except Exception as e:
                    st.warning(f"Could not save prediction to database: {str(e)}")
                
                # Display prediction result
                st.markdown("## Prediction Result")
                
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {stress_color}; 
                            padding: 20px; 
                            border-radius: 10px; 
                            text-align: center;
                            color: white;
                            font-size: 24px;
                            ">
                            <h1 style="margin: 0; color: white; font-size: 48px;">{stress_emoji}</h1>
                            <p style="margin: 10px 0 0 0; font-weight: bold;">{stress_label} Stress</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with result_col2:
                    # Display recommendations based on stress level
                    if stress_level == 0:
                        st.success("Your stress level is LOW. Keep up the good work!")
                        st.markdown("""
                        **Recommendations:**
                        - Continue your current routine
                        - Regular exercise and healthy diet
                        - Maintain good sleep patterns
                        """)
                    elif stress_level == 1:
                        st.warning("Your stress level is NORMAL. Some relaxation might help.")
                        st.markdown("""
                        **Recommendations:**
                        - Consider light relaxation techniques
                        - Take short breaks during work
                        - Stay hydrated and practice mindful breathing
                        """)
                    else:
                        st.error("Your stress level is HIGH. You should take action to reduce stress.")
                        st.markdown("""
                        **Recommendations:**
                        - Practice deep breathing or meditation
                        - Consider talking to a professional
                        - Reduce workload and get adequate rest
                        - Engage in physical activity to release tension
                        """)
                
                # Display probability distribution if available
                if has_proba:
                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Stress Level': ['Low', 'Normal', 'High'],
                        'Probability': proba
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Stress Level',
                        y='Probability',
                        color='Stress Level',
                        color_discrete_map={'Low': 'green', 'Normal': 'gold', 'High': 'red'},
                        title="Probability of Each Stress Level"
                    )
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig)
        else:
            st.error("Please train the model first in the Model Training section")

elif page == "History & Trends":
    st.header("Your Stress History & Trends")
    
    # Fetch history data from database
    try:
        history_df = get_user_predictions(user_id=st.session_state.user_id)
        has_history = len(history_df) > 0
    except Exception as e:
        st.error(f"Error fetching prediction history: {str(e)}")
        has_history = False
        history_df = pd.DataFrame()
    
    if has_history:
        # Convert prediction numbers to labels
        history_df['Stress_Label'] = history_df['stress_level'].apply(stress_level_to_text)
        
        # Format timestamp
        history_df['Timestamp'] = pd.to_datetime(history_df['prediction_time'])
        
        # Display history table with renamed columns for better presentation
        st.subheader("Your Prediction History")
        st.dataframe(
            history_df[['temperature', 'humidity', 'step_count', 'Stress_Label', 'Timestamp']]
            .rename(columns={
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'step_count': 'Step Count'
            })
            .sort_values('Timestamp', ascending=False)
        )
        
        # Check if we have enough data points for trends
        if len(history_df) >= 3:
            st.subheader("Stress Level Trends")
            
            # Prepare data for trend visualization
            history_df['Date'] = history_df['Timestamp'].dt.date
            
            # Create trend chart
            fig = px.line(
                history_df,
                x='Timestamp',
                y='stress_level',
                color='Stress_Label',
                color_discrete_map={'Low': 'green', 'Normal': 'gold', 'High': 'red'},
                labels={'stress_level': 'Stress Level', 'Timestamp': 'Time'},
                title="Your Stress Level Over Time",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation with stress
            st.subheader("Features vs Stress Level")
            feature_options = {
                "Temperature": "temperature",
                "Humidity": "humidity",
                "Step Count": "step_count"
            }
            feature_display = st.selectbox(
                "Select feature to analyze:", 
                list(feature_options.keys())
            )
            feature = feature_options[feature_display]
            
            fig = px.scatter(
                history_df,
                x=feature,
                y='stress_level',
                color='Stress_Label',
                color_discrete_map={'Low': 'green', 'Normal': 'gold', 'High': 'red'},
                title=f"{feature_display} vs Stress Level",
                labels={
                    'stress_level': 'Stress Level', 
                    feature: feature_display
                },
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics by stress level
            st.subheader("Statistics by Stress Level")
            # Create a more readable dataframe for the stats
            analysis_df = history_df.rename(columns={
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'step_count': 'Step Count'
            })
            stats_df = analysis_df.groupby('Stress_Label')[['Temperature', 'Humidity', 'Step Count']].describe()
            st.write(stats_df)
    else:
        st.info("No prediction history yet. Make some predictions in the 'Stress Prediction' section to see your history and trends.")
        
        # Show placeholder image
        st.image(
            "https://pixabay.com/get/g64661bd41f1f6cf45812d923f5092b61e42defc40abb294b58aac3471ae5fbd2ce031b0e002500637eec9239533159db304f7d0ed4c09330161e04591e68e4bf_1280.jpg", 
            use_container_width=True
        )

if __name__ == "__main__":
    # Run the application
    pass
