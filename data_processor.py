import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import tempfile

class DataProcessor:
    """
    Class for handling data processing tasks including loading, 
    preprocessing, and splitting the dataset.
    """
    
    def __init__(self):
        """Initialize the data processor with empty attributes"""
        self.scaler = StandardScaler()
        self.features = ['Temperature', 'Humidity', 'Step_Count']
        self.target = 'Stress_Level'
    
    def download_dataset(self, dataset_name=None):
        """
        Download dataset from Kaggle using the Kaggle API.
        
        Parameters:
        -----------
        dataset_name : str
            The name of the dataset on Kaggle in the format 'username/dataset-name'
            If None, will use a default stress detection dataset
            
        Returns:
        --------
        pandas.DataFrame
            Downloaded dataset or synthetic data if download fails
        """
        if dataset_name is None:
            # Default dataset - Stress Detection Dataset
            dataset_name = "laavanya/stress-detection-dataset"
        
        try:
            import os
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Check if Kaggle credentials are available
            kaggle_username = os.environ.get('KAGGLE_USERNAME')
            kaggle_key = os.environ.get('KAGGLE_KEY')
            
            if not kaggle_username or not kaggle_key:
                print("Kaggle credentials not found. Using synthetic data instead.")
                return self.generate_synthetic_data()
            
            # Create data directory if it doesn't exist
            if not os.path.exists("data"):
                os.makedirs("data")
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download the dataset
            print(f"Downloading dataset: {dataset_name}")
            api.dataset_download_files(dataset_name, path="data", unzip=True)
            
            # Find the CSV file in the data directory
            csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]
            
            if not csv_files:
                print("No CSV files found in downloaded dataset. Using synthetic data instead.")
                return self.generate_synthetic_data()
            
            # Use the first CSV file found
            csv_file = csv_files[0]
            df = pd.read_csv(f"data/{csv_file}")
            
            # Check if dataset has required columns
            required_columns = ['Temperature', 'Humidity', 'Step_Count', 'Stress_Level']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Dataset missing required columns: {missing_columns}. Using synthetic data instead.")
                return self.generate_synthetic_data()
                
            print("Dataset downloaded successfully")
            return df
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Using synthetic data instead")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self, n_samples=2001):
        """
        Generate synthetic data for stress level prediction.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        pandas.DataFrame
            Synthetic dataset
        """
        np.random.seed(42)
        
        # Generate features
        temperature = np.random.normal(36.5, 1.0, n_samples)  # Body temperature in Celsius
        humidity = np.random.uniform(40, 80, n_samples)       # Body humidity percentage
        step_count = np.random.randint(0, 2000, n_samples)    # Steps taken in a period
        
        # Create stress levels based on conditions
        stress_level = np.zeros(n_samples, dtype=int)
        
        # Low stress: normal temp, moderate humidity, moderate steps
        condition_low = (
            (temperature > 36.0) & (temperature < 37.0) &
            (humidity > 45) & (humidity < 65) &
            (step_count > 300) & (step_count < 1000)
        )
        
        # High stress: high or low temp, very high or low humidity, very low or high steps
        condition_high = (
            (temperature < 35.5) | (temperature > 37.5) |
            (humidity < 40) | (humidity > 75) |
            (step_count < 100) | (step_count > 1500)
        )
        
        # Assign stress levels (0: Low, 1: Normal, 2: High)
        stress_level[condition_low] = 0
        stress_level[~condition_low & ~condition_high] = 1
        stress_level[condition_high] = 2
        
        # Ensure at least 30% of each class for balanced dataset
        n_class = n_samples // 3
        indices_0 = np.where(stress_level == 0)[0]
        indices_1 = np.where(stress_level == 1)[0]
        indices_2 = np.where(stress_level == 2)[0]
        
        if len(indices_0) < n_class:
            more_needed = n_class - len(indices_0)
            extra_indices = np.random.choice(np.where(stress_level != 0)[0], more_needed)
            stress_level[extra_indices] = 0
            
        if len(indices_1) < n_class:
            more_needed = n_class - len(indices_1)
            extra_indices = np.random.choice(np.where(stress_level != 1)[0], more_needed)
            stress_level[extra_indices] = 1
            
        if len(indices_2) < n_class:
            more_needed = n_class - len(indices_2)
            extra_indices = np.random.choice(np.where(stress_level != 2)[0], more_needed)
            stress_level[extra_indices] = 2
        
        # Create DataFrame
        df = pd.DataFrame({
            'Temperature': temperature,
            'Humidity': humidity,
            'Step_Count': step_count,
            'Stress_Level': stress_level
        })
        
        # Add some missing values to simulate real data (about 2%)
        for col in ['Temperature', 'Humidity', 'Step_Count']:
            mask = np.random.random(n_samples) < 0.02
            df.loc[mask, col] = np.nan
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataset
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with handled missing values
        """
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            # Impute missing values using mean for numerical features
            for feature in self.features:
                df[feature].fillna(df[feature].mean(), inplace=True)
        
        return df
    
    def scale_features(self, df):
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataset with features to scale
            
        Returns:
        --------
        numpy.ndarray
            Scaled features
        """
        # Get feature columns
        X = df[self.features].values
        
        # If the scaler is not fitted yet, fit it (for training data)
        if not hasattr(self.scaler, 'mean_'):
            return self.scaler.fit_transform(X)
        
        # Otherwise, use transform only (for new data)
        return self.scaler.transform(X)
    
    def process_data(self, df, test_size=0.3, random_state=42):
        """
        Process the data and split into training and testing sets.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataset
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : numpy.ndarray
            Processed and split data
        """
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Split features and target
        X = df[self.features].values
        y = df[self.target].values
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
