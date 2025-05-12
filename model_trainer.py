import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

class SVMModelTrainer:
    """
    Class for training and evaluating SVM models for stress level prediction.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize the SVM model trainer with specified parameters.
        
        Parameters:
        -----------
        kernel : str
            Kernel type to be used in the SVM algorithm
        C : float
            Regularization parameter
        gamma : str or float
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,
            random_state=42
        )
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train the SVM model.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target values
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Predicted stress levels
        """
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Please train the model first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for each class.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Probability estimates
        """
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Please train the model first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Testing features
        y_test : numpy.ndarray
            Testing target values
            
        Returns:
        --------
        accuracy : float
            Accuracy score as a percentage
        report : str
            Classification report
        conf_matrix : numpy.ndarray
            Confusion matrix
        """
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Please train the model first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=['Low', 'Normal', 'High'])
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return accuracy, report, conf_matrix
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Tune SVM hyperparameters using grid search.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target values
        X_val : numpy.ndarray
            Validation features
        y_val : numpy.ndarray
            Validation target values
            
        Returns:
        --------
        best_params : dict
            Best hyperparameters found
        best_score : float
            Best validation score
        """
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['linear', 'rbf', 'poly']
        }
        
        # Create grid search
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        
        # Update model with best parameters
        self.kernel = best_params['kernel']
        self.C = best_params['C']
        self.gamma = best_params['gamma']
        
        # Create new model with best parameters
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,
            random_state=42
        )
        
        # Train model with best parameters
        self.train(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = self.predict(X_val)
        best_score = accuracy_score(y_val, y_val_pred) * 100
        
        return best_params, best_score
