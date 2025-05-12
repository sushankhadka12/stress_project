import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import itertools

class Visualizer:
    """
    Class for creating visualizations for the stress level detection system.
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        pass
    
    def plot_feature_distributions(self, df, feature_names):
        """
        Plot the distribution of each feature.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
        feature_names : list
            List of feature names to plot
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Figure with feature distributions
        """
        fig = px.histogram(
            df, 
            x=feature_names[0],
            color="Stress_Level", 
            barmode="overlay",
            opacity=0.7,
            title=f"Distribution of {feature_names[0]} by Stress Level"
        )
        
        return fig
    
    def plot_feature_correlations(self, df, feature_names):
        """
        Plot the correlation matrix for features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset
        feature_names : list
            List of feature names to include
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Figure with correlation matrix
        """
        corr = df[feature_names].corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix"
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        Plot the confusion matrix.
        
        Parameters:
        -----------
        cm : numpy.ndarray
            Confusion matrix
        classes : list
            List of class names
        normalize : bool
            Whether to normalize the confusion matrix
        title : str
            Title for the plot
        cmap : matplotlib.colors.Colormap
            Colormap for the plot
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure with confusion matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=14)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
        plt.yticks(tick_marks, classes, fontsize=12)
        
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
        
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_prediction_probabilities(self, probabilities, class_names):
        """
        Plot prediction probabilities.
        
        Parameters:
        -----------
        probabilities : numpy.ndarray
            Prediction probabilities for each class
        class_names : list
            List of class names
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Figure with prediction probabilities
        """
        df = pd.DataFrame({
            'Class': class_names,
            'Probability': probabilities
        })
        
        fig = px.bar(
            df,
            x='Class',
            y='Probability',
            color='Class',
            color_discrete_map={'Low': 'green', 'Normal': 'gold', 'High': 'red'},
            title="Prediction Probabilities"
        )
        
        fig.update_layout(yaxis_range=[0, 1])
        
        return fig
    
    def plot_feature_importance(self, coefficients, feature_names):
        """
        Plot feature importance for linear SVM.
        
        Parameters:
        -----------
        coefficients : numpy.ndarray
            Coefficients from linear SVM
        feature_names : list
            List of feature names
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Figure with feature importance
        """
        importance = np.abs(coefficients)
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance[0]
        })
        
        df = df.sort_values('Importance', ascending=False)
        
        fig = px.bar(
            df,
            x='Feature',
            y='Importance',
            color='Importance',
            title="Feature Importance"
        )
        
        return fig
    
    def plot_stress_history(self, history_df):
        """
        Plot stress level history over time.
        
        Parameters:
        -----------
        history_df : pandas.DataFrame
            DataFrame with prediction history
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Figure with stress history
        """
        stress_map = {0: "Low", 1: "Normal", 2: "High"}
        history_df['Stress_Label'] = history_df['Predicted_Stress'].map(stress_map)
        
        fig = px.line(
            history_df,
            x='Timestamp',
            y='Predicted_Stress',
            color='Stress_Label',
            color_discrete_map={'Low': 'green', 'Normal': 'gold', 'High': 'red'},
            title="Stress Level History",
            markers=True
        )
        
        return fig
