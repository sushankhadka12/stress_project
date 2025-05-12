import requests
import numpy as np
import matplotlib.pyplot as plt
import itertools
from io import BytesIO
from PIL import Image
import pandas as pd
import seaborn as sns
import streamlit as st

def load_image_from_url(url):
    """
    Load an image from a URL.
    
    Parameters:
    -----------
    url : str
        URL of the image
        
    Returns:
    --------
    PIL.Image
        Loaded image
    """
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, ax=None):
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
    ax : matplotlib.axes.Axes
        Axes for the plot
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        Axes with confusion matrix plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap=cmap, 
        cbar=True, 
        ax=ax, 
        xticklabels=classes, 
        yticklabels=classes
    )
    
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    return ax

def create_color_map():
    """
    Create a color map for stress levels.
    
    Returns:
    --------
    dict
        Color map for stress levels
    """
    return {
        0: "green",   # Low stress
        1: "gold",    # Normal stress
        2: "red"      # High stress
    }

def stress_level_to_text(level):
    """
    Convert numeric stress level to text.
    
    Parameters:
    -----------
    level : int
        Numeric stress level (0, 1, or 2)
        
    Returns:
    --------
    str
        Text representation of stress level
    """
    stress_map = {
        0: "Low",
        1: "Normal",
        2: "High"
    }
    
    return stress_map.get(level, "Unknown")

def get_recommendations(stress_level):
    """
    Get recommendations based on stress level.
    
    Parameters:
    -----------
    stress_level : int
        Stress level (0, 1, or 2)
        
    Returns:
    --------
    list
        List of recommendations
    """
    recommendations = {
        0: [  # Low stress
            "Continue your current routine",
            "Regular exercise and healthy diet",
            "Maintain good sleep patterns"
        ],
        1: [  # Normal stress
            "Consider light relaxation techniques",
            "Take short breaks during work",
            "Stay hydrated and practice mindful breathing"
        ],
        2: [  # High stress
            "Practice deep breathing or meditation",
            "Consider talking to a professional",
            "Reduce workload and get adequate rest",
            "Engage in physical activity to release tension"
        ]
    }
    
    return recommendations.get(stress_level, [])

def format_time(seconds):
    """
    Format time in seconds to a readable string.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
        
    Returns:
    --------
    str
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
