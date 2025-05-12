import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import json
from dotenv import load_dotenv
load_dotenv()
# Get database connection URL from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL environment variable not set")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Create base class for SQLAlchemy models
Base = declarative_base()

class User(Base):
    """User model for storing user information"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship with UserPrediction
    predictions = relationship("UserPrediction", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, name={self.name})>"

class UserPrediction(Base):
    """UserPrediction model for storing stress level predictions"""
    __tablename__ = 'user_predictions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    step_count = Column(Integer, nullable=False)
    stress_level = Column(Integer, nullable=False)
    prediction_time = Column(DateTime, default=datetime.datetime.utcnow)
    probabilities = Column(String)  # Store as JSON string
    
    # Relationship with User
    user = relationship("User", back_populates="predictions")
    
    def __repr__(self):
        return f"<UserPrediction(id={self.id}, stress_level={self.stress_level})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        probs = None
        if self.probabilities:
            try:
                probs = json.loads(self.probabilities)
            except:
                probs = None
                
        return {
            'id': self.id,
            'user_id': self.user_id,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'step_count': self.step_count,
            'stress_level': self.stress_level,
            'prediction_time': self.prediction_time,
            'probabilities': probs
        }

class ModelMetadata(Base):
    """ModelMetadata model for storing model training information"""
    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String, nullable=False)
    kernel = Column(String, nullable=True)
    c_param = Column(Float, nullable=True)
    gamma = Column(String, nullable=True)
    accuracy = Column(Float, nullable=True)
    trained_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelMetadata(id={self.id}, model_type={self.model_type}, accuracy={self.accuracy})>"

# Create tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(engine)

def save_prediction(user_id, temperature, humidity, step_count, stress_level, probabilities=None):
    """
    Save a prediction to the database
    
    Parameters:
    -----------
    user_id : int
        User ID
    temperature : float
        Body temperature
    humidity : float
        Humidity
    step_count : int
        Step count
    stress_level : int
        Predicted stress level (0, 1, or 2)
    probabilities : list or None
        Prediction probabilities for each class
    
    Returns:
    --------
    user_prediction : UserPrediction
        Created UserPrediction instance
    """
    # Convert probabilities to JSON string if provided
    probs_str = json.dumps(probabilities) if probabilities is not None else None
    
    # Create new UserPrediction instance
    user_prediction = UserPrediction(
        user_id=user_id,
        temperature=temperature,
        humidity=humidity,
        step_count=step_count,
        stress_level=stress_level,
        probabilities=probs_str
    )
    
    # Add to session and commit
    session.add(user_prediction)
    session.commit()
    
    return user_prediction

def get_user_predictions(user_id=None, limit=100):
    """
    Get user predictions from the database
    
    Parameters:
    -----------
    user_id : int or None
        User ID. If None, get all predictions
    limit : int
        Maximum number of predictions to return
    
    Returns:
    --------
    predictions_df : pandas.DataFrame
        DataFrame with predictions
    """
    query = session.query(UserPrediction)
    
    if user_id is not None:
        query = query.filter(UserPrediction.user_id == user_id)
    
    # Order by time and limit
    predictions = query.order_by(UserPrediction.prediction_time.desc()).limit(limit).all()
    
    # Convert to list of dictionaries
    predictions_list = [p.to_dict() for p in predictions]
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions_list)
    
    return predictions_df

def save_model_metadata(model_type, kernel, c_param, gamma, accuracy):
    """
    Save model metadata to the database
    
    Parameters:
    -----------
    model_type : str
        Type of model (e.g., 'SVM')
    kernel : str
        Kernel type
    c_param : float
        C parameter
    gamma : str or float
        Gamma parameter
    accuracy : float
        Model accuracy
    
    Returns:
    --------
    model_metadata : ModelMetadata
        Created ModelMetadata instance
    """
    # Create new ModelMetadata instance
    model_metadata = ModelMetadata(
        model_type=model_type,
        kernel=kernel,
        c_param=c_param,
        gamma=str(gamma),
        accuracy=accuracy
    )
    
    # Add to session and commit
    session.add(model_metadata)
    session.commit()
    
    return model_metadata

def get_latest_model_metadata():
    """
    Get the latest model metadata from the database
    
    Returns:
    --------
    model_metadata : ModelMetadata or None
        Latest ModelMetadata instance
    """
    return session.query(ModelMetadata).order_by(ModelMetadata.trained_at.desc()).first()

def get_or_create_user(name=None, email=None):
    """
    Get or create a user in the database
    
    Parameters:
    -----------
    name : str or None
        User name
    email : str or None
        User email
    
    Returns:
    --------
    user : User
        User instance
    """
    # Check if user exists with given email
    user = None
    if email is not None:
        user = session.query(User).filter(User.email == email).first()
    
    # If no user found, create new user
    if user is None:
        user = User(name=name, email=email)
        session.add(user)
        session.commit()
    
    return user

# Initialize database tables
init_db()