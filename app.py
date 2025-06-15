from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
warnings.filterwarnings('ignore')
import joblib
import time
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables to store data and models
train_data = None
test_data = None
processed_data = None
models = {}
model_results = {}

# Global variables for model persistence and timing
MODEL_DIR = 'saved_models'
training_start_time = None
training_duration = None

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_data():
    """Load the training and test datasets"""
    global train_data, test_data
    try:
        # In a real scenario, you'd load from uploaded files or database
        # For now, we'll create sample data based on the structure provided
        train_data = pd.read_csv('train.csv') if os.path.exists('train.csv') else create_sample_data(True)
        test_data = pd.read_csv('test.csv') if os.path.exists('test.csv') else create_sample_data(False)
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def create_sample_data(include_target=True):
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 500 if include_target else 250  # Reduced sample size for faster training
    
    # Create feature columns based on the dataset description
    data = {
        'id': range(n_samples),
        'ps_ind_01': np.random.randint(0, 6, n_samples),
        'ps_ind_02_cat': np.random.randint(1, 5, n_samples),
        'ps_ind_03': np.random.randint(0, 12, n_samples),
        'ps_ind_04_cat': np.random.randint(0, 2, n_samples),
        'ps_ind_05_cat': np.random.choice([0, 1, 2, 4, 6, -1], n_samples),
        'ps_ind_06_bin': np.random.randint(0, 2, n_samples),
        'ps_ind_07_bin': np.random.randint(0, 2, n_samples),
        'ps_ind_08_bin': np.random.randint(0, 2, n_samples),
        'ps_ind_09_bin': np.random.randint(0, 2, n_samples),
        'ps_ind_14': np.random.randint(0, 15, n_samples),
        'ps_ind_15': np.random.randint(0, 15, n_samples),
        'ps_ind_16_bin': np.random.randint(0, 2, n_samples),
        'ps_ind_17_bin': np.random.randint(0, 2, n_samples),
        'ps_ind_18_bin': np.random.randint(0, 2, n_samples),
        'ps_reg_01': np.random.uniform(0, 1, n_samples),
        'ps_reg_02': np.random.uniform(0, 2, n_samples),
        'ps_reg_03': np.random.uniform(-1, 3, n_samples),
        'ps_car_01_cat': np.random.randint(6, 12, n_samples),
        'ps_car_02_cat': np.random.choice([0, 1, -1], n_samples),
        'ps_car_03_cat': np.random.choice([-1, 0, 1, 2], n_samples),
        'ps_car_04_cat': np.random.randint(0, 10, n_samples),
        'ps_car_05_cat': np.random.choice([-1, 0, 1], n_samples),
        'ps_car_06_cat': np.random.randint(0, 18, n_samples),
        'ps_car_07_cat': np.random.choice([-1, 0, 1], n_samples),
        'ps_car_08_cat': np.random.randint(0, 2, n_samples),
        'ps_car_09_cat': np.random.randint(0, 6, n_samples),
        'ps_car_11': np.random.randint(0, 105, n_samples),
        'ps_car_12': np.random.uniform(0, 1, n_samples),
        'ps_car_13': np.random.uniform(0, 4, n_samples),
        'ps_car_14': np.random.uniform(-1, 1, n_samples),
        'ps_car_15': np.random.uniform(0, 4, n_samples),
    }
    
    # Add calculated features
    for i in range(1, 21):
        if i <= 14:
            data[f'ps_calc_{i:02d}'] = np.random.uniform(0, 1, n_samples)
        else:
            data[f'ps_calc_{i}_bin'] = np.random.randint(0, 2, n_samples)
    
    if include_target:
        # Create target with class imbalance (typical in insurance)
        data['target'] = np.random.choice([0, 1], n_samples, p=[0.96, 0.04])
    
    return pd.DataFrame(data)

def preprocess_data(df, is_training=True):
    """Preprocess the data"""
    df_processed = df.copy()
    
    # Handle missing values (represented as -1)
    df_processed = df_processed.replace(-1, np.nan)
    
    # Separate features by type
    binary_features = [col for col in df_processed.columns if col.endswith('_bin')]
    categorical_features = [col for col in df_processed.columns if col.endswith('_cat')]
    continuous_features = [col for col in df_processed.columns 
                          if not col.endswith(('_bin', '_cat')) and col not in ['id', 'target']]
    
    # Impute missing values
    # For binary and categorical: mode
    for col in binary_features + categorical_features:
        if col in df_processed.columns:
            mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 0
            df_processed[col] = df_processed[col].fillna(mode_val)
    
    # For continuous: median
    for col in continuous_features:
        if col in df_processed.columns:
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
    
    return df_processed

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and evaluate them with timing and persistence"""
    global models, model_results, training_start_time, training_duration
    
    training_start_time = time.time()
    
    # Define models - Using only 3 fast models for better performance
    model_configs = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),  # Reduced trees, parallel processing
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, solver='liblinear'),  # Faster solver
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)  # Limited depth for speed
    }
    
    results = {}
    
    for name, model in model_configs.items():
        try:
            print(f"Training {name}...")
            model_start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5
            
            model_training_time = time.time() - model_start_time
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'training_time': model_training_time
            }
            
            models[name] = model
            
            # Save model to disk
            model_filename = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}_model.pkl")
            joblib.dump(model, model_filename)
            print(f"Saved {name} model to {model_filename}")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    training_duration = time.time() - training_start_time
    model_results = results
    return results

def load_saved_models():
    """Load previously saved models"""
    global models
    loaded_models = {}
    
    if os.path.exists(MODEL_DIR):
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith('_model.pkl'):
                model_name = filename.replace('_model.pkl', '').replace('_', ' ').title()
                try:
                    model_path = os.path.join(MODEL_DIR, filename)
                    model = joblib.load(model_path)
                    loaded_models[model_name] = model
                    print(f"Loaded {model_name} from {model_path}")
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
    
    models.update(loaded_models)
    return loaded_models

def create_plot(plot_type, data=None, **kwargs):
    """Create various plots and return as base64 string"""
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'target_distribution':
        if 'target' in train_data.columns:
            counts = train_data['target'].value_counts()
            plt.bar(['No Claim (0)', 'Claim (1)'], counts.values)
            plt.title('Target Variable Distribution')
            plt.ylabel('Count')
        
    elif plot_type == 'missing_values':
        missing_data = train_data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            plt.bar(range(len(missing_data)), missing_data.values)
            plt.xticks(range(len(missing_data)), missing_data.index, rotation=45)
            plt.title('Missing Values by Feature')
            plt.ylabel('Count of Missing Values')
        else:
            plt.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Missing Values Analysis')
    
    elif plot_type == 'correlation_matrix':
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = train_data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
        
    elif plot_type == 'model_comparison':
        if model_results:
            models_list = list(model_results.keys())
            accuracies = [model_results[model]['accuracy'] for model in models_list]
            f1_scores = [model_results[model]['f1_score'] for model in models_list]
            
            x = np.arange(len(models_list))
            width = 0.35
            
            plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            plt.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
            
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x, models_list, rotation=45)
            plt.legend()
            plt.tight_layout()
    
    elif plot_type == 'confusion_matrix' and 'model_name' in kwargs:
        model_name = kwargs['model_name']
        if model_name in model_results:
            y_true = kwargs.get('y_true', [])
            y_pred = model_results[model_name]['predictions']
            
            if len(y_true) > 0:
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data-analysis')
def data_analysis():
    if train_data is None:
        load_data()
    
    if train_data is not None:
        # Basic statistics
        stats = {
            'total_rows': len(train_data),
            'total_columns': len(train_data.columns),
            'missing_values': train_data.isnull().sum().sum(),
            'target_distribution': train_data['target'].value_counts().to_dict() if 'target' in train_data.columns else {},
            'feature_types': {
                'binary': len([col for col in train_data.columns if col.endswith('_bin')]),
                'categorical': len([col for col in train_data.columns if col.endswith('_cat')]),
                'continuous': len([col for col in train_data.columns if not col.endswith(('_bin', '_cat')) and col not in ['id', 'target']])
            }
        }
        
        # Create visualizations
        target_plot = create_plot('target_distribution')
        missing_plot = create_plot('missing_values')
        
        return render_template('data_analysis.html', stats=stats, 
                             target_plot=target_plot, missing_plot=missing_plot,
                             sample_data=train_data.head().to_html(classes='table table-striped'))
    
    return render_template('data_analysis.html', error="Data not loaded")

@app.route('/preprocessing')
def preprocessing():
    global processed_data
    
    if train_data is None:
        load_data()
    
    if train_data is not None:
        # Show before preprocessing
        before_stats = {
            'missing_values': train_data.isnull().sum().sum(),
            'shape': train_data.shape
        }
        
        # Preprocess data
        processed_data = preprocess_data(train_data)
        
        # Show after preprocessing
        after_stats = {
            'missing_values': processed_data.isnull().sum().sum(),
            'shape': processed_data.shape
        }
        
        preprocessing_steps = [
            "Replaced -1 values with NaN to represent missing values",
            "Imputed missing values in binary/categorical features with mode",
            "Imputed missing values in continuous features with median",
            "Applied SMOTE for handling class imbalance (if needed)"
        ]
        
        return render_template('preprocessing.html', 
                             before_stats=before_stats,
                             after_stats=after_stats,
                             preprocessing_steps=preprocessing_steps,
                             before_sample=train_data.head().to_html(classes='table table-striped'),
                             after_sample=processed_data.head().to_html(classes='table table-striped'))
    
    return render_template('preprocessing.html', error="Data not loaded")

@app.route('/visualization')
def visualization():
    if train_data is None:
        load_data()
    
    plots = {}
    if train_data is not None:
        plots['target_distribution'] = create_plot('target_distribution')
        plots['missing_values'] = create_plot('missing_values')
        plots['correlation_matrix'] = create_plot('correlation_matrix')
    
    return render_template('visualization.html', plots=plots)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/train-models', methods=['POST'])
def train_models_route():
    global processed_data, model_results, training_duration
    
    try:
        if processed_data is None:
            if train_data is None:
                load_data()
            processed_data = preprocess_data(train_data)
        
        # Prepare features and target
        X = processed_data.drop(['id', 'target'], axis=1, errors='ignore')
        y = processed_data['target'] if 'target' in processed_data.columns else None
        
        if y is None:
            return jsonify({'error': 'Target variable not found'})
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler for later use
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Train models
        results = train_models(X_train_balanced, y_train_balanced, X_test_scaled, y_test)
        
        # Create comparison plot
        comparison_plot = create_plot('model_comparison')
        
        return jsonify({
            'success': True,
            'results': {name: {k: v for k, v in result.items() if k not in ['model', 'predictions', 'probabilities']} 
                       for name, result in results.items()},
            'comparison_plot': comparison_plot,
            'training_duration': f"{training_duration:.2f} seconds",
            'total_models': len(results),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        feature_data = {}
        for key, value in request.form.items():
            if key != 'model_choice':
                try:
                    feature_data[key] = float(value)
                except:
                    feature_data[key] = 0.0  # Default value for missing features
        
        model_choice = request.form.get('model_choice', 'Random Forest')
        
        if model_choice not in models:
            # Try to load saved model
            loaded_models = load_saved_models()
            if model_choice not in loaded_models:
                return jsonify({'error': f'Model {model_choice} not trained yet. Please train models first.'})
        
        # Create feature vector with all features in correct order
        feature_names = [
            'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat', 'ps_ind_05_cat',
            'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin',
            'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15',
            'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
            'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat',
            'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat',
            'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15'
        ]
        
        # Add calculated features
        for i in range(1, 15):
            feature_names.append(f'ps_calc_{i:02d}')
        for i in range(15, 21):
            feature_names.append(f'ps_calc_{i}_bin')
        
        # Create feature vector
        feature_vector = np.array([[feature_data.get(name, 0.0) for name in feature_names]])
        
        # Load and apply scaler if available
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            feature_vector = scaler.transform(feature_vector)
        
        # Make prediction
        model = models[model_choice]
        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        confidence = max(probability) * 100
        
        return jsonify({
            'prediction': int(prediction),
            'risk_level': risk_level,
            'confidence': f"{confidence:.1f}%",
            'probability_no_claim': f"{probability[0]*100:.1f}%",
            'probability_claim': f"{probability[1]*100:.1f}%",
            'model_used': model_choice,
            'features_processed': len(feature_names)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/about')
def about():
    team_members = [
        {
            'name': 'Data Scientist',
            'role': 'Machine Learning Engineer',
            'description': 'Specialized in developing predictive models and data preprocessing pipelines.',
            'skills': ['Python', 'Scikit-learn', 'Pandas', 'Machine Learning']
        },
        {
            'name': 'Web Developer',
            'role': 'Full Stack Developer',
            'description': 'Responsible for creating the Flask web application and user interface.',
            'skills': ['Flask', 'HTML/CSS', 'JavaScript', 'Bootstrap']
        },
        {
            'name': 'Data Analyst',
            'role': 'Data Visualization Specialist',
            'description': 'Expert in creating insightful visualizations and statistical analysis.',
            'skills': ['Matplotlib', 'Seaborn', 'Statistical Analysis', 'Data Visualization']
        }
    ]
    
    return render_template('about.html', team_members=team_members)

@app.route('/load-models', methods=['POST'])
def load_models_route():
    """Load previously saved models"""
    try:
        loaded_models = load_saved_models()
        if loaded_models:
            return jsonify({
                'success': True,
                'loaded_models': list(loaded_models.keys()),
                'message': f"Loaded {len(loaded_models)} saved models"
            })
        else:
            return jsonify({
                'success': False,
                'message': "No saved models found"
            })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Initialize data on startup
    load_data()
    # Load any previously saved models
    load_saved_models()
    app.run(debug=True)
