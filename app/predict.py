"""app/predict.py

Core prediction explanation pipeline for Explainable Cloud AI Monitor.
Uses a scikit-learn RandomForestClassifier trained on the Iris dataset as the local model.
The predictor is wrapped in a CloudPredictor class whose interface mirrors Google Cloud 
Vertex AI's prediction endpoint, making it trivially swappable for a real Cloud endpoint 
by overriding predict().

Pipeline:
  1. Train/load a model (local sklearn or Cloud API)
  2. Run prediction on input
  3. Pass prediction & model to explainer.py for LIME/SHAP analysis
  4. Pass explanations to visualize.py for plotting
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from explainer import LIMEExplainer, SHAPExplainer
from visualize import plot_lime, plot_shap_summary, plot_shap_force


def train_model():
    """Train a RandomForestClassifier on the Iris dataset.
    
    Returns:
        model: Trained RandomForestClassifier
        scaler: StandardScaler fit on training data
        X_test, y_test: Test set features and labels
        feature_names: List of feature names
        class_names: List of class names
    """
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model trained. Test accuracy: {acc:.3f}")
    
    return model, scaler, X_train, X_test, y_test, feature_names, class_names


class CloudPredictor:
    """Wraps a local sklearn model with a Cloud API-compatible interface.
    
    To swap in a real Google Cloud Vertex AI endpoint:
        - Override predict() to call the Vertex AI REST endpoint
        - Pass your endpoint URL and credentials in __init__()
    
    Example (Vertex AI swap):
        from google.cloud import aiplatform
        endpoint = aiplatform.Endpoint(endpoint_name='projects/.../endpoints/...')
        predictions = endpoint.predict(instances=X.tolist())
    """
    
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input X.
        
        Returns:
            Array of shape (n_samples, n_classes)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def run_pipeline(instance_idx: int = 0):
    """Full explain pipeline on a single test instance.
    
    Args:
        instance_idx: Index of the test sample to explain
    
    Steps:
        1. Train model
        2. Wrap in CloudPredictor
        3. LIME explanation for the instance
        4. SHAP explanation (local & global)
        5. Visualize and save plots
    """
    print("-" * 60)
    print("Explainable Cloud AI Monitor")
    print("-" * 60)
    
    # 1. Train model
    model, scaler, X_train, X_test, y_test, feature_names, class_names = train_model()
    
    # 2. Wrap in CloudPredictor
    predictor = CloudPredictor(model, scaler)
    
    # 2.5 Select instance
    instance = X_test[instance_idx].reshape(1, -1)
    true_label = class_names[y_test[instance_idx]]
    pred_probs = predictor.predict(instance)[0]
    pred_label = class_names[np.argmax(pred_probs)]
    
    print(f"\nInstance {instance_idx}:")
    print(f"Features: {dict(zip(feature_names, X_test[instance_idx].round(3)))}")
    print(f"True label: {true_label}")
    print(f"Predicted: {pred_label} (confidence: {max(pred_probs):.3f})")
    
    # 3. LIME
    print("\nLIME: Generating local explanation...")
    lime_exp = LIMEExplainer(
        predictor=predictor.predict,
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
    )
    lime_explanation = lime_exp.explain(instance)
    print(f"Top features: {lime_exp.top_features(lime_explanation, n=3)}")
    plot_lime(lime_explanation, pred_label, save_path='lime_explanation.png')
    
    # 4. SHAP
    print("\nSHAP: Computing Shapley values...")
    shap_exp = SHAPExplainer(
        model=model,
        scaler=scaler,
        X_train=X_train,
        feature_names=feature_names,
    )
    shap_values = shap_exp.compute(X_test)
    plot_shap_summary(shap_values, X_test, feature_names, save_path='shap_summary.png')
    plot_shap_force(shap_exp, instance, scaler, save_path='shap_force.png')
    
    print(".")
    print("Plots saved: lime_explanation.png, shap_summary.png, shapforce.png")
    print("-" * 60)


if __name__ == "__main__":
    run_pipeline(instance_idx=0)
