"""app/explainer.py

LIME and SHAP explainer classes for the Explainable Cloud AI Monitor.

Both explainers are model-agnostic: they treat the predictor as a black box,
requiring only a callable that maps input arrays to probability arrays.
This makes them compatible with any backend: local sklearn, Cloud ML API,
Vertex AI endpoint, or any other prediction service.

Key design choice:
  - LIME: fast, local, perturbation-based - good for quick individual explanations
  - SHAP: slower, theoretically grounded, global+local - good for rigorous analysis
"""

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer


class LIMEExplainer:
    """Local Interpretable Model-agnostic Explanations (LIME) for tabular data.
    
    LIME works by:
        1. Perturbing the input instance (generating nearby synthetic samples)
        2. Getting predictions from the black-box model for each sample
        3. Fitting a simple interpretable model (linear) on the perturbed samples
        4. Using the linear model's coefficients as feature importances
    
    Known limitation: explanations can vary across runs due to random perturbation.
    Use num_samples=1000 for more stable results.
    """
    
    def __init__(
        self,
        predictor,
        training_data: np.ndarray,
        feature_names: list,
        class_names: list,
        mode: str = "classification",
        num_samples: int = 1000,
    ):
        """Initialize LIME explainer.
        
        Args:
            predictor: Callable that takes (n_samples, n_features) array,
                returns (n_samples, n_classes) probability array
            training_data: Used to compute feature statistics for perturbation
            feature_names: List of feature name strings
            class_names: List of class label strings
            mode: 'classification' or 'regression'
            num_samples: Number of perturbed samples per explanation
        """
        self.predictor = predictor
        self.num_samples = num_samples
        self.explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode,
            random_state=42,
        )
    
    def explain(self, instance: np.ndarray, label_idx: int = None):
        """Generate LIME explanation for a single instance.
        
        Args:
            instance: Shape (1, n_features) or (n_features,)
            label_idx: Class index to explain (defaults to predicted class)
        
        Returns:
            LimeTabularExplainer explanation object
        """
        instance = instance.flatten()
        probs = self.predictor(instance.reshape(1, -1))[0]
        
        if label_idx is None:
            label_idx = int(np.argmax(probs))
        
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.predictor,
            num_features=len(instance),
            num_samples=self.num_samples,
            labels=(label_idx,),
        )
        return explanation
    
    def top_features(self, explanation, n: int = 5):
        """Extract top-N feature importances from a LIME explanation.
        
        Returns:
            List of (feature_name, importance_score) tuples, sorted by importance
        """
        label_idx = explanation.available_labels()[0]
        features = explanation.as_list(label=label_idx)
        return sorted(features, key=lambda x: abs(x[1]), reverse=True)[:n]


class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) for tree-based models.
    
    SHAP computes Shapley values - a game-theoretic measure of each feature's
    contribution to a prediction, relative to the expected model output.
    
    Properties:
        - Local accuracy: explanations sum to the model output
        - Consistency: if a feature's contribution increases, its Shapley value increases
        - Global+local: individual predictions AND dataset-wide feature importance
    
    Uses TreeExplainer for tree-based models (exact, fast).
    Falls back to KernelExplainer for any model type (approximate, slower).
    """
    
    def __init__(
        self,
        model,
        scaler,
        X_train: np.ndarray,
        feature_names: list,
        use_kernel: bool = False,
    ):
        """Initialize SHAP explainer.
        
        Args:
            model: Trained sklearn model or any callable (for KernelExplainer)
            scaler: Fitted scaler for preprocessing
            X_train: Training data (scaled) for background distribution
            feature_names: List of feature name strings
            use_kernel: Force KernelExplainer (model-agnostic, slower)
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
        if use_kernel:
            # Model-agnostic: works with any predictor (including Cloud APIs)
            background = shap.kmeans(X_train, k=10)
            self.explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            # TreeExplainer: fast, exact Shapley values for tree models
            self.explainer = shap.TreeExplainer(model)
    
    def compute(self, X: np.ndarray):
        """Compute SHAP values for a dataset.
        
        Args:
            X: Input array (scaled), shape (n_samples, n_features)
        
        Returns:
            SHAP values array (shape varies by model type)
        """
        shap_values = self.explainer.shap_values(X)
        return shap_values
    
    def compute_single(self, instance: np.ndarray):
        """Compute SHAP values for a single instance.
        
        Args:
            instance: Shape (1, n_features), already scaled
        
        Returns:
            SHAP values for the instance
        """
        return self.explainer.shap_values(instance)
