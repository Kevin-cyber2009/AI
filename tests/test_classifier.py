# tests/test_classifier.py

"""
Unit tests cho classifier module
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.classifier import VideoClassifier
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def classifier(config):
    return VideoClassifier(config)


@pytest.fixture
def synthetic_data():
    """Tạo synthetic training data"""
    np.random.seed(42)
    
    # Real samples: lower artifact scores
    X_real = np.random.randn(50, 30) * 0.5 + 0.3
    y_real = np.zeros(50)
    
    # Fake samples: higher artifact scores
    X_fake = np.random.randn(50, 30) * 0.5 + 0.7
    y_fake = np.ones(50)
    
    X = np.vstack([X_real, X_fake]).astype(np.float32)
    y = np.concatenate([y_real, y_fake]).astype(np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def test_classifier_initialization(classifier, config):
    """Test khởi tạo classifier"""
    assert classifier.model_type == config['classifier']['model_type']
    assert classifier.model is None  # Not trained yet
    assert classifier.scaler is not None


def test_create_model(classifier):
    """Test tạo model"""
    model = classifier._create_model()
    
    assert model is not None
    
    if classifier.model_type == 'lightgbm':
        from lightgbm import LGBMClassifier
        assert isinstance(model, LGBMClassifier)
    elif classifier.model_type == 'svm':
        from sklearn.svm import SVC
        assert isinstance(model, SVC)


def test_train_classifier(classifier, synthetic_data):
    """Test training"""
    X, y = synthetic_data
    
    metrics = classifier.train(X, y)
    
    assert isinstance(metrics, dict)
    assert 'auc' in metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    
    # Check metrics are valid
    assert 0 <= metrics['auc'] <= 1
    assert 0 <= metrics['accuracy'] <= 1
    
    # Model should be trained
    assert classifier.model is not None
    assert classifier.calibrator is not None


def test_predict(classifier, synthetic_data):
    """Test prediction"""
    X, y = synthetic_data
    
    # Train first
    classifier.train(X, y)
    
    # Predict
    preds, probs = classifier.predict(X)
    
    assert len(preds) == len(X)
    assert len(probs) == len(X)
    assert preds.dtype == np.int32 or preds.dtype == np.int64
    
    # Predictions should be 0 or 1
    assert np.all((preds == 0) | (preds == 1))
    
    # Probabilities should be [0, 1]
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_evaluate(classifier, synthetic_data):
    """Test evaluation"""
    X, y = synthetic_data
    
    classifier.train(X, y)
    metrics = classifier.evaluate(X, y)
    
    assert isinstance(metrics, dict)
    assert 'auc' in metrics
    assert 'fpr_at_tpr_0.9' in metrics
    
    # AUC should be decent for synthetic separable data
    assert metrics['auc'] > 0.6


def test_cross_validation(classifier, synthetic_data):
    """Test cross-validation"""
    X, y = synthetic_data
    
    cv_results = classifier.cross_validate(X, y, cv=3)
    
    assert isinstance(cv_results, dict)
    assert 'cv_scores' in cv_results
    assert 'mean_auc' in cv_results
    assert 'std_auc' in cv_results
    
    assert len(cv_results['cv_scores']) == 3
    assert 0 <= cv_results['mean_auc'] <= 1


def test_save_and_load(classifier, synthetic_data):
    """Test save và load model"""
    X, y = synthetic_data
    
    # Train
    classifier.train(X, y)
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        model_path = f.name
    
    try:
        classifier.save(model_path)
        
        # Check file exists
        assert Path(model_path).exists()
        
        # Load into new classifier
        new_classifier = VideoClassifier()
        new_classifier.load(model_path)
        
        # Predict with both
        _, probs1 = classifier.predict(X[:10])
        _, probs2 = new_classifier.predict(X[:10])
        
        # Should be very similar
        assert np.allclose(probs1, probs2, atol=1e-5)
        
    finally:
        Path(model_path).unlink(missing_ok=True)


def test_feature_importance(classifier, synthetic_data):
    """Test feature importance"""
    X, y = synthetic_data
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    classifier.train(X, y, feature_names)
    
    if classifier.model_type == 'lightgbm':
        importance = classifier.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(feature_names)
        
        # All importances should be >= 0
        assert all(v >= 0 for v in importance.values())


def test_predict_before_training(classifier, synthetic_data):
    """Test predict trước khi train"""
    X, y = synthetic_data
    
    with pytest.raises(ValueError):
        classifier.predict(X)