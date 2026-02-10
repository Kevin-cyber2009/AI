# tests/test_pipeline.py

"""
Integration tests cho full pipeline
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path

from src.features import FeatureExtractor
from src.fusion import ScoreFusion
from src.classifier import VideoClassifier
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def dummy_video():
    """Tạo dummy video"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (512, 288))
    
    # Tạo video có pattern đơn giản
    for i in range(40):
        frame = np.zeros((288, 512, 3), dtype=np.uint8)
        
        # Moving circle
        x = int(256 + 100 * np.sin(i * 0.2))
        y = int(144 + 50 * np.cos(i * 0.2))
        cv2.circle(frame, (x, y), 30, (255, 100, 100), -1)
        
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    Path(video_path).unlink(missing_ok=True)


def test_full_pipeline(config, dummy_video):
    """Test pipeline đầy đủ: video -> features -> scores -> decision"""
    
    # Step 1: Extract features
    extractor = FeatureExtractor(config)
    features, metadata = extractor.extract_from_video(dummy_video)
    
    assert isinstance(features, dict)
    assert len(features) > 20
    
    # Step 2: Compute scores
    fusion = ScoreFusion(config)
    
    artifact_score = fusion.compute_artifact_score(features)
    reality_score = fusion.compute_reality_score(features)
    
    # Mock stress score
    stress_results = {'aggregate_stability_score': 0.75}
    stress_score = fusion.compute_stress_score(stress_results)
    
    assert 0 <= artifact_score <= 1
    assert 0 <= reality_score <= 1
    assert 0 <= stress_score <= 1
    
    # Step 3: Fuse scores
    result = fusion.fuse_scores(artifact_score, reality_score, stress_score)
    
    assert isinstance(result, dict)
    assert 'final_probability' in result
    assert 'prediction' in result
    assert 'confidence' in result
    
    assert 0 <= result['final_probability'] <= 1
    assert result['prediction'] in ['REAL', 'FAKE']
    assert result['confidence'] in ['LOW', 'MEDIUM', 'HIGH']
    
    # Step 4: Generate explanations
    explanations = fusion.generate_explanation(features, result)
    
    assert isinstance(explanations, list)
    assert len(explanations) == 3
    assert all(isinstance(exp, str) for exp in explanations)


def test_feature_vector_conversion(config, dummy_video):
    """Test chuyển đổi features thành vector"""
    extractor = FeatureExtractor(config)
    
    features, _ = extractor.extract_from_video(dummy_video)
    feature_names = extractor.get_feature_names()
    
    vector = extractor.features_to_vector(features, feature_names)
    
    assert isinstance(vector, np.ndarray)
    assert len(vector) == len(feature_names)
    assert not np.any(np.isnan(vector))
    assert not np.any(np.isinf(vector))


def test_batch_processing(config):
    """Test xử lý nhiều videos"""
    extractor = FeatureExtractor(config)
    
    # Tạo nhiều dummy videos
    video_paths = []
    
    for idx in range(3):
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10, (512, 288))
        
        for i in range(20):
            frame = np.random.randint(0, 255, (288, 512, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        video_paths.append(video_path)
    
    try:
        # Extract features từ tất cả
        all_features = []
        
        for vp in video_paths:
            features, _ = extractor.extract_from_video(vp)
            all_features.append(features)
        
        assert len(all_features) == 3
        
        # Convert to vectors
        feature_names = extractor.get_feature_names()
        vectors = [extractor.features_to_vector(f, feature_names) for f in all_features]
        
        X = np.vstack(vectors)
        assert X.shape == (3, len(feature_names))
        
    finally:
        for vp in video_paths:
            Path(vp).unlink(missing_ok=True)


def test_end_to_end_with_classifier(config, dummy_video):
    """Test end-to-end với classifier"""
    
    # Create synthetic training data
    np.random.seed(42)
    X_train = np.random.randn(100, 30).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    
    # Train classifier
    classifier = VideoClassifier(config)
    classifier.train(X_train, y_train)
    
    # Extract features từ test video
    extractor = FeatureExtractor(config)
    features, _ = extractor.extract_from_video(dummy_video)
    
    # Convert to vector (đảm bảo đúng số features)
    feature_vector = np.random.randn(30).astype(np.float32)  # Mock vector
    
    # Predict
    pred, prob = classifier.predict(feature_vector.reshape(1, -1))
    
    assert pred[0] in [0, 1]
    assert 0 <= prob[0] <= 1


def test_stress_integration(config):
    """Test tích hợp stress lab"""
    from src.stress_lab import StressLab
    from src.preprocessing import VideoPreprocessor
    
    # Create dummy video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (512, 288))
    
    for i in range(20):
        frame = np.random.randint(100, 200, (288, 512, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    try:
        # Preprocess
        preprocessor = VideoPreprocessor(config)
        frames, _ = preprocessor.preprocess(video_path)
        
        # Extract features
        extractor = FeatureExtractor(config)
        original_features = extractor.extract_features(frames)
        
        # Apply perturbation
        stress_lab = StressLab(config)
        perturbed = stress_lab.apply_light_jitter(frames)
        
        # Extract features from perturbed
        perturbed_features = extractor.extract_features(perturbed)
        
        # Compute stability
        stability = stress_lab.compute_stability_score(original_features, perturbed_features)
        
        assert 'overall_stability' in stability
        assert 0 <= stability['overall_stability'] <= 2
        
    finally:
        Path(video_path).unlink(missing_ok=True)
