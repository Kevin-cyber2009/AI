# tests/test_features.py

"""
Unit tests cho features module
"""

import pytest
import numpy as np
import tempfile
import cv2
from pathlib import Path

from src.features import FeatureExtractor
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def extractor(config):
    return FeatureExtractor(config)


@pytest.fixture
def sample_frames():
    np.random.seed(42)
    frames = np.random.rand(20, 288, 512, 3).astype(np.float32)
    return frames


@pytest.fixture
def dummy_video():
    """Tạo dummy video"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (512, 288))
    
    for i in range(30):
        frame = np.random.randint(0, 255, (288, 512, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    Path(video_path).unlink(missing_ok=True)


def test_extractor_initialization(extractor, config):
    """Test khởi tạo feature extractor"""
    assert extractor.expected_dim == config['features']['expected_dimension']
    assert extractor.normalization == config['features']['normalization']
    assert extractor.preprocessor is not None
    assert extractor.forensic is not None
    assert extractor.reality is not None


def test_extract_features_from_frames(extractor, sample_frames):
    """Test trích xuất features từ frames"""
    features = extractor.extract_features(sample_frames)
    
    assert isinstance(features, dict)
    assert len(features) > 20  # Should have many features
    
    # Check all features are valid
    for key, value in features.items():
        assert isinstance(value, (int, float))
        assert not np.isnan(value)
        assert not np.isinf(value)


def test_extract_from_video(extractor, dummy_video):
    """Test extract từ video file"""
    features, metadata = extractor.extract_from_video(dummy_video)
    
    assert isinstance(features, dict)
    assert isinstance(metadata, dict)
    
    assert len(features) > 20
    assert 'num_frames' in metadata
    assert metadata['num_frames'] > 0


def test_features_to_vector(extractor, sample_frames):
    """Test convert features dict sang vector"""
    features = extractor.extract_features(sample_frames)
    feature_names = extractor.get_feature_names()
    
    vector = extractor.features_to_vector(features, feature_names)
    
    assert isinstance(vector, np.ndarray)
    assert len(vector.shape) == 1
    assert len(vector) == len(feature_names)
    assert vector.dtype == np.float32
    
    # No NaN or Inf
    assert not np.any(np.isnan(vector))
    assert not np.any(np.isinf(vector))


def test_normalize_features(extractor):
    """Test normalization"""
    vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    normalized = extractor.normalize_features(vector)
    
    assert normalized.shape == vector.shape
    
    # Standard normalization: mean ~0, std ~1
    if extractor.normalization == 'standard':
        assert abs(np.mean(normalized)) < 0.1
        assert abs(np.std(normalized) - 1.0) < 0.1


def test_get_feature_names(extractor):
    """Test lấy feature names"""
    names = extractor.get_feature_names()
    
    assert isinstance(names, list)
    assert len(names) > 20
    assert all(isinstance(n, str) for n in names)
    
    # Check some expected names
    assert 'fft_mean' in names
    assert 'entropy_mean' in names
    assert 'fractal_dim_mean' in names


def test_features_consistency(extractor, sample_frames):
    """Test features nhất quán qua nhiều lần extract"""
    features1 = extractor.extract_features(sample_frames)
    features2 = extractor.extract_features(sample_frames)
    
    # Should be identical (deterministic)
    for key in features1:
        if key in features2:
            assert abs(features1[key] - features2[key]) < 1e-6


def test_handle_missing_features(extractor):
    """Test xử lý missing features"""
    partial_features = {
        'fft_mean': 0.5,
        'entropy_mean': 0.3
    }
    
    feature_names = extractor.get_feature_names()
    vector = extractor.features_to_vector(partial_features, feature_names)
    
    # Missing features should be filled with 0
    assert len(vector) == len(feature_names)
    
    # First two should have values, rest should be 0
    non_zero_count = np.count_nonzero(vector)
    assert non_zero_count >= 2