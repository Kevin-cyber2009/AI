# tests/test_reality_engine.py

"""
Unit tests cho reality_engine module
"""

import pytest
import numpy as np

from src.reality_engine import RealityEngine
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def engine(config):
    return RealityEngine(config)


@pytest.fixture
def sample_frames():
    np.random.seed(42)
    frames = np.random.rand(30, 288, 512, 3).astype(np.float32)
    return frames


def test_engine_initialization(engine, config):
    """Test khởi tạo reality engine"""
    assert engine.entropy_scales == config['reality_engine']['entropy_scales']
    assert engine.fractal_boxes == config['reality_engine']['fractal_box_sizes']


def test_compute_multiscale_entropy(engine, sample_frames):
    """Test multi-scale entropy"""
    features = engine.compute_multiscale_entropy(sample_frames)
    
    assert isinstance(features, dict)
    assert 'entropy_mean' in features
    assert 'entropy_std' in features
    assert 'entropy_slope' in features
    
    # Entropy should be positive
    assert features['entropy_mean'] >= 0
    assert features['entropy_std'] >= 0


def test_compute_fractal_dimension(engine, sample_frames):
    """Test fractal dimension"""
    features = engine.compute_fractal_dimension(sample_frames)
    
    assert isinstance(features, dict)
    assert 'fractal_dim_mean' in features
    assert 'fractal_dim_std' in features
    
    # Fractal dimension typically in range [0, 3]
    assert 0 <= features['fractal_dim_mean'] <= 3


def test_compute_causal_motion(engine, sample_frames):
    """Test causal motion prediction"""
    features = engine.compute_causal_motion(sample_frames)
    
    assert isinstance(features, dict)
    assert 'causal_prediction_error' in features
    assert 'causal_predictability' in features
    
    # Predictability should be [0, 1]
    assert 0 <= features['causal_predictability'] <= 1


def test_compute_information_conservation(engine, sample_frames):
    """Test information conservation"""
    features = engine.compute_information_conservation(sample_frames)
    
    assert isinstance(features, dict)
    assert 'compression_mean' in features
    assert 'compression_std' in features
    assert 'compression_delta_mean' in features
    assert 'complexity_mean' in features


def test_full_reality_analysis(engine, sample_frames):
    """Test full reality analysis"""
    all_features = engine.analyze(sample_frames)
    
    assert isinstance(all_features, dict)
    assert len(all_features) >= 10
    
    # All features should be valid numbers
    for key, value in all_features.items():
        assert isinstance(value, (int, float))
        assert not np.isnan(value)
        assert not np.isinf(value)


def test_causal_motion_short_video(engine):
    """Test causal motion với video ngắn"""
    short_frames = np.random.rand(5, 288, 512, 3).astype(np.float32)
    
    features = engine.compute_causal_motion(short_frames)
    
    # Should handle gracefully
    assert 'causal_prediction_error' in features
    assert 'causal_predictability' in features