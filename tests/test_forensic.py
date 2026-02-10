# tests/test_forensic.py

"""
Unit tests cho forensic module
"""

import pytest
import numpy as np

from src.forensic import ForensicAnalyzer
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def analyzer(config):
    return ForensicAnalyzer(config)


@pytest.fixture
def sample_frames():
    """Tạo sample frames"""
    np.random.seed(42)
    frames = np.random.rand(20, 288, 512, 3).astype(np.float32)
    return frames


def test_analyzer_initialization(analyzer, config):
    """Test khởi tạo analyzer"""
    assert analyzer.fft_components == config['forensic']['fft_components']
    assert analyzer.dct_components == config['forensic']['dct_components']


def test_compute_fft_features(analyzer, sample_frames):
    """Test FFT features"""
    features = analyzer.compute_fft_features(sample_frames)
    
    assert isinstance(features, dict)
    assert 'fft_mean' in features
    assert 'fft_std' in features
    assert 'fft_high_freq_energy' in features
    assert 'fft_radial_slope' in features
    
    # Check values are numeric
    for key, value in features.items():
        assert isinstance(value, (int, float))
        assert not np.isnan(value)


def test_compute_dct_features(analyzer, sample_frames):
    """Test DCT features"""
    features = analyzer.compute_dct_features(sample_frames)
    
    assert isinstance(features, dict)
    assert 'dct_mean' in features
    assert 'dct_std' in features
    assert 'dct_dc_mean' in features
    assert 'dct_ac_energy' in features
    
    for value in features.values():
        assert not np.isnan(value)


def test_compute_prnu_residual(analyzer, sample_frames):
    """Test PRNU residual"""
    features = analyzer.compute_prnu_residual(sample_frames)
    
    assert isinstance(features, dict)
    assert 'prnu_mean' in features
    assert 'prnu_std' in features
    assert 'prnu_autocorr' in features
    assert 'prnu_temporal_consistency' in features


def test_compute_optical_flow(analyzer, sample_frames):
    """Test optical flow"""
    features = analyzer.compute_optical_flow(sample_frames)
    
    assert isinstance(features, dict)
    assert 'flow_mean_magnitude' in features
    assert 'flow_std_magnitude' in features
    assert 'flow_smoothness' in features
    assert 'flow_temporal_consistency' in features


def test_full_analysis(analyzer, sample_frames):
    """Test full forensic analysis"""
    all_features = analyzer.analyze(sample_frames)
    
    assert isinstance(all_features, dict)
    assert len(all_features) > 15  # Should have many features
    
    # Check all features are numeric and not NaN
    for key, value in all_features.items():
        assert isinstance(value, (int, float)), f"Feature {key} is not numeric"
        assert not np.isnan(value), f"Feature {key} is NaN"
        assert not np.isinf(value), f"Feature {key} is Inf"


def test_optical_flow_short_video(analyzer):
    """Test optical flow với video quá ngắn"""
    short_frames = np.random.rand(2, 288, 512, 3).astype(np.float32)
    
    features = analyzer.compute_optical_flow(short_frames)
    
    # Should return default values without crashing
    assert features['flow_mean_magnitude'] >= 0
    assert features['flow_temporal_consistency'] >= 0