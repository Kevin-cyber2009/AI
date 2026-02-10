# tests/test_stress_lab.py

"""
Unit tests cho stress_lab module
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.stress_lab import StressLab
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def stress_lab(config):
    return StressLab(config)


@pytest.fixture
def sample_frames():
    np.random.seed(42)
    frames = np.random.rand(20, 288, 512, 3).astype(np.float32)
    return frames


def test_stress_lab_initialization(stress_lab, config):
    """Test khởi tạo stress lab"""
    assert stress_lab.light_jitter == config['stress_lab']['light_jitter_strength']
    assert stress_lab.blur_range == config['stress_lab']['blur_kernel_range']
    assert stress_lab.noise_std == config['stress_lab']['noise_std']


def test_apply_light_jitter(stress_lab, sample_frames):
    """Test light jitter perturbation"""
    perturbed = stress_lab.apply_light_jitter(sample_frames)
    
    assert perturbed.shape == sample_frames.shape
    assert perturbed.dtype == np.float32
    assert np.min(perturbed) >= 0.0
    assert np.max(perturbed) <= 1.0
    
    # Should be different from original
    assert not np.array_equal(perturbed, sample_frames)


def test_apply_blur(stress_lab, sample_frames):
    """Test blur perturbation"""
    perturbed = stress_lab.apply_blur(sample_frames)
    
    assert perturbed.shape == sample_frames.shape
    assert perturbed.dtype == sample_frames.dtype
    
    # Blurred image should have lower variance than original
    original_var = np.var(sample_frames)
    perturbed_var = np.var(perturbed)
    assert perturbed_var <= original_var * 1.1  # Allow some tolerance


def test_apply_affine_jitter(stress_lab, sample_frames):
    """Test affine transformation"""
    perturbed = stress_lab.apply_affine_jitter(sample_frames)
    
    assert perturbed.shape == sample_frames.shape
    assert perturbed.dtype == sample_frames.dtype
    assert np.min(perturbed) >= 0.0
    assert np.max(perturbed) <= 1.0


def test_apply_noise(stress_lab, sample_frames):
    """Test noise injection"""
    perturbed = stress_lab.apply_noise(sample_frames)
    
    assert perturbed.shape == sample_frames.shape
    assert np.min(perturbed) >= 0.0
    assert np.max(perturbed) <= 1.0
    
    # Mean should be similar
    assert abs(np.mean(perturbed) - np.mean(sample_frames)) < 0.1


def test_apply_temporal_shuffle(stress_lab, sample_frames):
    """Test temporal shuffle"""
    perturbed = stress_lab.apply_temporal_shuffle(sample_frames, window_size=5)
    
    assert perturbed.shape == sample_frames.shape
    
    # Some frames should be in different positions
    # (but not guaranteed to be different everywhere due to random shuffle)
    assert isinstance(perturbed, np.ndarray)


def test_compute_stability_score(stress_lab):
    """Test stability score computation"""
    features_original = {
        'feature1': 0.5,
        'feature2': 0.8,
        'feature3': 0.3
    }
    
    features_perturbed = {
        'feature1': 0.52,
        'feature2': 0.79,
        'feature3': 0.31
    }
    
    stability = stress_lab.compute_stability_score(features_original, features_perturbed)
    
    assert isinstance(stability, dict)
    assert 'overall_stability' in stability
    assert 0 <= stability['overall_stability'] <= 1
    
    # Stability ratios should exist
    assert 'stability_feature1' in stability
    assert 'stability_feature2' in stability
    assert 'stability_feature3' in stability


def test_stability_score_edge_cases(stress_lab):
    """Test stability với edge cases"""
    # Case 1: Identical features
    features = {'feat1': 0.5, 'feat2': 0.8}
    stability = stress_lab.compute_stability_score(features, features)
    
    # All ratios should be 1.0
    assert stability['stability_feat1'] == 1.0
    assert stability['stability_feat2'] == 1.0
    assert stability['overall_stability'] == 1.0
    
    # Case 2: Zero values
    features_zero = {'feat1': 0.0, 'feat2': 0.5}
    features_nonzero = {'feat1': 0.1, 'feat2': 0.5}
    
    stability = stress_lab.compute_stability_score(features_zero, features_nonzero)
    
    # Should handle gracefully
    assert isinstance(stability['overall_stability'], float)
    assert not np.isnan(stability['overall_stability'])


def test_frames_to_video_conversion(stress_lab, sample_frames):
    """Test chuyển đổi frames sang video"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_video = f.name
    
    try:
        stress_lab._frames_to_video(sample_frames[:10], temp_video, fps=6)
        
        # Check file exists
        assert Path(temp_video).exists()
        assert Path(temp_video).stat().st_size > 0
        
    finally:
        Path(temp_video).unlink(missing_ok=True)


def test_video_to_frames_conversion(stress_lab):
    """Test đọc frames từ video"""
    # Create temp video first
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_video = f.name
    
    try:
        # Create simple video
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, 6, (512, 288))
        
        for _ in range(10):
            frame = np.random.randint(0, 255, (288, 512, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        # Read back
        frames = stress_lab._video_to_frames(temp_video)
        
        assert isinstance(frames, np.ndarray)
        assert len(frames) == 10
        assert frames.dtype == np.float32
        
    finally:
        Path(temp_video).unlink(missing_ok=True)