# tests/test_preprocessing.py

"""
Unit tests cho preprocessing module
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path

from src.preprocessing import VideoPreprocessor
from src.utils import load_config


@pytest.fixture
def config():
    """Load config fixture"""
    return load_config()


@pytest.fixture
def preprocessor(config):
    """VideoPreprocessor fixture"""
    return VideoPreprocessor(config)


@pytest.fixture
def dummy_video():
    """Tạo dummy video để test"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    # Tạo video đơn giản
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (512, 288))
    
    for i in range(30):
        frame = np.random.randint(0, 255, (288, 512, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    yield video_path
    
    # Cleanup
    Path(video_path).unlink(missing_ok=True)


def test_preprocessor_initialization(preprocessor, config):
    """Test khởi tạo preprocessor"""
    assert preprocessor.fps == config['preprocessing']['fps']
    assert preprocessor.resize_width == config['preprocessing']['resize_width']
    assert preprocessor.resize_height == config['preprocessing']['resize_height']


def test_extract_frames(preprocessor, dummy_video):
    """Test trích xuất frames"""
    frames = preprocessor.extract_frames(dummy_video)
    
    assert isinstance(frames, np.ndarray)
    assert len(frames.shape) == 4  # (N, H, W, C)
    assert frames.shape[1] == preprocessor.resize_height
    assert frames.shape[2] == preprocessor.resize_width
    assert frames.dtype == np.uint8


def test_normalize_frames(preprocessor):
    """Test normalize frames"""
    frames = np.random.randint(0, 255, (10, 288, 512, 3), dtype=np.uint8)
    
    normalized = preprocessor.normalize_frames(frames)
    
    assert normalized.dtype == np.float32
    assert np.min(normalized) >= 0.0
    assert np.max(normalized) <= 1.0


def test_preprocess_pipeline(preprocessor, dummy_video):
    """Test full preprocess pipeline"""
    frames, metadata = preprocessor.preprocess(dummy_video)
    
    assert isinstance(frames, np.ndarray)
    assert frames.dtype == np.float32
    assert 'num_frames' in metadata
    assert 'shape' in metadata
    assert metadata['num_frames'] > 0


def test_handle_short_video(preprocessor):
    """Test xử lý video ngắn"""
    short_frames = np.random.rand(5, 288, 512, 3).astype(np.float32)
    
    padded = preprocessor.handle_short_video(short_frames, min_frames=15)
    
    assert len(padded) >= 15
    assert padded.shape[1:] == short_frames.shape[1:]


def test_extract_frames_invalid_video(preprocessor):
    """Test với video không tồn tại"""
    with pytest.raises(FileNotFoundError):
        preprocessor.extract_frames('nonexistent.mp4')