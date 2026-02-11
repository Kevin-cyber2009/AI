# src/features.py
"""
Module features: Aggregator tổng hợp forensic + reality features thành vector
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

from .preprocessing import VideoPreprocessor
from .forensic import ForensicAnalyzer
from .reality_engine import RealityEngine
from .utils import load_config, normalize_array


logger = logging.getLogger('hybrid_detector.features')


class FeatureExtractor:
    """
    Class tổng hợp trích xuất features từ video
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo FeatureExtractor
        
        Args:
            config: Dictionary cấu hình
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.feature_config = config.get('features', {})
        self.expected_dim = self.feature_config.get('expected_dimension', 35)
        self.normalization = self.feature_config.get('normalization', 'standard')
        
        # Initialize sub-analyzers
        self.preprocessor = VideoPreprocessor(config)
        self.forensic = ForensicAnalyzer(config)
        self.reality = RealityEngine(config)
        
        logger.info(f"FeatureExtractor initialized, expected dim: {self.expected_dim}")
    
    def extract_features(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Trích xuất tất cả features từ frames
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary chứa tất cả features
        """
        all_features = {}
        
        # Forensic features
        forensic_feats = self.forensic.analyze(frames)
        all_features.update(forensic_feats)
        
        # Reality features
        reality_feats = self.reality.analyze(frames)
        all_features.update(reality_feats)
        
        logger.info(f"Extracted {len(all_features)} features total")
        
        return all_features
    
    def extract_from_video(self, video_path: str) -> Tuple[Dict[str, float], dict]:
        """
        Pipeline đầy đủ: video -> frames -> features
        
        Args:
            video_path: Đường dẫn video
            
        Returns:
            Tuple (features_dict, metadata)
        """
        logger.info(f"Extracting features from video: {video_path}")
        
        # Preprocess
        frames, metadata = self.preprocessor.preprocess(video_path)
        
        # Handle short videos
        if len(frames) < 10:
            logger.warning(f"Video ngắn ({len(frames)} frames), padding...")
            frames = self.preprocessor.handle_short_video(frames, min_frames=10)
        
        # Extract features
        features = self.extract_features(frames)
        
        return features, metadata
    
    def features_to_vector(
        self, 
        features: Dict[str, float],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Convert features dict thành numpy vector theo thứ tự cố định
        
        Args:
            features: Dictionary features
            feature_names: List tên features theo thứ tự (nếu None, tự động sort)
            
        Returns:
            1D numpy array
        """
        if feature_names is None:
            # Sort keys để đảm bảo thứ tự nhất quán
            feature_names = sorted(features.keys())
        
        vector = []
        for name in feature_names:
            value = features.get(name, 0.0)
            # Handle NaN/Inf
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            vector.append(value)
        
        return np.array(vector, dtype=np.float32)
    
    def normalize_features(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa feature vector
        
        Args:
            feature_vector: 1D array features
            
        Returns:
            Normalized array
        """
        return normalize_array(feature_vector, method=self.normalization)
    
    def get_feature_names(self) -> List[str]:
        """
        Lấy danh sách tên features theo thứ tự chuẩn
        
        Returns:
            List tên features
        """
        # Định nghĩa thứ tự features chuẩn
        forensic_names = [
            'fft_mean', 'fft_std', 'fft_max', 'fft_high_freq_energy', 'fft_radial_slope',
            'dct_mean', 'dct_std', 'dct_dc_mean', 'dct_ac_energy',
            'prnu_mean', 'prnu_std', 'prnu_autocorr', 'prnu_temporal_consistency',
            'flow_mean_magnitude', 'flow_std_magnitude', 'flow_smoothness', 'flow_temporal_consistency'
        ]
        
        reality_names = [
            'entropy_mean', 'entropy_std', 'entropy_slope',
            'fractal_dim_mean', 'fractal_dim_std',
            'causal_prediction_error', 'causal_predictability',
            'compression_mean', 'compression_std', 'compression_delta_mean', 'complexity_mean'
        ]
        
        return forensic_names + reality_names


from typing import Tuple  # Import thiếu