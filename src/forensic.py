# src/forensic.py
"""
Module forensic: Phân tích forensic (FFT/DCT, PRNU residual, optical flow)
"""

import cv2
import numpy as np
from scipy import fft, signal
from typing import Dict, Any, Optional, Tuple
import logging

from .utils import load_config, safe_divide


logger = logging.getLogger('hybrid_detector.forensic')


class ForensicAnalyzer:
    """
    Class phân tích forensic features từ frames
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Khởi tạo ForensicAnalyzer
        
        Args:
            config: Dictionary cấu hình
        """
        if config is None:
            config = load_config()
        
        self.forensic_config = config.get('forensic', {})
        self.fft_components = self.forensic_config.get('fft_components', 10)
        self.dct_components = self.forensic_config.get('dct_components', 10)
        self.spectrum_bins = self.forensic_config.get('spectrum_bins', 8)
        
        # PRNU config
        self.prnu_kernel = self.forensic_config.get('prnu_denoise_kernel', 5)
        self.prnu_method = self.forensic_config.get('prnu_method', 'bilateral')
        
        # Optical flow config
        self.flow_quality = self.forensic_config.get('optical_flow_quality', 0.01)
        self.flow_min_dist = self.forensic_config.get('optical_flow_min_distance', 10)
        self.flow_block_size = self.forensic_config.get('optical_flow_block_size', 7)
        
        logger.info("ForensicAnalyzer initialized")
    
    def compute_fft_features(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Tính toán FFT spectrum features
        
        Args:
            frames: Array frames (N, H, W, C) normalized [0, 1]
            
        Returns:
            Dictionary chứa FFT features
        """
        features = {}
        
        # Convert to grayscale nếu cần
        if frames.shape[-1] == 3:
            gray_frames = np.mean(frames, axis=-1)
        else:
            gray_frames = frames.squeeze(-1)
        
        fft_magnitudes = []
        
        for frame in gray_frames:
            # FFT 2D
            f_transform = fft.fft2(frame)
            f_shift = fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Log magnitude để tránh giá trị quá lớn
            magnitude_log = np.log1p(magnitude)
            fft_magnitudes.append(magnitude_log)
        
        fft_stack = np.array(fft_magnitudes)
        
        # Thống kê trên FFT spectrum
        features['fft_mean'] = float(np.mean(fft_stack))
        features['fft_std'] = float(np.std(fft_stack))
        features['fft_max'] = float(np.max(fft_stack))
        
        # High frequency energy (góc phải trên của spectrum)
        h, w = fft_stack[0].shape
        high_freq_region = fft_stack[:, :h//4, :w//4]
        features['fft_high_freq_energy'] = float(np.mean(high_freq_region))
        
        # Radial profile
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        r_int = r.astype(int)
        
        radial_profile = []
        for i in range(self.spectrum_bins):
            mask = (r_int == i * (max(h, w) // (2 * self.spectrum_bins)))
            if np.any(mask):
                radial_profile.append(np.mean(fft_stack[:, mask]))
        
        if len(radial_profile) > 0:
            features['fft_radial_slope'] = float(
                np.polyfit(range(len(radial_profile)), radial_profile, 1)[0]
            )
        else:
            features['fft_radial_slope'] = 0.0
        
        logger.debug(f"FFT features: {features}")
        return features
    
    def compute_dct_features(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Tính toán DCT (Discrete Cosine Transform) features
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary chứa DCT features
        """
        features = {}
        
        if frames.shape[-1] == 3:
            gray_frames = np.mean(frames, axis=-1)
        else:
            gray_frames = frames.squeeze(-1)
        
        dct_coeffs = []
        
        for frame in gray_frames:
            # DCT 2D using scipy
            dct_frame = fft.dctn(frame, norm='ortho')
            dct_coeffs.append(dct_frame)
        
        dct_stack = np.array(dct_coeffs)
        
        # Thống kê DCT
        features['dct_mean'] = float(np.mean(dct_stack))
        features['dct_std'] = float(np.std(dct_stack))
        
        # DC component (góc trái trên)
        features['dct_dc_mean'] = float(np.mean(dct_stack[:, 0, 0]))
        
        # AC energy (tất cả trừ DC)
        ac_coeffs = dct_stack.copy()
        ac_coeffs[:, 0, 0] = 0
        features['dct_ac_energy'] = float(np.mean(np.abs(ac_coeffs)))
        
        logger.debug(f"DCT features: {features}")
        return features
    
    def compute_prnu_residual(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Tính PRNU (Photo Response Non-Uniformity) residual approximation
        
        Phương pháp đơn giản: denoise frame -> residual = frame - denoised
        Không phải full sensor fingerprint nhưng đủ để phát hiện patterns
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary chứa PRNU features
        """
        features = {}
        
        # Convert to uint8 cho denoising
        frames_uint8 = (frames * 255).astype(np.uint8)
        
        if frames.shape[-1] == 3:
            gray_frames = np.mean(frames_uint8, axis=-1).astype(np.uint8)
        else:
            gray_frames = frames_uint8.squeeze(-1)
        
        residuals = []
        
        for frame in gray_frames:
            # Denoising
            if self.prnu_method == 'bilateral':
                denoised = cv2.bilateralFilter(
                    frame, self.prnu_kernel, 75, 75
                )
            else:  # gaussian
                denoised = cv2.GaussianBlur(
                    frame, (self.prnu_kernel, self.prnu_kernel), 0
                )
            
            # Compute residual
            residual = frame.astype(np.float32) - denoised.astype(np.float32)
            residuals.append(residual)
        
        residuals = np.array(residuals)
        
        # Thống kê residual
        features['prnu_mean'] = float(np.mean(np.abs(residuals)))
        features['prnu_std'] = float(np.std(residuals))
        
        # Autocorrelation của residual (indicator của sensor pattern)
        # Tính autocorr cho frame đầu tiên
        if len(residuals) > 0:
            r = residuals[0]
            autocorr = signal.correlate2d(r, r, mode='same')
            autocorr_center = autocorr[autocorr.shape[0]//2, autocorr.shape[1]//2]
            
            # Normalize
            if autocorr_center != 0:
                autocorr_norm = autocorr / autocorr_center
                # Peak ở vị trí offset nhỏ
                h, w = autocorr_norm.shape
                offset_region = autocorr_norm[h//2-5:h//2+5, w//2-5:w//2+5]
                features['prnu_autocorr'] = float(np.mean(np.abs(offset_region)))
            else:
                features['prnu_autocorr'] = 0.0
        else:
            features['prnu_autocorr'] = 0.0
        
        # Temporal consistency của residual
        if len(residuals) > 1:
            temporal_var = np.var([np.mean(r) for r in residuals])
            features['prnu_temporal_consistency'] = float(temporal_var)
        else:
            features['prnu_temporal_consistency'] = 0.0
        
        logger.debug(f"PRNU features: {features}")
        return features
    
    def compute_optical_flow(self, frames: np.ndarray) -> Dict[str, float]:
        """
        Tính optical flow statistics giữa consecutive frames
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary chứa optical flow features
        """
        features = {}
        
        # Convert to uint8 grayscale
        frames_uint8 = (frames * 255).astype(np.uint8)
        
        if frames.shape[-1] == 3:
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_uint8]
        else:
            gray_frames = [f.squeeze(-1) for f in frames_uint8]
        
        flow_magnitudes = []
        flow_angles = []
        
        for i in range(len(gray_frames) - 1):
            prev_frame = gray_frames[i]
            next_frame = gray_frames[i + 1]
            
            # Sử dụng Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, next_frame,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Tính magnitude và angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            flow_magnitudes.append(mag)
            flow_angles.append(ang)
        
        if len(flow_magnitudes) == 0:
            # Video quá ngắn
            features['flow_mean_magnitude'] = 0.0
            features['flow_std_magnitude'] = 0.0
            features['flow_smoothness'] = 0.0
            features['flow_temporal_consistency'] = 0.0
            return features
        
        # Stack flows
        mag_stack = np.array(flow_magnitudes)
        ang_stack = np.array(flow_angles)
        
        # Statistics
        features['flow_mean_magnitude'] = float(np.mean(mag_stack))
        features['flow_std_magnitude'] = float(np.std(mag_stack))
        
        # Smoothness: variance của spatial gradients
        mag_grad_x = np.gradient(mag_stack, axis=2)
        mag_grad_y = np.gradient(mag_stack, axis=1)
        smoothness = 1.0 / (1.0 + np.mean(mag_grad_x**2 + mag_grad_y**2))
        features['flow_smoothness'] = float(smoothness)
        
        # Temporal consistency: correlation giữa consecutive flows
        if len(mag_stack) > 1:
            correlations = []
            for i in range(len(mag_stack) - 1):
                corr = np.corrcoef(mag_stack[i].flatten(), mag_stack[i+1].flatten())[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if len(correlations) > 0:
                features['flow_temporal_consistency'] = float(np.mean(correlations))
            else:
                features['flow_temporal_consistency'] = 0.0
        else:
            features['flow_temporal_consistency'] = 0.0
        
        logger.debug(f"Optical flow features: {features}")
        return features
    
    def analyze(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        Phân tích forensic đầy đủ: FFT + DCT + PRNU + Optical Flow
        
        Args:
            frames: Array frames normalized
            
        Returns:
            Dictionary gộp tất cả forensic features
        """
        logger.info("Bắt đầu forensic analysis...")
        
        all_features = {}
        
        # FFT
        fft_feats = self.compute_fft_features(frames)
        all_features.update(fft_feats)
        
        # DCT
        dct_feats = self.compute_dct_features(frames)
        all_features.update(dct_feats)
        
        # PRNU
        prnu_feats = self.compute_prnu_residual(frames)
        all_features.update(prnu_feats)
        
        # Optical Flow
        flow_feats = self.compute_optical_flow(frames)
        all_features.update(flow_feats)
        
        logger.info(f"Forensic analysis hoàn tất, {len(all_features)} features")
        return all_features