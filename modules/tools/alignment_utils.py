import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import splprep, splev

def calculate_signature(contour):
    """
    Calculate signature vector for a contour
    """
    # Calculate contour center
    center = contour.mean(axis=0)
    
    # Calculate distance and angle for each point
    distances = np.linalg.norm(contour - center, axis=1)
    angles = np.arctan2(contour[:, 1] - center[1], 
                       contour[:, 0] - center[0])
    
    # Combine distance and angle information
    signature = np.column_stack([distances, angles])
    
    return signature

def circular_cross_correlation(signature_a, signature_b):
    """
    Calculate circular cross-correlation between two contour signatures using FFT
    """
    # Ensure inputs are 1D arrays
    signature_a = np.asarray(signature_a).flatten()
    signature_b = np.asarray(signature_b).flatten()
    
    # Normalize signatures
    signature_a = (signature_a - np.mean(signature_a)) / np.std(signature_a)
    signature_b = (signature_b - np.mean(signature_b)) / np.std(signature_b)
    
    # Calculate FFT
    fft_a = fft(signature_a)
    fft_b = fft(signature_b)
    
    # Calculate cross-correlation
    corr = ifft(fft_a * np.conj(fft_b)).real
    
    # Find optimal shift
    shift = np.argmax(corr)
    
    # Calculate correlation score
    correlation_score = corr[shift] / len(signature_a)
    
    return shift, correlation_score

def estimate_rigid(source_points, target_points):
    """
    Calculate optimal rigid transformation using Procrustes analysis
    """
    # Center the point sets
    source_centered = source_points - np.mean(source_points, axis=0)
    target_centered = target_points - np.mean(target_points, axis=0)
    
    # Calculate covariance matrix
    H = source_centered.T @ target_centered
    
    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Calculate rotation matrix
    R = Vt.T @ U.T
    
    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Calculate translation vector
    t = np.mean(target_points, axis=0) - R @ np.mean(source_points, axis=0)
    
    return R, t

def resample_contour(contour, n_points=100):
    """
    Resample contour using arc-length parameterization
    """
    # Calculate arc lengths
    diff = np.diff(contour, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    total_length = np.sum(segment_lengths)
    
    # Calculate cumulative arc lengths
    cum_length = np.cumsum(segment_lengths)
    cum_length = np.insert(cum_length, 0, 0)
    
    # Generate uniformly distributed points
    target_lengths = np.linspace(0, total_length, n_points, endpoint=False)
    
    # Interpolate new points
    new_points = []
    for target_length in target_lengths:
        # Find segment containing target length
        idx = np.searchsorted(cum_length, target_length) - 1
        if idx < 0:
            idx = 0
        
        # Calculate position on segment
        segment_ratio = (target_length - cum_length[idx]) / segment_lengths[idx]
        new_point = contour[idx] + segment_ratio * (contour[idx + 1] - contour[idx])
        new_points.append(new_point)
    
    return np.array(new_points)

def detect_alignment_outliers(correlation_matrix, threshold_factor=2.0):
    """
    Detect alignment outliers using correlation matrix
    """
    # Calculate alignment cost for each section
    costs = np.mean(np.abs(correlation_matrix), axis=1)
    
    # Calculate cost threshold
    threshold = np.percentile(costs, 95) * threshold_factor
    
    # Identify outliers
    outliers = np.where(costs > threshold)[0]
    
    # Calculate outlier scores
    outlier_scores = costs[outliers] / threshold
    
    return outliers, outlier_scores, costs

def create_template_bounding_box(contour, padding=3):
    """
    Create a tight bounding box around the contour
    """
    min_x = np.min(contour[:, 0]) - padding
    max_x = np.max(contour[:, 0]) + padding
    min_y = np.min(contour[:, 1]) - padding
    max_y = np.max(contour[:, 1]) + padding
    
    corners = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])
    
    return corners 