#!/usr/bin/env python
"""
Evaluation metrics for crop yield prediction.

This module provides metrics for evaluating model performance:
- RMSE: Root Mean Square Error
- R²: Coefficient of Determination  
- PCC: Pearson Correlation Coefficient
"""

from typing import Tuple

import numpy as np
from scipy.stats import pearsonr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² value
    """
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


def pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Pearson Correlation Coefficient.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Correlation coefficient
    """
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Tuple of (RMSE, R², PCC)
    """
    return rmse(y_true, y_pred), r2_score(y_true, y_pred), pcc(y_true, y_pred)


# Legacy function names for backward compatibility
RMSE = rmse
R2_Score = r2_score
PCC = pcc


if __name__ == '__main__':
    """Test the metrics."""
    y = np.asarray([10, 20, 30, 40, 50])
    y_hat = np.asarray([11, 21, 32, 41, 51])

    print(f"RMSE: {rmse(y, y_hat):.4f}")
    print(f"R²: {r2_score(y, y_hat):.4f}")
    print(f"PCC: {pcc(y, y_hat):.4f}")
