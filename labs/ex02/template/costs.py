# -*- coding: utf-8 -*-
"""Exercise 2.

Cost functions for linear regression (MSE and MAE).

Kept in sync with ``ex02.ipynb``; see the Wrap-Up section of that notebook.
"""

import numpy as np


def compute_error(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the error vector e = y - tx @ w.

    Args:
        y: numpy array of shape=(N, ). The output/target vector.
        tx: numpy array of shape=(N, 2). The input data matrix.
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (N, ) containing the residuals.
    """
    return y - tx @ w


def compute_mse(e: np.ndarray) -> float:
    """Compute the mean square error from the error vector.

    Implements L(w) = 1/(2N) * e.T @ e, i.e. equation (2) of exercise02.pdf.

    Args:
        e: numpy array of shape=(N, ). The error vector y - tx @ w.

    Returns:
        The MSE (a scalar).
    """
    return float(e.dot(e) / (2 * len(e)))


def compute_mae(e: np.ndarray) -> float:
    """Compute the mean absolute error from the error vector.

    Implements L(w) = 1/N * sum(|e_n|).

    Args:
        e: numpy array of shape=(N, ). The error vector y - tx @ w.

    Returns:
        The MAE (a scalar).
    """
    return float(np.mean(np.abs(e)))


def compute_loss(
    y: np.ndarray, tx: np.ndarray, w: np.ndarray, loss_type: str = "mse"
) -> float:
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, ). The output/target vector.
        tx: numpy array of shape=(N, 2). The input data matrix.
        w: numpy array of shape=(2, ). The vector of model parameters.
        loss_type: either ``"mse"`` (default) or ``"mae"``.

    Returns:
        The value of the loss (a scalar), corresponding to the parameters w.

    Raises:
        ValueError: If ``loss_type`` is neither ``"mse"`` nor ``"mae"``.

    Example:
        >>> compute_loss(np.array([1.0, 2.0]), np.ones((2, 2)), np.zeros(2))
        1.25
    """
    e = compute_error(y, tx, w)
    if loss_type == "mse":
        return compute_mse(e)
    if loss_type == "mae":
        return compute_mae(e)
    raise ValueError("Invalid loss_type. Choose either 'mse' or 'mae'.")
