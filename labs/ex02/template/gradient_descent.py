# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent

Kept in sync with ``ex02.ipynb``; see the Wrap-Up section of that notebook.
"""

import numpy as np

from costs import compute_error, compute_loss


def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the gradient of the MSE at w.

    Implements equation (7) of exercise02.pdf:
        grad L(w) = -1/N * tx.T @ e,   with e = y - tx @ w.

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the
        loss at w.
    """
    e = compute_error(y, tx, w)
    return -tx.T @ e / len(y)


def gradient_descent(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
    verbose: bool = True,
) -> tuple[list[float], list[np.ndarray]]:
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        initial_w: shape=(2, ). The initial guess for the model parameters.
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.
        verbose: whether to print the loss and parameters at each iteration.

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for
            each iteration of GD.
        ws: a list of length max_iters + 1 containing the model parameters as numpy
            arrays of shape (2, ), for each iteration of GD (plus the final weights).
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, loss_type="mse")
        gradient = compute_gradient(y, tx, w)

        w = w - gamma * gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        if verbose:
            print(
                "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )

    return losses, ws
