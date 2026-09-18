# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent

Kept in sync with ``ex02.ipynb``; see the Wrap-Up section of that notebook.
"""

import numpy as np

from costs import compute_error, compute_loss
from helpers import batch_iter


def compute_stoch_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute a stochastic gradient at w from a mini-batch of B examples.

    It is the same formula as the full gradient, only evaluated on the batch:
        grad L_B(w) = -1/B * tx.T @ e.

    Args:
        y: shape=(B, )
        tx: shape=(B, 2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic
        gradient of the loss at w.
    """
    e = compute_error(y, tx, w)
    return -tx.T @ e / len(y)


def stochastic_gradient_descent(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    batch_size: int,
    max_iters: int,
    gamma: float,
    verbose: bool = True,
) -> tuple[list[float], list[np.ndarray]]:
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        initial_w: shape=(2, ). The initial guess for the model parameters.
        batch_size: the number of data points in a mini-batch used for computing
            the stochastic gradient.
        max_iters: a scalar denoting the total number of iterations of SGD.
        gamma: a scalar denoting the stepsize.
        verbose: whether to print the loss and parameters at each iteration.

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for
            each iteration of SGD.
        ws: a list of length max_iters + 1 containing the model parameters as numpy
            arrays of shape (2, ), for each iteration of SGD.
    """
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = compute_loss(minibatch_y, minibatch_tx, w, loss_type="mse")
            stoch_gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)

            w = w - gamma * stoch_gradient

            # store w and loss
            ws.append(w)
            losses.append(loss)

        if verbose:
            print(
                "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )
    return losses, ws
