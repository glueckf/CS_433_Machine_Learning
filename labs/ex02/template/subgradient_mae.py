# -*- coding: utf-8 -*-
"""Problem Sheet 2, Exercise 6.

Subgradient descent for the MAE cost function.

Kept in sync with ``ex02.ipynb``; see the Wrap-Up section of that notebook.
"""

import numpy as np

from costs import compute_error, compute_loss
from helpers import batch_iter


def compute_subgradient_mae(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute a subgradient of the MAE at w.

    With L(w) = 1/N * sum(|e_n|) and e = y - tx @ w, the chain rule gives
        dL(w) = -1/N * tx.T @ s,   s_n in d|e_n|.

    ``np.sign`` returns 0 at e_n = 0, which is the valid subgradient choice at the
    non-differentiable points (0 lies in the subdifferential [-1, 1] of |.| at 0).

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing a subgradient of the
        MAE at w.
    """
    e = compute_error(y, tx, w)
    return -tx.T @ np.sign(e) / len(y)


def subgradient_descent(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
    decreasing_gamma: bool = False,
    verbose: bool = True,
) -> tuple[list[float], list[np.ndarray]]:
    """The SubGradient Descent (SubGD) algorithm for the MAE cost.

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        initial_w: shape=(2, ). The initial guess for the model parameters.
        max_iters: a scalar denoting the total number of iterations of SubGD.
        gamma: a scalar denoting the stepsize.
        decreasing_gamma: if True, use the step size gamma / sqrt(t + 1). A constant
            step size only reaches a neighbourhood of the optimum, because the MAE
            subgradient does not vanish there.
        verbose: whether to print the loss and parameters at each iteration.

    Returns:
        losses: a list of length max_iters containing the MAE value (scalar) for
            each iteration of SubGD.
        ws: a list of length max_iters + 1 containing the model parameters as numpy
            arrays of shape (2, ), for each iteration of SubGD.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w, loss_type="mae")
        subgradient = compute_subgradient_mae(y, tx, w)

        step = gamma / np.sqrt(n_iter + 1) if decreasing_gamma else gamma
        w = w - step * subgradient

        ws.append(w)
        losses.append(loss)
        if verbose:
            print(
                "SubGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )

    return losses, ws


def stochastic_subgradient_descent(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    batch_size: int,
    max_iters: int,
    gamma: float,
    decreasing_gamma: bool = False,
    verbose: bool = True,
) -> tuple[list[float], list[np.ndarray]]:
    """Stochastic SubGradient Descent (SubSGD) for the MAE cost.

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        initial_w: shape=(2, ). The initial guess for the model parameters.
        batch_size: the number of data points in a mini-batch used for computing
            the stochastic subgradient.
        max_iters: a scalar denoting the total number of iterations of SubSGD.
        gamma: a scalar denoting the stepsize.
        decreasing_gamma: if True, use the step size gamma / sqrt(t + 1).
        verbose: whether to print the loss and parameters at each iteration.

    Returns:
        losses: a list of length max_iters containing the mini-batch MAE value
            (scalar) for each iteration of SubSGD.
        ws: a list of length max_iters + 1 containing the model parameters as numpy
            arrays of shape (2, ), for each iteration of SubSGD.
    """
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = compute_loss(minibatch_y, minibatch_tx, w, loss_type="mae")
            subgradient = compute_subgradient_mae(minibatch_y, minibatch_tx, w)

            step = gamma / np.sqrt(n_iter + 1) if decreasing_gamma else gamma
            w = w - step * subgradient

            ws.append(w)
            losses.append(loss)

        if verbose:
            print(
                "SubSGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )
    return losses, ws
