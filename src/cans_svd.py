from functools import partial
from typing import List

import jax
import jax.numpy as jnp
from jax.lax.linalg import EighImplementation


from src.cans_iteration import cans_iteration


@partial(
    jax.jit,
    static_argnames=[
        "max_iter",
        "degree",
        "preprocess",
        "preprocess_iters",
        "delta",
        "a",
        "eigh_impl",
        "cans_tol",
        "eps_qr",
    ],
)
def cans_svd(
    matrix: jnp.array,
    max_iter: int = 50,
    degree: int = 3,
    preprocess: bool = True,
    preprocess_iters: int = 2,
    delta: float = 0.99,
    a: float = 0,
    eigh_impl: EighImplementation | None = None,
    cans_tol: float = 1e-5,
    eps_qr: float = 1e-5,
) -> List[jnp.array]:
    """
    Compute the SVD of a matrix using the CANS method.

    Args:
        matrix (jnp.array): The input matrix.
        max_iter (int): Number of iterations for the CANS method.
        degree (int): The degree of the polynomial approximation.
        preprocess (bool): Whether to use preprocessing.
        preprocess_iters (int): Number of preprocessing iterations.
        delta (float): The delta parameter for CANS.
        a (float): The scaling parameter for CANS.
        eigh_impl (EighImplementation): Algorithm for finding eigh in JAX.
        cans_tol (float): Tolerance for CANS algorithm.
        eps_qr (float): Tolerance for checking rank deficient cases.
    Returns:
        U: jnp.array: Left singular vectors.
        S: jnp.array: Singular values.
        Vt: jnp.array: Right singular vectors (transposed).
    """
    n_start = matrix.shape[0]
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    W = cans_iteration(
        matrix,
        max_iter=max_iter,
        a=a,
        degree=degree,
        preprocess=preprocess,
        preprocess_iters=preprocess_iters,
        delta=delta,
        tol=cans_tol,
    )

    H = W.T @ matrix
    V, s = jax.lax.linalg.eigh(
        H,
        symmetrize_input=True,
        implementation=eigh_impl,
    )
    U = W @ V

    # reverse the order of singular values and corresponding singular vectors
    U = U[:, jnp.arange(U.shape[-1] - 1, -1, -1)]
    V = V[:, jnp.arange(V.shape[-1] - 1, -1, -1)]
    s = s[jnp.arange(s.shape[-1] - 1, -1, -1)]

    # this implementation of QR decomposition can change sign of columns of U,
    # so we need to fix that to ensure singular values are positive
    def perform_qr(U, s):
        U, R = jax.lax.linalg.qr(U, full_matrices=False)
        s = jnp.diag(R) * s
        return U, s
    U, s = jax.lax.cond(
        jnp.any(jax.lax.abs(jnp.linalg.norm(U, axis=0) - 1) > eps_qr),
        perform_qr,
        lambda U, s: (U, s),
        U,
        s,
    )
    V = jnp.where(s < 0, -V, V)
    s = jnp.abs(s)
    idx = jnp.argsort(s, descending=True)
    U = U.at[:, jnp.arange(s.shape[0])].set(U[:, idx])
    V = V[:, idx]
    s = s[idx]

    if n_start == matrix.shape[0]:
        return U, s, V.T 
    else:
        return V, s, U.T
