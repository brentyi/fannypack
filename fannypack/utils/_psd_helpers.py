import numpy as np
import torch


def quadratic_matmul(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    r"""Computes $x^\top A x$, with support for arbitrary batch axes.

    Stolen from @alberthli/@wuphilipp.

    Args:
        x (torch.Tensor): Vectors. Shape should be `(*, D)`.
        A (torch.Tensor): Matrices. Shape should be `(*, D, D)`.

    Returns:
        torch.Tensor: Batched output of multiplication. Shape should be `(*)`.
    """
    assert x.shape[-1] == A.shape[-1] == A.shape[-2]

    x_T = x.unsqueeze(-2)  # shape=(*, 1, X)
    x_ = x.unsqueeze(-1)  # shape(*, X, 1)
    quadratic = x_T @ A @ x_  # shape=(*, 1, 1)
    return quadratic.squeeze(-1).squeeze(-1)


def gaussian_log_prob(
    mean: torch.Tensor, covariance: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Computes log probabilities under multivariate Gaussian distributions,
    with support for arbitrary batch axes.

    Naive version of...
    ```
    torch.distributions.MultivariateNormal(
        mean, covariance
    ).log_prob(value)
    ```
    that avoids some Cholesky-related CUDA errors.
    https://discuss.pytorch.org/t/cuda-illegal-memory-access-when-using-batched-torch-cholesky/51624

    Stolen from @alberthli/@wuphilipp.

    Args:
        mean (torch.Tensor): Means vectors. Shape should be `(*, D)`.
        covariance (torch.Tensor): Covariances matrices. Shape should be `(*, D, D)`.
        value (torch.Tensor): State vectors. Shape should be `(*, D)`.

    Returns:
        torch.Tensor: Batched log probabilities. Shape should be `(*)`.
    """
    D = mean.shape[-1]
    assert covariance.shape[:-1] == mean.shape == value.shape

    exponential = quadratic_matmul(value - mean, torch.inverse(covariance))
    other_terms = torch.logdet(covariance) + D * np.log(2.0 * np.pi)
    log_p = -0.5 * (exponential + other_terms)
    return log_p


def matrix_dim_from_tril_count(tril_count: int):
    """Computes the dimension of a square matrix given its number of lower-triangular
    components.

    Args:
        tril_count (int): Count of lower-triangular terms.
    Returns:
        int: Dimension of square matrix.
    """
    matrix_dim = int(0.5 * (1 + 8 * tril_count) ** 0.5 - 0.5)
    return matrix_dim


def tril_count_from_matrix_dim(matrix_dim: int):
    """Computes the number of lower triangular terms in a square matrix of a given
    dimension `(matrix_dim, matrix_dim)`.

    Args:
        matrix_dim (int): Dimension of square matrix.
    Returns:
        int: Count of lower-triangular terms.
    """
    tril_count = (matrix_dim ** 2 - matrix_dim) // 2
    return tril_count


def vector_to_tril(lower_vector: torch.Tensor) -> torch.Tensor:
    """Computes lower-triangular square matrices from a flattened vector of nonzero
    terms. Supports arbitrary batch dimensions.

    Args:
        lower_vector (torch.Tensor): Vectors containing the nonzero terms of a
            square lower-triangular matrix. Shape should be (*, tril_count).
    Returns:
        torch.Tensor: Square matrices. Shape should be (*, matrix_dim, matrix_dim)
    """
    batch_dims = lower_vector.shape[:-1]
    tril_count = lower_vector.shape[-1]
    matrix_dim = matrix_dim_from_tril_count(tril_count)

    output = torch.zeros(batch_dims + (matrix_dim, matrix_dim))
    tril_indices = torch.tril_indices(matrix_dim, matrix_dim)
    output[..., tril_indices[0], tril_indices[1]] = lower_vector
    return output


def tril_to_vector(tril_matrix: torch.Tensor) -> torch.Tensor:
    """Retrieves the lower triangular terms of square matrices as vectors. Supports
    arbitrary batch dimensions.

    Args:
        tril_matrix (torch.Tensor): Square matrices. Shape should be
            (*, matrix_dim, matrix_dim)
    Returns:
        torch.Tensor: Flattened vectors. Shape should be (*, tril_count).
    """
    matrix_dim = tril_matrix.shape[-1]
    assert tril_matrix.shape[-2] == matrix_dim

    tril_indices = torch.tril_indices(matrix_dim, matrix_dim)
    return tril_matrix[..., tril_indices[0], tril_indices[1]]
