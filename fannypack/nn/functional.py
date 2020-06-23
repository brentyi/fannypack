import numpy as np
import torch


def quadratic_matmul(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Computes $x^\top A x$, with support for arbitrary batch axes.

    Stolen from @alberthli/@wuphilipp.

    Args:
        x (torch.Tensor): Vectors. Shape should be `(*, D)`.
        A (torch.Tensor): Matrices. Shape should be `(*, D, D)`.

    Returns:
        torch.Tensor: Batched output of multiplication. Shape should be `(...)`.
    """
    assert x.shape[-1] == A.shape[-1] == A.shape[-2]

    x_T = x.unsqueeze(-2)  # shape=(*, 1, X)
    x_ = x.unsqueeze(-1)  # shape(*, X, 1)
    quadratic = x_T @ A @ x_  # shape=(*, 1, 1)
    return quadratic.squeeze(-1).squeeze(-1)


def gaussian_log_prob(
    mean: torch.Tensor, covariance: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Computes log probabilities under multivariate Gaussian distributions, with
    support for arbitrary batch axes.

    Equivalent to...
    ```
    torch.distributions.MultivariateNormal(
        mean, covariance
    ).log_prob(value)
    ```
    but avoids some CUDA errors (https://discuss.pytorch.org/t/cuda-illegal-memory-access-when-using-batched-torch-cholesky/51624).

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
