import numpy as np
import torch

import fannypack.utils


def test_quadratic_matmul():
    """Tests quadratic_matmul() by checking its backward pass.
    """
    N = 100
    D = 5

    # Deterministic test
    torch.random.manual_seed(0)

    # Compute quadratic
    x = torch.randn((N, D), requires_grad=True)
    A = torch.randn((N, D, D))
    xTAx = fannypack.utils.quadratic_matmul(x, A)

    # Check shape
    assert xTAx.shape == (N,)

    # Check gradient
    torch.sum(xTAx).backward()
    np.testing.assert_allclose(
        *fannypack.utils.to_numpy(
            [x.grad, ((A.transpose(-1, -2) + A) @ x[:, :, None]).squeeze()]
        ),
        rtol=1e-4,
        atol=1e-6,
    )


def test_gaussian_log_prob():
    """Check that our Gaussian log probability implementation matches the native PyTorch
    one.
    """
    N = 100
    D = 5

    # Deterministic test
    torch.random.manual_seed(0)

    # Compute quadratic
    mean = torch.randn((N, D))
    value = torch.randn((N, D))
    covariance = torch.randn((N, D, D))
    covariance = covariance @ covariance.transpose(-1, -2)

    # Check output
    np.testing.assert_allclose(
        *fannypack.utils.to_numpy(
            [
                torch.distributions.MultivariateNormal(
                    loc=mean, covariance_matrix=covariance
                ).log_prob(value=value),
                fannypack.utils.gaussian_log_prob(
                    mean=mean, covariance=covariance, value=value,
                ),
            ]
        ),
        rtol=1e-3,
        atol=1e10,
    )


def test_tril_count_simple():
    """Basic counting utility check.
    """
    assert fannypack.utils.matrix_dim_from_tril_count(6) == 3
    assert fannypack.utils.tril_count_from_matrix_dim(3) == 6


def test_tril_count_bijective():
    """Check bijectivity of our counting utilities.
    """

    for matrix_dim in range(1, 100):
        assert (
            fannypack.utils.matrix_dim_from_tril_count(
                fannypack.utils.tril_count_from_matrix_dim(matrix_dim)
            )
            == matrix_dim
        )


def test_tril_vector_conversions():
    """Check that our `tril_to_vector` and `vector_to_tril` functions return the correct
    shapes + bijectivity.
    """

    for batch_dims in ((), (3, 2, 5), (7,)):
        for matrix_dim in range(1, 10):
            tril_count = fannypack.utils.tril_count_from_matrix_dim(matrix_dim)
            vectors = torch.randn(batch_dims + (tril_count,))
            trils = fannypack.utils.tril_from_vector(vectors)

            assert trils.shape == batch_dims + (matrix_dim, matrix_dim)
            assert torch.all(fannypack.utils.vector_from_tril(trils) == vectors)
