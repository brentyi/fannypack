import numpy as np
import torch

import fannypack.utils


def test_cholupdate():
    """Checks rank-1 Cholesky update forward pass."""
    batch_dims = (1,)  # (5, 3, 7)
    matrix_dim = 2
    L = fannypack.utils.tril_from_vector(
        torch.randn(
            batch_dims + (fannypack.utils.tril_count_from_matrix_dim(matrix_dim),)
        )
    )
    x = torch.randn(batch_dims + (matrix_dim,))

    updated_L = fannypack.utils.cholupdate(L, x)

    torch.testing.assert_allclose(
        L @ L.transpose(-1, -2) + x[..., :, None] @ x[..., None, :],
        updated_L @ updated_L.transpose(-1, -2),
    )


def test_cholupdate_negative():
    """Checks rank-1 Cholesky update forward pass, with weight set to -2."""
    batch_dims = tuple()
    matrix_dim = 3
    L = fannypack.utils.tril_from_vector(
        torch.randn(
            batch_dims + (fannypack.utils.tril_count_from_matrix_dim(matrix_dim),)
        )
    ) + torch.eye(matrix_dim)
    x = torch.ones(matrix_dim) * 0.2

    # Make sure our output will be PSD
    L = torch.cholesky(L @ L.transpose(-1, -2) + x[..., :, None] @ x[..., None, :])

    updated_L = fannypack.utils.cholupdate(
        L, x, weight=torch.tensor(-0.5, dtype=torch.float32)
    )

    torch.testing.assert_allclose(
        L @ L.transpose(-1, -2) - 0.5 * x[..., :, None] @ x[..., None, :],
        updated_L @ updated_L.transpose(-1, -2),
    )


def test_cholupdate_negative_float_weight():
    """Checks rank-1 Cholesky update forward pass, with weight set to -2."""
    batch_dims = tuple()
    matrix_dim = 3
    L = fannypack.utils.tril_from_vector(
        torch.randn(
            batch_dims + (fannypack.utils.tril_count_from_matrix_dim(matrix_dim),)
        )
    ) + torch.eye(matrix_dim)
    x = torch.ones(matrix_dim) * 0.2

    # Make sure our output will be PSD
    L = torch.cholesky(L @ L.transpose(-1, -2) + x[..., :, None] @ x[..., None, :])

    updated_L = fannypack.utils.cholupdate(L, x, weight=-0.5)

    torch.testing.assert_allclose(
        L @ L.transpose(-1, -2) - 0.5 * x[..., :, None] @ x[..., None, :],
        updated_L @ updated_L.transpose(-1, -2),
    )


def test_cholupdate_backward():
    """Smoke test for rank-1 Cholesky update backward pass."""
    torch.autograd.set_detect_anomaly(True)
    batch_dims = (5, 3, 7)
    matrix_dim = 5
    L = fannypack.utils.tril_from_vector(
        torch.randn(
            batch_dims + (fannypack.utils.tril_count_from_matrix_dim(matrix_dim),),
            requires_grad=True,
        )
    )
    x = torch.randn(batch_dims + (matrix_dim,))

    updated_L = fannypack.utils.cholupdate(L, x)

    # If the Cholesky update is implemented incorrectly, we'll get an "inplace
    # operation" RuntimeError here
    torch.sum(updated_L).backward()


def test_quadratic_matmul():
    """Tests quadratic_matmul() by checking its backward pass."""
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
                    mean=mean,
                    covariance=covariance,
                    value=value,
                ),
            ]
        ),
        rtol=1e-3,
        atol=1e10,
    )


def test_tril_count_simple():
    """Basic counting utility check."""
    assert fannypack.utils.matrix_dim_from_tril_count(6) == 3
    assert fannypack.utils.tril_count_from_matrix_dim(3) == 6


def test_tril_count_bijective():
    """Check bijectivity of our counting utilities."""

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


def test_tril_inverse():
    """Check that our `tril_inverse` function correctly inverts some full-rank
    matrices.
    """
    for matrix_dim in range(2, 5):
        tril_matrix = fannypack.utils.tril_from_vector(
            torch.randn((5, fannypack.utils.tril_count_from_matrix_dim(matrix_dim)))
        )
        inverse = fannypack.utils.tril_inverse(tril_matrix)

        torch.testing.assert_allclose(tril_matrix @ inverse, torch.eye(matrix_dim))
