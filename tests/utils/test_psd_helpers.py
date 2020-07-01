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
            [x.grad, ((A.transpose(-1, -2) + A) @ x[:, :, None]).squeeze(),]
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
