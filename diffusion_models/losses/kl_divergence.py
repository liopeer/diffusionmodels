import torch
from torch import nn, Tensor
from jaxtyping import Float

def gaussian_kl(
        p_mean: Float[Tensor, "1"], 
        p_var: Float[Tensor, "1"], 
        q_mean: Float[Tensor, "1"], 
        q_var: Float[Tensor, "1"]
    ) -> Float[Tensor, "1"]:
    """Calculate KL Divergence of 2 Gaussian distributions.

    KL divergence between two univariate Gaussians, as derived in [1]_, with k=1 (dimensionality).

    Parameters
    ----------
    p_mean
        mean value of first distribution
    p_var
        variance value of first distribution
    q_mean
        mean value of second distribution
    q_var
        variance value of second distribution

    Returns
    -------
    out
        KL divergence of inputs

    References
    ----------
    .. [1] https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    """
    return 0.5 * (torch.log(torch.abs(q_var) / torch.abs(p_var)) - 1.0 + ((p_mean-q_mean)**2)/q_var + p_var/q_var)

def log_gaussian_kl(
        p_mean: Float[Tensor, "1"], 
        p_logvar: Float[Tensor, "1"], 
        q_mean: Float[Tensor, "1"],
        q_logvar: Float[Tensor, "1"]
    ) -> Float[Tensor, "1"]:
    """Calculate KL Divergence of 2 Gaussian distributions.

    KL divergence between two univariate Gaussians, as derived in [1]_, with k=1 (dimensionality) and log variances.

    Parameters
    ----------
    p_mean
        mean value of first distribution
    p_logvar
        log of variance value of first distribution
    q_mean
        mean value of second distribution
    q_logvar
        log of variance value of second distribution

    Returns
    -------
    out
        KL divergence of inputs

    References
    ----------
    .. [1] https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    """
    return 0.5 * (q_logvar - p_logvar - 1.0 + torch.exp(p_logvar - q_logvar) + ((p_mean - q_mean)**2)*torch.exp(-q_logvar))