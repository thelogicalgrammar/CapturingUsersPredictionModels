"""
Custom PyMC distributions used by the agent models.

At present this module provides a native non-central chi-squared distribution,
which PyMC does not ship out of the box. It exists so that the single-prediction
collapsed model can sample the plate auxiliary ``R`` directly from its target
law instead of rebuilding that law by hand through the central-chi-squared
(Cochran) construction. \n

The reduced auxiliary ``R`` of the collapsed model satisfies \n
:math:`n R \\sim \\chi'^2_k(\\lambda)` with :math:`\\lambda = n \\Delta_X^2`, \n
i.e. a non-central chi-squared with ``k`` degrees of freedom and non-centrality
:math:`\\lambda`, rescaled by the plate size ``n``. Cochran's theorem rewrites
that same law as :math:`(\\Delta_X + G/\\sqrt{n})^2 + Z/n` with
:math:`G \\sim \\mathcal{N}(0, 1)` and :math:`Z \\sim \\chi^2_{k-1}`; the two are
distributionally identical, so exposing the non-central chi-squared as a first
class distribution removes the auxiliary variables ``G`` and ``Z`` and lets the
model read one-to-one off the derivation. See the "Single-prediction collapsed
reduction" document of the paper derivations.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.shape_utils import rv_size_is_none
from scipy import stats as st


# Below this non-centrality the density is indistinguishable from a central
# chi-squared (they differ by O(lam)), and the log-density has a removable
# lam -> 0 singularity that evaluates to a NaN at exactly lam = 0. Because the
# collapsed model feeds lam = n * Delta_X^2 and Delta_X can pass through 0
# during warmup, we switch to the central branch below this threshold.
LAM_EPS = 1e-8


def _central_chi2_logp(value, nu):
    """
    Log density of a central chi-squared, the ``lam -> 0`` limit of the
    non-central chi-squared.

    Args:
        value: Point at which to evaluate the density (``value > 0``).
        nu: Degrees of freedom.

    Returns:
        The log probability density (unclipped on the support).
    """
    return (
        -(nu / 2.0) * np.log(2.0)
        - pt.gammaln(nu / 2.0)
        + (nu / 2.0 - 1.0) * pt.log(value)
        - value / 2.0
    )


def ncx2_logp(value, nu, lam):
    """
    Log density of a non-central chi-squared distribution.

    Matches ``scipy.stats.ncx2(df=nu, nc=lam).logpdf`` to machine precision. The
    density is \n
    :math:`f(x) = \\tfrac12 e^{-(x+\\lambda)/2} (x/\\lambda)^{\\nu/4 - 1/2}
    I_{\\nu/2 - 1}(\\sqrt{\\lambda x})` \n
    for :math:`x > 0`. It is evaluated through the exponentially scaled Bessel
    function ``ive`` using :math:`\\log I_v(z) = \\log \\mathrm{ive}(v, z) + z`,
    which stays finite for the large arguments that arise when ``lam`` is large.
    The Bessel order :math:`\\nu/2 - 1` may be a negative half-integer (e.g.
    ``nu = 1``); ``ive`` handles that case directly.

    Args:
        value: Point at which to evaluate the density (support ``value > 0``).
        nu: Degrees of freedom.
        lam: Non-centrality parameter (``lam >= 0``).

    Returns:
        The log probability density, ``-inf`` off the support.
    """
    # Clamp lam INSIDE the non-central branch so that, even though the switch
    # discards this branch for tiny lam, its gradient never differentiates
    # through log(lam = 0) and leaks a NaN back into the sampler.
    lam_safe = pt.switch(lam < LAM_EPS, LAM_EPS, lam)
    z = pt.sqrt(lam_safe * value)
    logp_ncx = (
        -np.log(2.0)
        - (value + lam_safe) / 2.0
        + (nu / 4.0 - 0.5) * (pt.log(value) - pt.log(lam_safe))
        + pt.log(pt.math.ive(nu / 2.0 - 1.0, z))
        + z
    )
    logp = pt.switch(lam < LAM_EPS, _central_chi2_logp(value, nu), logp_ncx)
    return pt.switch(value > 0, logp, -np.inf)


def _ncx2_random(nu, lam, rng=None, size=None):
    """
    Draw from the non-central chi-squared for prior/posterior predictive
    sampling, delegating to ``scipy.stats.ncx2``.

    Args:
        nu: Degrees of freedom.
        lam: Non-centrality parameter.
        rng: NumPy random generator supplied by PyMC.
        size: Output shape supplied by PyMC (``None`` broadcasts the params).

    Returns:
        np.ndarray: Samples with the requested shape.
    """
    nu = np.asarray(nu)
    lam = np.asarray(lam)
    if size is None:
        size = np.broadcast(nu, lam).shape
    return st.ncx2.rvs(df=nu, nc=lam, size=size, random_state=rng)


def _ncx2_support_point(rv, size, nu, lam):
    """
    Initial-point function for the non-central chi-squared.

    Returns the distribution mean ``nu + lam``, which is strictly positive and
    therefore always inside the support. Without it PyMC would fall back to a
    zero initial value, which lies outside the support ``x > 0`` and makes the
    initial model log density ``-inf``.

    Args:
        rv: The random variable (unused, required by the PyMC signature).
        size: Requested output shape.
        nu: Degrees of freedom.
        lam: Non-centrality parameter.

    Returns:
        The support point, broadcast to ``size`` when given.
    """
    mean = nu + lam
    if not rv_size_is_none(size):
        mean = pt.full(size, mean)
    return mean


def NoncentralChiSquared(name, nu, lam, **kwargs):
    """
    A non-central chi-squared random variable for use inside a PyMC model.

    Thin :func:`pm.CustomDist` wrapper pairing :func:`ncx2_logp` (used by NUTS,
    differentiable in both ``value`` and ``lam``) with :func:`_ncx2_random`
    (used by predictive sampling). Parametrized like
    ``scipy.stats.ncx2(df=nu, nc=lam)``. The variable is sampled on the log
    scale (like PyMC's own positive distributions, e.g. ``ChiSquared``), so
    NUTS never proposes values outside the support ``x > 0``, and its initial
    value is the distribution mean ``nu + lam``.

    Args:
        name (str): Name of the random variable in the model.
        nu: Degrees of freedom (may be a constant or a model quantity).
        lam: Non-centrality parameter (may be a constant or a model quantity).
        **kwargs: Forwarded to :func:`pm.CustomDist` (e.g. ``observed``,
            ``shape``, ``dims``).

    Returns:
        The PyMC random variable.
    """
    return pm.CustomDist(
        name,
        nu,
        lam,
        logp=ncx2_logp,
        random=_ncx2_random,
        support_point=_ncx2_support_point,
        signature="(),()->()",
        default_transform=pm.distributions.transforms.log,
        **kwargs,
    )
