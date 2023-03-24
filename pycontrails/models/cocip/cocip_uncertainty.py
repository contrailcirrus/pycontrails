"""Parameters for CoCiP uncertainty calculations."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import stats
from scipy.stats.distributions import rv_frozen

from pycontrails.models.cocip.cocip_params import CocipParams
from pycontrails.models.humidity_scaling import ExponentialBoostHumidityScaling

logger = logging.getLogger(__name__)


class habit_dirichlet(rv_frozen):
    r"""Custom dirichlet distribution for habit weight distribution uncertainty.

    Scale the habit distributions by a dirichlet distribution
    with alpha parameter :math:`\alpha_{i} = 0.5 + \text{C} \text{G}_{i}`
    where :math:`\text{G}_{i}` is the approximate habit weight distributions
    defined in :attr:`CocipParams().habit_distributions`.

    References
    ----------
    - Table 2 in :cite:`schumannEffectiveRadiusIce2011`
    """

    def __init__(self, C: float = 96.0):
        self.C = C

    def rvs(self, *args: Any, **kwds: Any) -> np.ndarray:
        """Generate sample set of habit distributions.

        Sampled using dirichlet distribution.

        Parameters
        ----------
        *args
            Used to create a number of habit distributions
        **kwds
            Passed through to :func:`scipy.stats.dirichlet.rvs()`

        Returns
        -------
        np.ndarray
            Sampled habit weight distribution with the same shape
            as :attr:`CocipParams().habit_distributions`
        """
        if args or ("size" in kwds and kwds["size"] is not None and kwds["size"] > 1):
            raise ValueError("habit_dirichlet distribution only supports creating one rv at a time")

        default_distributions = CocipParams().habit_distributions
        alpha_i = [0.5 + self.C * G_i for G_i in default_distributions]
        distr = [stats.dirichlet(a) for a in alpha_i]

        habit_weights = default_distributions.copy()
        for i in range(1, habit_weights.shape[0]):
            habit_weights[i] = distr[i].rvs(**kwds)

        return habit_weights


@dataclass
class CocipUncertaintyParams(CocipParams):
    """Model parameters for CoCiP epistemic uncertainty.

    Any uncertainty parameters should end with the suffix `uncertainty`. See `__post_init__`.

    Uncertainty parameters take a `scipy.stats.distributions.rv_frozen` distribution.

    Default distributions have mean at CocipParam default values.

    To retain specific parameters as defaults, set the value of the key to None.

    Examples
    --------
    >>> import scipy.stats
    >>> from pycontrails.models.cocip import CocipUncertaintyParams

    # Override the `rhi_adj` field from `CocipParams` with a value in [0.97, 0.99]
    >>> distr = scipy.stats.uniform(loc=0.97, scale=0.02)
    >>> params = CocipUncertaintyParams(seed=123, rhi_adj_uncertainty=distr)
    >>> params.rhi_adj
    0.98364703

    # Once seeded, calling the class again gives a new value
    >>> params = CocipUncertaintyParams(rhi_adj_uncertainty=distr)
    >>> params.rhi_adj
    0.98310570

    >>> params = CocipUncertaintyParams(rf_lw_enhancement_factor_uncertainty=None)
    >>> params.rf_lw_enhancement_factor
    1.0
    """

    #: The random number generator is attached to the class (as opposed to an instance).
    #: If many instances of :class:`CocipUncertaintyParams` are needed, it is best to seed
    #: the random number generator once initially. Reseeding the generator will also
    #: reset it, giving rise to duplicate random numbers.
    rng: ClassVar[np.random.Generator] = np.random.default_rng(None)

    #: Reseed the random generator defined in ``__post_init__``
    seed: int | None = None

    # This might need fixing!
    # Whenever we overhaul / redo an uncertainty analysis, we may need to change
    # how it interacts with a humidity scaler
    humidity_scaling: ExponentialBoostHumidityScaling = ExponentialBoostHumidityScaling()

    #: Parameters for specific humidity and RHi enhancement
    #: This assumes Cocip.Params.humidity_scaling is an
    #: :class:`ExponentialBoostHumidityScaling`` instance.
    rhi_adj_uncertainty: rv_frozen | None = stats.norm(
        loc=ExponentialBoostHumidityScaling.default_params.rhi_adj, scale=0.1
    )
    rhi_boost_exponent_uncertainty: rv_frozen = stats.triang(
        loc=1.0,
        c=ExponentialBoostHumidityScaling.default_params.rhi_boost_exponent - 1.0,
        scale=1.0,
    )

    #: Schumann takes ``wind_shear_enhancement_exponent`` = 0.5 and discusses the case of 0 and 2/3
    #: as possibilities.
    #: With a value of 0, wind shear is not enhanced.
    wind_shear_enhancement_exponent_uncertainty: rv_frozen | None = stats.triang(
        loc=0.0, c=CocipParams.wind_shear_enhancement_exponent, scale=1.0
    )

    #: Schumann takes ``initial_wake_vortex_depth`` = 0.5 and discusses some
    #: uncertainty in this value. This parameter should be non-negative.
    initial_wake_vortex_depth_uncertainty: rv_frozen | None = stats.triang(
        loc=0.3, c=CocipParams.initial_wake_vortex_depth, scale=0.4
    )

    #: Schumann takes a default value of 0.1 and describes it as an "important adjustable parameter"
    #: Currently, `CocipParams` uses a default value of 0.5
    sedimentation_impact_factor_uncertainty: rv_frozen | None = stats.norm(
        loc=CocipParams.sedimentation_impact_factor, scale=0.1
    )

    #: Teoh 2022 (to appear) takes values between 70% decrease and 100% increase.
    #: This coincides with the log normal distribution defined below.
    nvpm_ei_n_enhancement_factor_uncertainty: rv_frozen | None = stats.lognorm(
        s=0.15, scale=1 / stats.lognorm(s=0.15).mean()
    )

    #: Scale shortwave radiative forcing.
    #: Table 2 in :cite:`schumannParametricRadiativeForcing2012`
    #: provides relative RMS error for SW/LW fit to the data generated
    #: by `libRadTran <http://www.libradtran.org/doku.php>`_.
    #: We use the average RMS error across all habit types (pg 1397) as the standard deviation
    #: of a normally distributed scaling factor for SW forcing
    rf_sw_enhancement_factor_uncertainty: rv_frozen | None = stats.norm(
        loc=CocipParams.rf_sw_enhancement_factor, scale=0.106
    )

    #: Scale longwave radiative forcing.
    #: Table 2 in :cite:`schumannParametricRadiativeForcing2012` provides relative error for SW/LW
    #: fit to the data generated by `libRadTran <http://www.libradtran.org/doku.php>`_.
    #: We use the average RMS error across all habit types (pg 1397) as the standard deviation
    #: of a normally distributed scaling factor for LW forcing.
    rf_lw_enhancement_factor_uncertainty: rv_frozen | None = stats.norm(
        loc=CocipParams.rf_lw_enhancement_factor, scale=0.071
    )

    #: Scale the habit distributions by a dirichlet distribution
    #: with alpha parameter :math:`\alpha_{i} = 0.5 + \text{C} \text{G}_{i}`
    #: where :math:`\text{G}_{i}` is the approximate habit weight distributions
    #: defined in :attr:`CocipParams().habit_distributions`.
    #: Higher values of :math:`\text{C}` correspond to higher confidence in initial estimates.
    habit_distributions_uncertainty: rv_frozen | None = habit_dirichlet(C=96)

    def __post_init__(self) -> None:
        """Override values of model parameters according to ranges."""
        if self.seed is not None:
            # Reset the class variable `rng`
            logger.info("Reset %s random seed to %s", self.__class__.__name__, self.seed)
            self.__class__.rng = np.random.default_rng(self.seed)

        # Override defaults value on `CocipParams` with random parameters
        for param, value in self.rvs().items():
            setattr(self, param, value)

    @property
    def uncertainty_params(self) -> dict[str, rv_frozen]:
        """Get dictionary of uncertainty parameters.

        Method checks for attributes ending in `"_uncertainty"`.

        Returns
        -------
        dict[str, dict[str, Any]]
            Uncertainty parameters and values
        """
        # handle these differently starting in version 0.27.0
        exclude = {"rhi_adj", "rhi_boost_exponent"}

        out = {}

        param_dict = asdict(self)
        for uncertainty_param, dist in param_dict.items():
            if uncertainty_param.endswith("_uncertainty") and dist is not None:
                param = uncertainty_param.split("_uncertainty")[0]
                if param not in exclude and param not in param_dict:
                    raise AttributeError(
                        f"Parameter {param} corresponding to uncertainty parameter "
                        f"{uncertainty_param} does not exist"
                    )

                if not isinstance(dist, rv_frozen):
                    raise AttributeError(
                        f"Uncertainty parameter '{uncertainty_param}' must be instance of "
                        "'scipy.stats.distributions.rv_frozen'"
                    )

                out[param] = dist

        return out

    def rvs(self, size: None | int = None) -> dict[str, float | np.ndarray]:
        """Call each distribution's `rvs` method to generate random parameters.

        Seed calls to `rvs` with class variable `rng`.

        Parameters
        ----------
        size : None | int, optional
            If specified, an `array` of values is generated for each uncertainty parameter.

        Returns
        -------
        dict[str, float | np.ndarray]
            Dictionary of random parameters. Dictionary keys consists of names of parameters in
            `CocipParams` to be overridden by random value.

        Examples
        --------
        >>> from pprint import pprint
        >>> from pycontrails.models.cocip import CocipUncertaintyParams
        >>> params = CocipUncertaintyParams(seed=456)
        >>> pprint(params.rvs())
        {'habit_distributions': array([[0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
               0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00],
              [3.6062175e-04, 2.9157758e-01, 6.5901526e-04, 1.3415117e-03,
               5.0444747e-03, 2.6956137e-02, 6.7330402e-01, 7.5663364e-04],
              [2.3834489e-03, 3.6896354e-01, 1.3977488e-03, 4.7844747e-04,
               3.0474603e-01, 4.8765671e-03, 3.1715181e-01, 2.4031256e-06],
              [4.0944587e-03, 5.1772928e-01, 4.8383059e-05, 1.3166944e-03,
               1.0002965e-01, 3.7468293e-01, 3.9641498e-04, 1.7021718e-03],
              [1.5192166e-05, 3.8423434e-01, 5.0536448e-01, 9.4547205e-02,
               1.1378267e-03, 3.3812344e-03, 4.6305601e-03, 6.6891452e-03],
              [1.1620003e-03, 2.3161094e-03, 1.6705607e-04, 2.0091899e-02,
               9.7001791e-01, 1.5312615e-03, 3.9347797e-03, 7.7895907e-04]],
             dtype=float32),
        'initial_wake_vortex_depth': 0.4752143043559352,
        'nvpm_ei_n_enhancement_factor': 1.0094726146454185,
        'rf_lw_enhancement_factor': 0.9718354129386728,
        'rf_sw_enhancement_factor': 1.016450567639041,
        'rhi_adj': 1.0293074966694602,
        'rhi_boost_exponent': 1.3308280817638443,
        'sedimentation_impact_factor': 0.6433086271594898,
        'wind_shear_enhancement_exponent': 0.31735944665388505}
        """
        return {
            param: distr.rvs(size=size, random_state=self.rng)
            for param, distr in self.uncertainty_params.items()
        }

    def as_dict(self) -> dict[str, Any]:
        """Convert object to dictionary.

        Wrapper around :meth:`ModelBase.as_dict` that removes
        uncertainty specific parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary version of self.
        """
        obj = super().as_dict()

        # remove seed and _uncertainty attributes
        # these will throw an error in `ModelBase._load_params()`
        keys_to_remove = ["seed"]
        for key in obj:
            if key.endswith("_uncertainty"):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            obj.pop(key, None)

        return obj
