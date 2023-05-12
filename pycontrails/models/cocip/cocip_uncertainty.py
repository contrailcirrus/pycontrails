"""Parameters for CoCiP uncertainty calculations."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.stats.distributions import rv_frozen

from pycontrails.models.cocip import cocip_params
from pycontrails.models.cocip.cocip_params import CocipParams

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

    def rvs(self, *args: Any, **kwds: Any) -> npt.NDArray[np.float32]:
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
        npt.NDArray[np.float32]
            Sampled habit weight distribution with the same shape
            as :attr:`CocipParams().habit_distributions`
        """
        if args or (kwds.get("size") is not None and kwds["size"] > 1):
            raise ValueError("habit_dirichlet distribution only supports creating one rv at a time")

        default_distributions = cocip_params._habit_distributions()
        alpha_i = 0.5 + self.C * default_distributions

        # In the first distribution, we assume all ice particles are droxtals
        # There is no way to quantify the uncertainty in this assumption
        # Consequently, we leave the first distribution in default_distributions
        # alone, and only perturb the rest
        distr_list = [stats.dirichlet(a) for a in alpha_i[1:]]

        habit_weights = default_distributions.copy()
        for i, distr in enumerate(distr_list, start=1):
            habit_weights[i] = distr.rvs(**kwds)

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

    >>> # Override the 'initial_wake_vortex_depth' field from
    >>> # CocipParams with a uniform value in [0.4, 0.6]
    >>> distr = scipy.stats.uniform(loc=0.4, scale=0.2)
    >>> params = CocipUncertaintyParams(seed=123, initial_wake_vortex_depth_uncertainty=distr)
    >>> params.initial_wake_vortex_depth
    0.41076420

    >>> # Once seeded, calling the class again gives a new value
    >>> params = CocipUncertaintyParams(initial_wake_vortex_depth=distr)
    >>> params.initial_wake_vortex_depth
    0.43526372

    >>> # To retain the default value, set the uncertainty to None
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
    habit_distributions_uncertainty: rv_frozen | None = habit_dirichlet(C=96.0)

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

    def rvs(self, size: None | int = None) -> dict[str, float | npt.NDArray[np.float_]]:
        """Call each distribution's `rvs` method to generate random parameters.

        Seed calls to `rvs` with class variable `rng`.

        Parameters
        ----------
        size : None | int, optional
            If specified, an `array` of values is generated for each uncertainty parameter.

        Returns
        -------
        dict[str, float | npt.NDArray[np.float_]]
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
               [1.5554131e-02, 2.1363135e-01, 7.7715185e-03, 1.7690966e-02,
                3.1576434e-03, 3.2992734e-06, 7.4009895e-01, 2.0921326e-03],
               [3.8193921e-03, 2.1235342e-01, 3.3554080e-04, 5.2846869e-04,
                3.1945917e-01, 4.8709914e-04, 4.6250960e-01, 5.0730183e-04],
               [5.7327619e-04, 4.7781631e-01, 4.2596990e-03, 6.7235163e-04,
                1.4447135e-01, 3.6184600e-01, 1.0150939e-02, 2.1006212e-04],
               [1.5397545e-02, 4.0522218e-01, 4.2781001e-01, 1.4331797e-01,
                7.1088417e-04, 9.4511814e-04, 3.3900745e-03, 3.2062260e-03],
               [7.9063961e-04, 3.0336906e-03, 7.7571563e-04, 2.0577813e-02,
                9.4205803e-01, 4.3379897e-03, 3.6786550e-03, 2.4747452e-02]],
              dtype=float32),
         'initial_wake_vortex_depth': 0.39805019708566847,
         'nvpm_ei_n_enhancement_factor': 0.9371878437312526,
         'rf_lw_enhancement_factor': 1.1017491252832377,
         'rf_sw_enhancement_factor': 0.99721639115012,
         'sedimentation_impact_factor': 0.5071779847244678,
         'wind_shear_enhancement_exponent': 0.34100931239701004}
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
