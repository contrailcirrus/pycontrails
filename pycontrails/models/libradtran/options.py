"""Default options for a variety of types of calculations."""


def get_default_options(calculation: str) -> dict[str, str]:
    """Get set of default libRadtran options."""

    if calculation == "thermal radiance":
        return {
            "rte_solver": "disort",
            "source": "thermal",
            "mol_abs_param": "reptran fine",
            "number_of_streams": "16",
            "zout": "TOA",
            "z_interpolate h2o": "spline",
            "z_interpolate o3": "spline",
            "umu": "1",
            "phi": "0",
            "output_user": "lambda uu",
        }

    if calculation == "thermal irradiance":
        return {
            "rte_solver": "twostr",
            "source": "thermal",
            "mol_abs_param": "fu",
            "zout": "TOA",
            "z_interpolate h2o": "spline",
            "z_interpolate o3": "spline",
            "output_user": "edir eglo edn eup",
        }
    msg = f"No default options available for {calculation} calculation."
    raise ValueError(msg)
