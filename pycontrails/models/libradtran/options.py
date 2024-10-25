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
            "umu": "1",
            "phi": "0",
            "output_user": "lambda uu",
        }

    if calculation == "thermal irradiance":
        return {
            "rte_solver": "disort",
            "source": "thermal",
            "mol_abs_param": "fu",
            "number_of_streams": "6",
            "zout": "TOA",
            "output_process": "sum",
            "output_user": "edir eglo edn eup enet esum",
        }
    msg = f"No default options available for {calculation} calculation."
    raise ValueError(msg)
