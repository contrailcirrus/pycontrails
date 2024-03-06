"""Raise ``ImportError`` when dependencies are not met."""

from __future__ import annotations

from typing import NoReturn


def raise_module_not_found_error(
    name: str,
    package_name: str,
    module_not_found_error: ImportError,
    pycontrails_optional_package: str | None = None,
    extra: str | None = None,
) -> NoReturn:
    """Raise ``ImportError`` with a helpful message.

    Parameters
    ----------
    name : str
        The name describing the context of the ``ImportError``. For example,
        if the module is required for a specific function, the name could be
        "my_function function". If the module is required for a specific method,
        the name could be "MyClass.my_method method". If the module is required
        for an entire ``pycontrails`` module, the name could be "my_module module".
    package_name : str
        The name of the package that is required. This should be the full name of
        the python package, which may be different from the name of the module
        that is actually imported. For example, if ``import sklearn`` triggers
        the ``ImportError``, the ``package_name`` should be "scikit-learn".
    module_not_found_error : ImportError
        The ``ImportError`` that was raised. This is passed to the
        ``from`` clause of the ``raise`` statement below. The subclass of the
        ``ImportError`` is preserved (e.g., ``ModuleNotFoundError`` or
        ``ImportError``).
    pycontrails_optional_package : str, optional
        The name of the optional ``pycontrails`` package that can be used to
        install the required package. See the ``pyproject.toml`` file.
    extra : str, optional
        Any extra information that should be included in the error message.
        This is appended to the end of the error message.
    """
    # Put the function or method or module name in quotes if the full name
    # contains a space.
    try:
        n1, n2 = name.split(" ")
    except ValueError:
        if "'" not in name:
            name = f"'{name}'"
    else:
        if "'" not in n1:
            n1 = f"'{n1}'"
        name = f"{n1} {n2}"

    msg = (
        f"The {name} requires the '{package_name}' package. "
        f"This can be installed with 'pip install {package_name}'"
    )
    if pycontrails_optional_package:
        msg = f"{msg} or 'pip install pycontrails[{pycontrails_optional_package}]'."
    else:
        msg = f"{msg}."

    if extra:
        msg = f"{msg} {extra}"

    raise type(module_not_found_error)(msg) from module_not_found_error
