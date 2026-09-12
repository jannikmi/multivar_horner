"""Run outside the checkout after installing a distribution into a fresh venv."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import multivar_horner
from multivar_horner import HornerMultivarPolynomial, helpers_fcts_numba


def main():
    backend = sys.argv[1]
    assert backend in {"c", "numba"}
    package_path = Path(multivar_horner.__file__).resolve()
    assert package_path.is_relative_to(Path(sys.prefix).resolve()), package_path
    assert helpers_fcts_numba.using_numba == (backend == "numba")
    if backend == "c":
        assert importlib.util.find_spec("numba") is None
    with tempfile.TemporaryDirectory() as empty_path:
        if backend == "numba":
            os.environ["PATH"] = empty_path
        poly = HornerMultivarPolynomial(
            [5.0, 1.0, 2.0, 3.0],
            [[0, 0, 0], [3, 1, 0], [2, 0, 1], [1, 1, 1]],
            rectify_input=True,
        )
        assert poly.use_c_eval == (backend == "c")
        np.testing.assert_allclose(poly(np.array([-2.0, 3.0, 1.0])), -29.0)
        x = np.array([1 + 2j, -3j, 2 - 1j])
        expected = 5 + x[0] ** 3 * x[1] + 2 * x[0] ** 2 * x[2] + 3 * x[0] * x[1] * x[2]
        np.testing.assert_allclose(
            poly.eval_complex(x), expected, rtol=1e-13, atol=1e-13
        )
        if backend == "numba":
            assert helpers_fcts_numba.eval_recipe.nopython_signatures
    print(f"Installed {package_path}: {backend} real/complex evaluation passed")


if __name__ == "__main__":
    main()
