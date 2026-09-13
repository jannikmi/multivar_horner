import os

import pytest

from multivar_horner import global_settings, helpers_fcts_numba
from multivar_horner.classes import horner_poly


@pytest.fixture(scope="session", autouse=True)
def evaluation_environment(tmp_path_factory):
    expected = os.environ.get("EXPECT_NUMBA")
    if expected is not None:
        assert helpers_fcts_numba.using_numba == (expected == "1")
    # Never reuse compiled instructions from another interpreter or architecture.
    with pytest.MonkeyPatch.context() as patch:
        cache = tmp_path_factory.mktemp("evaluation-cache")
        patch.setattr(global_settings, "PATH2CACHE", cache)
        patch.setattr(horner_poly, "PATH2CACHE", cache)
        yield
